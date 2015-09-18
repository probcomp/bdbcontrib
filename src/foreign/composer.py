# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math

from bayeslite.core import *
import bayeslite.metamodel
import bayeslite.bqlfn

import bdbcontrib
from bdbcontrib.foreign.orbital_mech import OrbitalMechanics
from bdbcontrib.foreign.random_forest import RandomForest

class Composer(bayeslite.metamodel.IBayesDBMetamodel):

    """A metamodel which composes foreign predictors with CrossCat. The
    dependency of the variables must form a directed acyclic graph.
    """

    def __init__(self):
        self.cc_id = None
        self.cc = None

    def name(self):
        """Return the name of the metamodel as a str."""
        return 'fp_composer'

    def register(self, bdb):
        # XXX Figure out what to do.
        # XXX Causes serialization problem, should have v0 of schema which
        # is only ever before the first time this metamodel is registered
        # in the history of the bdb.
        bdb.sql_execute("""
            INSERT INTO bayesdb_metamodel (name, version)
                VALUES ('fp_composer', 1);
            """)
        bdb.sql_execute("""
            CREATE TABLE bayesdb_fp_composer_cc_lookup (
                generator_id INTEGER NOT NULL PRIMARY KEY
                        REFERENCES bayesdb_generator(id),
                crosscat_generator_id INTEGER NOT NULL
                        REFERENCES bayesdb_generator(id)
            );
            """)

    def create_generator(self, bdb, table, schema, instantiate):
        # TODO: register foreign predictors externally.
        # TODO: Serialize the mapping from FP names to objects into database.
        self.fp_constructors = {'random_forest': RandomForest,
            'orbital_mechanics' : OrbitalMechanics}

        # TODO: Parse the local, foreign, and condition columns
        # and (foreign predictors) all from schema.
        schema = [
            ('Country_of_Operator', 'CATEGORICAL'),
            ('Operator_Owner', 'CATEGORICAL'),
            ('Users', 'CATEGORICAL'),
            ('Purpose', 'CATEGORICAL'),
            ('Type_of_Orbit', 'CATEGORICAL'),
            ('Perigee_km', 'NUMERICAL'),
            ('Apogee_km', 'NUMERICAL'),
            ('Eccentricity', 'NUMERICAL'),
            ('Launch_Mass_kg', 'NUMERICAL'),
            ('Dry_Mass_kg', 'NUMERICAL'),
            ('Power_watts', 'NUMERICAL'),
            ('Date_of_Launch', 'NUMERICAL'),
            ('Anticipated_Lifetime', 'NUMERICAL'),
            ('Contractor', 'CATEGORICAL'),
            ('Country_of_Contractor', 'CATEGORICAL'),
            ('Launch_Site', 'CATEGORICAL'),
            ('Launch_Vehicle', 'CATEGORICAL'),
            ('Source_Used_for_Orbital_Data', 'CATEGORICAL'),
            ('longitude_radians_of_geo', 'NUMERICAL'),
            ('Inclination_radians', 'NUMERICAL'),
            ('Class_of_orbit', 'CATEGORICAL'),
            ('Period_minutes', 'NUMERICAL')
        ]

        # Maps FP target columns (fcol) to their conditions columns .
        # Currenlty strings, but will be mapped to columns after instantiate.
        fcols = {
            # Orbital Mechanics
            'Period_minutes' : ['Apogee_km', 'Perigee_km'],
            # Random Forest
            'Type_of_Orbit' : ['Apogee_km', 'Perigee_km', 'Eccentricity',
                'Period_minutes', 'Launch_Mass_kg', 'Power_watts',
                'Anticipated_Lifetime', 'Class_of_orbit']
        }
        # Maps target FP columns to FP object. Currently these are strings
        # but will be mapped to actual instances when the initialize_models
        # is called.
        fp_lookup = {
            'Period_minutes' : 'orbital_mechanics',
            'Type_of_Orbit' : 'random_forest',
        }

        # locls = local columns modeled by CrossCat.
        lcols = [(col,stat) for (col,stat) in schema if col not in fcols]

        # Instantiate **this** generator.
        generator_id, columns = instantiate(schema)

        # Convert all strings to to BayesDB column numbers.
        self.fcols = {}
        self.fp_lookup = {}
        for f in fcols:
            fcolno = bayesdb_generator_column_number(bdb, generator_id, f)
            self.fcols[fcolno] = [bayesdb_generator_column_number(bdb,
                generator_id, col) for col in fcols[f]]
            self.fp_lookup[fcolno] = fp_lookup[f]

        # Convert lcols from strings to BayesDB column numbers.
        self.lcols = [bayesdb_generator_column_number(bdb, generator_id, col[0])
            for col in lcols]

        # Sort fcols in topological order.
        self.topo = self.topolgical_sort(self.fcols)

        # Create internal crosscat generator.
        self.cc_name = bayeslite.core.bayesdb_generator_name(bdb,
            generator_id) + '_cc'
        ignore = ','.join(['{} IGNORE'.format(pair[0]) for pair in schema
            if pair[0] in fcols])
        cc = ','.join(['{} {}'.format(pair[0], pair[1]) for pair in lcols])
        bql = """
            CREATE GENERATOR {} FOR satellites USING crosscat(
                GUESS(*),
                {} ,
                {}
            );
        """.format(self.cc_name, ignore, cc)
        bdb.execute(bql)

        # Obtain the generator_id of internal cc, and the metamodel object.
        # Should be stored in the database table bayesdb_fp_composer_cc_lookup
        self.cc_id = bayesdb_get_generator(bdb, self.cc_name)
        self.cc = bayesdb_generator_metamodel(bdb, self.cc_id)

    def drop_generator(self, bdb, generator_id):
        # Drop the internal crosscat instance.
        bdb.execute('''DROP GENERATOR {}'''.format(self.cc_name))
        # XXX TODO: Additional clean up operations.

    def initialize_models(self, bdb, generator_id, modelnos, model_config):
        # Initialize internal Crosscat. If k models of Composer are instantiated
        # then k internal CC models will be created, with a 1-1 mapping.
        bql = """
            INITIALIZE {} MODELS FOR {};
            """.format(len(modelnos), self.cc_name)
        bdb.execute(bql)

        # Obtain the dataframe for foreign predictors.
        df = bdbcontrib.cursor_to_df(bdb.execute('''
            SELECT * FROM {}
            '''.format(bayesdb_generator_table(bdb, generator_id))))

        # Initialize the foriegn predictors.
        for f_target, f_conditions in self.fcols.items():
            # Convert column numbers to names.
            targets = \
                [(bayesdb_generator_column_name(bdb, generator_id, f_target),
                bayesdb_generator_column_stattype(bdb, generator_id,f_target))]
            conditions = \
            [(bayesdb_generator_column_name(bdb, generator_id, f_c),
                bayesdb_generator_column_stattype(bdb, generator_id, f_c))
                for f_c in f_conditions]

            # Replace the string with an actual trained FP instance.
            self.fp_lookup[f_target] = \
                self.fp_constructors[self.fp_lookup[f_target]](df, targets,
                conditions)

    def drop_models(self, bdb, generator_id, modelnos=None):
        # Delegate
        raise NotImplementedError()

    def analyze_models(self, bdb, generator_id, modelnos=None, iterations=1,
                max_seconds=None, ckpt_iterations=None, ckpt_seconds=None):
        # Delegate
        self.cc.analyze_models(bdb, self.cc_id, modelnos=modelnos,
            iterations=iterations, max_seconds=max_seconds,
            ckpt_iterations=ckpt_iterations, ckpt_seconds=ckpt_seconds)

    def column_dependence_probability(self, bdb, generator_id, modelno, colno0,
                colno1, recurse=False):
        if modelno is None:
            n_model = 1
            while bayesdb_generator_has_model(bdb, generator_id, n_model):
                n_model += 1
            p = sum(self._column_dependence_probability(bdb, generator_id,
                m, colno0, colno1) for m in xrange(n_model)) / float(n_model)
        else:
            p = self._column_dependence_probability(bdb, generator_id,
                modelno, colno0, colno1)
        return p

    # XXX Computes the dependence probability for a single model. Find
    # a better way to aggregate externally.
    def _column_dependence_probability(self, bdb, generator_id, modelno,
            colno0, colno1):
        if modelno is None:
            raise ValueError('Invalid modelno argument for internal.')
        # Trivial case.
        if colno0 == colno1:
            return 1

        # Determine which object modeled colno0, colno1
        c0_foreign = colno0 in self.fcols
        c1_foreign = colno1 in self.fcols

        # If neither col is foreign, delegate to CrossCat.
        if not (c0_foreign or c1_foreign):
            return self.cc.column_dependence_probability(bdb,
                self.cc_id, modelno,
                self._internal_cc_colno(bdb, generator_id, colno0),
                self._internal_cc_colno(bdb, generator_id, colno1))

        # If (colno0, colno1) form a (target, given) pair, then we
        # explicitly modeled them as dependent by assumption.
        # TODO: Strong assumption? What if FP determines it is not
        # dependent on one of its conditions? (ie 0 coeff in regression)
        if colno0 in self.fcols.get(colno1, []) or \
                colno1 in self.fcols.get(colno0, []):
            return 1

        # [col0 FP target], [col1 CC], [all conditions of col0 IND col1]
        # IMPLIES [col0 IND col1].
        # TODO : Write the mathematical proof of this statement (long, easy).
        # XXX The reverse is not true generally (counterxample), but we shall
        # assume an IFF condition. This assumption is not unlike the transitive
        # closure property of independence in CrossCat.
        if (c0_foreign and not c1_foreign) or \
                (c1_foreign and not c0_foreign):
            fcol = colno0 if c0_foreign else colno1
            lcol = colno1 if c0_foreign else colno0
            return any(self._column_dependence_probability(bdb, generator_id,
                modelno, cond, lcol)
                for cond in self.fcols[fcol])

        # XXX Both columns are foreign. (Recursively) return 1 if any of their
        # conditions (possibly FPs) have dependencies.
        assert c0_foreign and c1_foreign
        return any(self._column_dependence_probability(bdb, generator_id,
            modelno, cond0, cond1)
            for cond0 in self.fcols[colno0]
            for cond1 in self.fcols[colno1])

    def column_mutual_information(self, bdb, generator_id, modelno, colno0,
                colno1, numsamples=100):
        # Delegate
        cc_colnos = self._internal_cc_colnos(bdb, generator_id,
            [colno0, colno1])
        return self.cc.column_mutual_information(bdb, self.cc_id, modelno,
            cc_colnos[0], cc_colnos[1], numsamples=numsamples)

    def column_value_probability(self, bdb, generator_id, modelno, colno,
                value, constraints):
        # Delegate
        cc_colno = self._internal_cc_colno(bdb, generator_id, colno)
        return self.cc.column_value_probability(bdb, self.cc_id, modelno,
            cc_colno, value, constraints)

    def row_similarity(self, bdb, generator_id, modelno, rowid, target_rowid,
                colnos):
        # Delegate
        cc_colnos = self._internal_cc_colnos(bdb, generator_id, colnos)
        return self.cc.row_similarity(bdb, self.cc_id, modelno, rowid,
            target_rowid, cc_colnos)

    def row_column_predictive_probability(self, bdb, generator_id, modelno,
                rowid, colno):
        # Delegate
        cc_colno = self._internal_cc_colno(bdb, generator_id, colno)
        return self.cc.row_column_predictive_probability(bdb, self.cc_id,
            modelno, rowid, cc_colno)

    def predict_confidence(self, bdb, generator_id, modelno, colno, rowid,
                numsamples=None):
        # Delegate
        cc_colno = self._internal_cc_colno(bdb, generator_id, colno)
        return self.predict_confidence(bdb, self.cc_id, modelno, cc_colno,
            rowid, numsamples=numsamples)

    def simulate(self, bdb, generator_id, modelno, constraints, colnos,
                numpredictions=1):
        # XXX This would be much easier written as a VentureScript program.
        # One SP for CC, an SP for each FP, and the composer would be a trivial?
        # VS program (issue: logpdf queries).

        # Small optimization; all cols are purely local, just delegate to CC.
        all_cols = colnos + [p[0] for p in constraints]
        if all(f not in all_cols for f in self.fcols):
            cc_constraints = [(self._internal_cc_colno(bdb, generator_id, c), v)
                for c, v in constraints]
            cc_targets = self._internal_cc_colnos(bdb, generator_id, colnos)
            sample = self.cc.simulate(bdb, generator_id, modelno,
                cc_constraints, cc_targets, numpredictions = numpredictions)

        # Likelihood weighting by forward sampling in the topo-sorted DAG.
        samples, weights = self,_forward_weighted_sample

        print 'wow I reached here...'

    def _forward_weighted_sample(self, bdb, generator_id, modelno, Y,
            ll_samples=None):
        """Generates a pair (sample, weight): sample is a joint sample of all
        the targ cols: and weight is the likelihood given the evidence
        in constraint cols. ll_samples is the number of weight samples to
        return, which must drawn from a categorical by the caller."""

        # XXX We have this problem over and over...
        if ll_samples is None:
            ll_samples = 10

        # Create  #ll_sample dicts; S[k] is the kth sample from full joint.
        S = [{c:v for (c,v) in Y} for _ in xrange(ll_samples)]

        # Sample any missing lcols (conditioned on lcol evidence only).
        cc_constraints = [(c, v) for c,v in S if c in self.lcols]
        cc_targets = [c for c in self.lcols if c not in S[0]]
        cc_sample = self.cc.simulate(bdb, generator_id, modelno, cc_constraints,
            cc_targets, numpredictions=ll_samples)

        weights = []
        for k in xrange(len(ll_samples)):
            # Add simulated lcols.
            S[k].update({c:v for c,v in zip(cc_targets, cc_sample[k])})
            w = 0
            for f in self.topo:
                # Assert all parents of FP are known (sampled or evidence).
                # Otherwise we are KO.
                assert set(self.fcols[f]).issubset(set(S[k]))
                # XXX TODO: Figure out a better way to communicate with FP.
                # Convert all column numbers to names ...
                fp_conditions = {}
                for q in S[k]:
                    fp_conditions[bayesdb_generator_column_name(bdb, generator_id,
                        q[0])] = S[k][q]
                # f is evidence, compute likelihood weight.
                if f in S[k]:
                    w += self.fp_lookup[f].logpdf(S[k][f], fp_conditions)
                # Sample from conditional distribution.
                else:
                    fs = self.fp_lookup[f].simulate(1, fp_conditions)
                    S[k][f] = fs

            weights.append(w)

        import ipdb; ipdb.set_trace()
        return S, weights

    # TODO: Convert to SQL table queries on bayesdb_fp_composer_cc_lookup.
    def _internal_cc_colno(self, bdb, generator_id, colno):
        return self._internal_cc_colnos(bdb, generator_id, [colno])[0]

    def _internal_cc_colnos(self, bdb, generator_id, colnos):
        # First get the names from this generator.
        colnames = [bayeslite.core.bayesdb_generator_column_name(bdb,
            generator_id, no) for no in colnos]
        # Now do the inverse mapping.
        return [bayeslite.core.bayesdb_generator_column_number(bdb, self.cc_id,
            name) for name in colnames]

    # TODO migrate to a reasonable place (ie sample_utils in CrossCat).
    def _cc_joint_logpdf(self, bdb, generator_id, modelno, Q, Y):
        # Computes the joint probability P(Q|Y) where Q =[(col, val)] are the
        # query columns, and Y = [(col, val)] are the constraint columns.

        # Validate inputs and map to internal cc columns.
        Qi = []
        for (col, val) in Q:
            if col not in self.lcols:
                raise ValueError('_cc_joint_probability called with a foreign '
                    ' column.')
            Qi.append((self._internal_cc_colno(bdb, generator_id, col), val))

        Yi = []
        for (col, val) in Y:
            if col not in self.lcols:
                raise ValueError('_cc_joint_probability called with a foreign '
                    ' column.')
            Yi.append((self._internal_cc_colno(bdb, generator_id, col), val))

        # Compute joint via the the chain rule: if Q is a vector with density 0
        # under the joint then return -float('inf').
        prob = 0
        for (col, val) in Qi:
            r = self.cc.column_value_probability(bdb, self.cc_id, modelno, col,
                val, Yi)
            if r == 0:
                return -float('inf')
            prob += math.log(r)
            Yi.append((col,val))

        return prob

    # TODO: Move this to somewhere reasonable?
    def topolgical_sort(self, graph):
        """Topologically sort a directed graph represented as an adjacency list.
        Assumes that edges are incoming, ie (10:[8,7]) means 8->10 and 7->10.

        Parameters
        ----------
        graph : dict or list
            Adjacency list or dict representing the graph, for example
                graph_d = {10: [8, 7], 5: [8, 7, 9, 10, 11, 13, 15]}
                graph_l = [(10, [8, 7]), (5, [8, 7, 9, 10, 11, 13, 15])]

        Returns
        -------
        graph_sorted : list
            An adjacency list, where the order of the nodes is listed
            in topological order.
        """
        graph_sorted = []
        graph = dict(graph)

        # Run until the unsorted graph is empty.
        while graph:
            acyclic = False
            for node, edges in graph.items():
                for edge in edges:
                    if edge in graph:
                        break
                else:
                    acyclic = True
                    del graph[node]
                    graph_sorted.append((node, edges))

            if not acyclic:
                raise RuntimeError('A cyclic dependency occurred in '
                    'topological_sort.')

        return graph_sorted
