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
        fcol_to_cond = {
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
        fcol_to_pred = {
            'Period_minutes' : 'orbital_mechanics',
            'Type_of_Orbit' : 'random_forest',
        }

        # locls = local columns modeled by CrossCat.
        lcols = [(col,stat) for (col,stat) in schema if col not in foreign]

        # Instantiate **this** generator.
        generator_id, columns = instantiate(schema)

        # Convert foriegn cols from strings to BayesDB column numbers.
        self.fcol_to_cond = {}
        self.fcol_to_pred = {}
        for f in foreign:
            f_col = bayesdb_generator_column_number(bdb, generator_id, f)
            self.fcol_to_cond[f_col] = [bayesdb_generator_column_number(bdb,
                generator_id, col) for col in foreign[f]]
            self.fcol_to_pred[f_col] = fcol_to_pred[f]

        # Convert lcols cols from strings to BayesDB column numbers.
        self.lcols = [bayesdb_generator_column_number(bdb, generator_id, col[0])
            for col in lcols]

        # Sort foriegn cols in topological order.
        self.topo = self.topolgical_sort(self.fcol_to_cond)

        # Create the internal crosscat generator.
        self.cc_name = bayeslite.core.bayesdb_generator_name(bdb,
            generator_id) + '_cc'
        ignore = ','.join(['{} IGNORE'.format(pair[0]) for pair in schema
            if pair[0] in foreign])
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
        for f_target, f_conditions in self.fcol_to_cond.items():
            # Convert column numbers to names.
            targets = \
                [(bayesdb_generator_column_name(bdb, generator_id, f_target),
                bayesdb_generator_column_stattype(bdb, generator_id,f_target))]
            conditions = \
            [(bayesdb_generator_column_name(bdb, generator_id, f_c),
                bayesdb_generator_column_stattype(bdb, generator_id, f_c))
                for f_c in f_conditions]

            # Replace the string with an actual trained FP instance.
            self.fcol_to_pred[f_target] = \
                self.fp_constructors[self.fcol_to_pred[f_target]](df, targets,
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
                colno1,):
        # Determine which object modeled colno0, colno1
        c0_foreign = colno0 not in self.fcol_to_cond
        c1_foreign = colno1 not in self.fcol_to_cond

        # If neither col is foreign, delegate to CrossCat
        if not (c0_foreign or c1_foreign):
            return self.cc.column_dependence_probability(bdb, self.cc_id,
                modelno,
                self._internal_cc_colno(bdb, generator_id, colno0),
                self._internal_cc_colno(bdb, generator_id, colno1))

        # We know at least one column is foreign.

        # If (colno0, colno1) form a (target, given) pair, then we explicitly
        # modeled them as dependent by assumption.
        # TODO: Strong assumption, what if FP determines it is not dependent
        # on one of its conditions? (ie regression w/ 0 coefficient.)
        if colno0 in self.fcol_to_cond.get(colno1) or \
                colno1 in self.fcol_to_cond.get(colno0):
            return 1

        # If one col is an FP target and another is modeled by CC, return
        # the average dependence probability of the CC col with FP's targets.
        # XXX TODO: Write out the mathematics of the theoretical justification
        # for this heuristic (not polished).
        elif (colno0_foreign and not colno1_foreign) or (colno1_foreign and not
                colno0_foreign):
            # Obtain foreign column, and its condition columns.
            foreign_col = colno0 if colno0_foreign else colno1
            cc_col = colno1 if colno0_foreign else colno0

            deps = 0.
            for fc in self._get_condition_cols(foreign_col):
                # If fc is modeled by CC we are golden.
                if fc not in self.unmodeled:
                    deps += self.cc.column_dependence_probability(bdb,
                        self.cc_id, modelno,
                        self._internal_cc_colno(bdb, generator_id, cc_col),
                        self._internal_cc_colno(bdb, generator_id, fc))
                # Otherwise fc is itself a target of the GPM, do the recursive
                # call. Will terminate iff the DAG of GPMs is acyclic.
                # TODO: Prove.
                else:
                    depth += 1  # For debugging.
                    deps += self.column_dependence_probability(bdb,
                        generator_id, modelno, cc_col, fc)
            return deps / len(self._get_condition_cols(foreign_col))

        # If both colno0 and colno1 are FP targets AND neither is a condition
        # of the other, we compute the average pairwise dependence probability
        # of all their targets.
        elif colno0_foreign and colno1_foreign:
            assert self._get_condition_cols(colno0) != self.get



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
        # Delegate
        cc_colnos = self._internal_cc_colnos(bdb, generator_id, colnos)
        return self.cc.simulate(bdb, self.cc_id, modelno, constraints, cc_colnos,
            numpredictions=numpredictions)

    # TODO: Conver to SQL table queries on bayesdb_fp_composer_cc_lookup.
    def _internal_cc_colno(self, bdb, generator_id, colno):
        return self._internal_cc_colnos(bdb, generator_id, colno)[0]

    def _internal_cc_colnos(self, bdb, generator_id, colnos):
        # First get the names from this generator.
        colnames = [bayeslite.core.bayesdb_generator_column_name(bdb,
            generator_id, no) for no in colnos]
        # Now do the inverse mapping.
        return [bayeslite.core.bayesdb_generator_column_number(bdb, self.cc_id,
            name) for name in colnames]

    # TODO: Move this to somewhere reasonable?
    def topolgical_sort(self, graph):
        """Topologically sort a graph represented as an adjacency list.

        Parameters
        ----------
        graph : dict or list
            Adjacency list or dict representing the graph, for example
                graph_d = {10: [8, 7], 5: [8, 7, 9, 10, 11, 13, 15]}
                graph_l = [(10, [8, 7]), (5, [8, 7, 9, 10, 11, 13, 15])]

        Returns
        -------
        graph_sorted : list
            The original graph, in topologically sorted order.
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
