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

import itertools
import importlib
import math

import numpy as np

from bayeslite.core import *
from bayeslite.exception import BQLError
from bayeslite.sqlite3_util import sqlite3_quote_name as quote
from bayeslite.util import logsumexp, logmeanexp

from crosscat.utils import sample_utils as su

import bayeslite.metamodel
import bayeslite.bqlfn

import bdbcontrib
from bdbcontrib.foreign.keplers_law import KeplersLaw
from bdbcontrib.foreign.random_forest import RandomForest

composer_schema_1 = '''
INSERT INTO bayesdb_metamodel
    (name, version) VALUES ('composer', 0);
@
CREATE TABLE bayesdb_composer_cc_id(
    generator_id INTEGER NOT NULL
        REFERENCES bayesdb_generator(id),
    crosscat_generator_id INTEGER NOT NULL
        REFERENCES bayesdb_generator(id),

    PRIMARY KEY(generator_id, crosscat_generator_id),
    CHECK (generator_id != crosscat_generator_id)
);
@
CREATE TABLE bayesdb_composer_column_owner(
    generator_id INTEGER NOT NULL REFERENCES bayesdb_generator(id),
    colno INTEGER NOT NULL,
    local BOOLEAN NOT NULL,

    PRIMARY KEY(generator_id, colno, local),
    FOREIGN KEY(generator_id, colno)
        REFERENCES bayesdb_generator_column(generator_id, colno)
);
@
CREATE TABLE bayesdb_composer_column_toposort(
    generator_id INTEGER NOT NULL REFERENCES bayesdb_generator(id),
    colno INTEGER NOT NULL,
        -- Cannot express check constraint with subquery in sqlite.
        -- CHECK (EXISTS(SELECT generator_id, colno
        --      FROM bayesdb_composer_column_owner WHERE local = FALSE)),
    position INTEGER NOT NULL,
        CHECK (0 <= position)

    PRIMARY KEY(generator_id, colno),
    FOREIGN KEY(generator_id, colno)
        REFERENCES bayesdb_generator_column(generator_id, colno),
    UNIQUE (generator_id, position)
);
@
CREATE TRIGGER bayesdb_composer_column_toposort_check
    BEFORE INSERT ON bayesdb_composer_column_toposort
BEGIN
    SELECT CASE
        WHEN (NOT EXISTS(SELECT generator_id, colno, local
                FROM bayesdb_composer_column_owner
                WHERE generator_id=NEW.generator_id AND colno=NEW.colno AND
                    local = 0))
        THEN RAISE(ABORT, 'Columns in bayesdb_composer_column_toposort
            must be foreign.')
    END;
END;
@
CREATE TABLE bayesdb_composer_column_parents(
    generator_id INTEGER NOT NULL REFERENCES bayesdb_generator(id),
    fcolno INTEGER NOT NULL,
    pcolno BOOLEAN NOT NULL,

    PRIMARY KEY(generator_id, fcolno, pcolno),
    FOREIGN KEY(generator_id, fcolno)
        REFERENCES bayesdb_generator_column(generator_id, colno),
    FOREIGN KEY(generator_id, pcolno)
        REFERENCES bayesdb_generator_column(generator_id, colno),
    CHECK (fcolno != pcolno)
);
@
CREATE TABLE bayesdb_composer_column_foreign_predictor(
    generator_id INTEGER NOT NULL REFERENCES bayesdb_generator(id),
    colno INTEGER NOT NULL,
    predictor_name TEXT COLLATE NOCASE NOT NULL,
    predictor_binary BLOB,

    PRIMARY KEY(generator_id, colno),
    FOREIGN KEY(generator_id, colno)
        REFERENCES bayesdb_generator_column(generator_id, colno)
);
@
CREATE TRIGGER bayesdb_composer_column_foreign_predictor_check
    BEFORE INSERT ON bayesdb_composer_column_foreign_predictor
BEGIN
    SELECT CASE
        WHEN (NOT EXISTS(SELECT generator_id, colno, local
                FROM bayesdb_composer_column_owner
                WHERE generator_id=NEW.generator_id AND colno=NEW.colno AND
                    local = 0))
        THEN RAISE(ABORT, 'Columns in bayesdb_composer_foreign_predictor
            must be foreign.')
    END;
END;
'''

class Composer(bayeslite.metamodel.IBayesDBMetamodel):
    """A metamodel which composes foreign predictors with CrossCat.
    """

    def __init__(self):
        # In-memory map of registered foreign predictor builders.
        self.fp_builders = {}
        self.fp_cache = {}

    def register_fp_builder(self, fp_builder):
        """Register an object which builds a foreign predictor. The `fp_builder`
        must have the methods:

        - create(df, targets, conditions)
            Returns a new foreign predictor, typically by calling its `train`
            method (see `IBayesDBForeignPredictor`).

        - serialize(predictor)
            Returns the binary represenation of `predictor`.

        - deserialize(binary)
            Returns the foreign predictor from its `binary` representation.

        - name()
            Returns the name of the fp_builder.

        Foreign predictor builders must be registered each time the database is
        launched.
        """
        # Validate the fp_builder.
        assert hasattr(fp_builder, 'create')
        assert hasattr(fp_builder, 'serialize')
        assert hasattr(fp_builder, 'deserialize')
        assert hasattr(fp_builder, 'name')
        # Check for duplicates.
        if fp_builder.name() in self.fp_builders:
            raise ValueError('A foreign predictor with name {} has already '
                'been registered. Currently registered: {}'.format(
                    fp_builder.name(), self.fp_builders))
        self.fp_builders[fp_builder.name()] = fp_builder

    def name(self):
        return 'composer'

    def register(self, bdb):
        cursor = bdb.sql_execute('''
            SELECT version FROM bayesdb_metamodel WHERE name = ?;
        ''', (self.name(),))
        version = None
        try:
            row = cursor.next()
        except StopIteration:
            version = 0
        else:
            version = row[0]
        assert version is not None
        if version == 0:
            # XXX `@` delimiter since `;` breaks TRIGGER.
            with bdb.savepoint():
                for stmt in composer_schema_1.split('@'):
                    bdb.sql_execute(stmt)
        return

    def create_generator(self, bdb, table, schema, instantiate):
        # BEGIN SCHEMA PARSE
        schema = [
            ('Country_of_Operator', 'CATEGORICAL'),
            ('Operator_Owner', 'CATEGORICAL'),
            ('Users', 'CATEGORICAL'),
            ('Purpose', 'CATEGORICAL'),
            ('Class_of_orbit', 'CATEGORICAL'),
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
            ('Type_of_Orbit', 'CATEGORICAL'),
            ('Period_minutes', 'NUMERICAL')
        ]
        # Maps target FP column to name of its predictor.
        fcolname_to_fpname = {
            'Type_of_Orbit' : 'random_forest',
            'Period_minutes' : 'keplers_law'
        }
        # Maps FP column to their conditions columns .
        fcolname_to_pcolname = {
            'Period_minutes' : ['Apogee_km', 'Perigee_km'],
            'Type_of_Orbit' : ['Apogee_km', 'Perigee_km', 'Eccentricity',
                'Period_minutes', 'Launch_Mass_kg', 'Power_watts',
                'Anticipated_Lifetime', 'Class_of_orbit']
        }
        # lcols are local columns modeled by CrossCat.
        lcol_schema = [(col,stat) for (col,stat) in schema if col not in
                fcolname_to_pcolname]
        # END SCHEMA PARSE
        # Instantiate **this** generator.
        genid, columns = instantiate(schema)
        # Convert all strings to to BayesDB column numbers.
        lcols = [bayesdb_generator_column_number(bdb, genid, col[0])
            for col in lcol_schema]
        fcolno_to_pcolnos = {}
        for f in fcolname_to_pcolname:
            fcolno = bayesdb_generator_column_number(bdb, genid, f)
            fcolno_to_pcolnos[fcolno] = [bayesdb_generator_column_number(bdb,
                genid, col) for col in fcolname_to_pcolname[f]]
        # Create internal crosscat generator.
        SUFFIX = '_cc'
        # cc_name = bayeslite.core.bayesdb_generator_name(bdb, genid) + SUFFIX
        cc_name = 'satcc'
        # ignore = ','.join(['{} IGNORE'.format(pair[0]) for pair in schema
        #     if pair[0] in fcolno_to_pcolnos])
        # cc_cols = ','.join(['{} {}'.format(pair[0], pair[1]) for pair in
        #     lcol_schema])
        # bql = """
        #     CREATE GENERATOR {} FOR satellites USING crosscat(
        #         GUESS(*), Name IGNORE, {} , {}
        #     );
        # """.format(cc_name, ignore, cc_cols)
        # bdb.execute(bql)
        # Save lcols/fcolnos.
        with bdb.savepoint():
            for colno, _, _ in columns:
                local = colno not in fcolno_to_pcolnos
                bdb.sql_execute('''
                    INSERT INTO bayesdb_composer_column_owner
                        (generator_id, colno, local) VALUES (?,?,?)
                ''', (genid, colno, int(local),))
            # Save parents of foreign columns.
            for fcolno in fcolno_to_pcolnos:
                for pcolno in fcolno_to_pcolnos[fcolno]:
                    bdb.sql_execute('''
                        INSERT INTO bayesdb_composer_column_parents
                            (generator_id, fcolno, pcolno) VALUES (?,?,?)
                    ''', (genid, fcolno, pcolno,))
            # Save topological order.
            topo = self.topolgical_sort(fcolno_to_pcolnos)
            position = 0
            for colno, _ in topo:
                bdb.sql_execute('''
                    INSERT INTO bayesdb_composer_column_toposort
                        (generator_id, colno, position) VALUES (?,?,?)
                    ''', (genid, colno, position,))
                position += 1
            # Save internal cc generator id.
            bdb.sql_execute('''
                INSERT INTO bayesdb_composer_cc_id
                    (generator_id, crosscat_generator_id) VALUES (?,?)
            ''', (genid, bayesdb_get_generator(bdb, cc_name),))
            # Save predictor names of foreign columns.
            for fcolno in fcolno_to_pcolnos:
                fp_name = fcolname_to_fpname[bayesdb_generator_column_name(bdb,
                    genid, fcolno)]
                bdb.sql_execute('''
                    INSERT INTO bayesdb_composer_column_foreign_predictor
                        (generator_id, colno, predictor_name) VALUES (?,?,?)
                ''', (genid, fcolno, fp_name))

    def drop_generator(self, bdb, genid):
        raise NotImplementedError('Composer generators cannot be dropped. '
            'Feature coming soon.')

    def initialize_models(self, bdb, genid, modelnos, model_config):
        # Initialize internal crosscat. If k models of composer are instantiated
        # then k internal CC models will be created (1-1 mapping).
        # bql = """
        #     INITIALIZE {} MODELS FOR {};
        #     """.format(len(modelnos),
        #             bayesdb_generator_name(bdb, self.get_cc_id(bdb, genid)))
        # bdb.execute(bql)
        # Obtain the dataframe for foreign predictors.
        df = bdbcontrib.cursor_to_df(bdb.execute('''
                SELECT * FROM {}
                '''.format(bayesdb_generator_table(bdb, genid))))
        # Initialize the foriegn predictors.
        for fcol in self.get_fcols(bdb, genid):
            # Convert column numbers to names.
            targets = \
                [(bayesdb_generator_column_name(bdb, genid, fcol),
                    bayesdb_generator_column_stattype(bdb, genid, fcol))]
            conditions = \
                [(bayesdb_generator_column_name(bdb, genid, pcol),
                    bayesdb_generator_column_stattype(bdb, genid, pcol))
                    for pcol in self.get_pcols(bdb, genid, fcol)]
            # Initialize the foreign predictor.
            predictor_name = self.get_predictor_name(bdb, genid, fcol)
            builder = self.fp_builders[predictor_name]
            predictor = builder.create(df, targets, conditions)
            # Store in the database.
            with bdb.savepoint():
                sql = '''
                    UPDATE bayesdb_composer_column_foreign_predictor SET
                        predictor_binary = :predictor_binary
                        WHERE generator_id = :genid AND colno = :colno
                '''
                predictor_binary = builder.serialize(predictor)
                bdb.sql_execute(sql, {
                    'genid': genid,
                    'predictor_binary': predictor_binary,
                    'colno': fcol
                })

    def drop_models(self, bdb, genid, modelnos=None):
        raise NotImplementedError('Composer generator models cannot be '
            'dropped. Feature coming soon.')

    def analyze_models(self, bdb, genid, modelnos=None, iterations=1,
                max_seconds=None, ckpt_iterations=None, ckpt_seconds=None):
        # XXX Composer currently does not perform joint inference.
        # (Need full GPM interface, active research project).
        self.get_cc(bdb, genid).analyze_models(bdb, self.get_cc_id(bdb, genid),
            modelnos=modelnos, iterations=iterations, max_seconds=max_seconds,
            ckpt_iterations=ckpt_iterations, ckpt_seconds=ckpt_seconds)

    def column_dependence_probability(self, bdb, genid, modelno, colno0,
                colno1):
        # XXX Aggregator only.
        if modelno is None:
            n_model = 1
            while bayesdb_generator_has_model(bdb, genid, n_model):
                n_model += 1
            p = sum(self._column_dependence_probability(bdb, genid, m, colno0,
                colno1) for m in xrange(n_model)) / float(n_model)
        else:
            p = self._column_dependence_probability(bdb, genid, modelno, colno0,
                colno1)
        return p

    def _column_dependence_probability(self, bdb, genid, modelno, colno0,
            colno1):
        # XXX Computes the dependence probability for a single model.
        if modelno is None:
            raise ValueError('Invalid modelno argument for '
                'internal _column_dependence_probability. An integer modelno '
                'is required, not None.')
        # Trivial case.
        if colno0 == colno1:
            return 1
        fcols = set(self.get_fcols(bdb, genid))
        c0_foreign = colno0 in fcols
        c1_foreign = colno1 in fcols
        # Neither col is foreign, delegate to CrossCat.
        # XXX Fails for future implementation of conditional dependence.
        if not (c0_foreign or c1_foreign):
            return self.get_cc(bdb, genid).column_dependence_probability(bdb,
                self.get_cc_id(bdb, genid), modelno,
                self.get_cc_colno(bdb, genid, colno0),
                self.get_cc_colno(bdb, genid, colno1))
        # (colno0, colno1) form a (target, given) pair.
        # WE explicitly modeled them as dependent by assumption.
        # TODO: Strong assumption? What if FP determines it is not
        # dependent on one of its conditions? (ie 0 coeff in regression)
        if colno0 in self.get_pcols(bdb, genid, colno1) or \
                colno1 in self.get_pcols(bdb, genid, colno0):
            return 1
        # (colno0, colno1) form a local, foreign pair.
        # IF [col0 FP target], [col1 CC], and [all conditions of col0 IND col1]
        #   then [col0 IND col1].
        # XXX Reverse is not true generally (counterxample), but we shall
        # assume an IFF condition. This assumption is not unlike the transitive
        # closure property of independence in crosscat.
        if (c0_foreign and not c1_foreign) or \
                (c1_foreign and not c0_foreign):
            fcol = colno0 if c0_foreign else colno1
            lcol = colno1 if c0_foreign else colno0
            return any(self._column_dependence_probability(bdb, genid, modelno,
                pcol, lcol) for pcol in self.get_pcols(bdb, genid, fcol))
        # XXX TODO: Determine independence semantics for this case.
        # Both columns are foreign. (Recursively) return 1 if any of their
        # conditions (possibly FPs) have dependencies.
        assert c0_foreign and c1_foreign
        return any(self._column_dependence_probability(bdb, genid, modelno,
                pcol0, pcol1) for pcol0 in self.get_pcols(bdb, genid, colno0)
                for pcol1 in self.get_pcols(bdb, genid, colno1))

    def column_mutual_information(self, bdb, genid, modelno, colno0, colno1,
            numsamples=100):
        # TODO: Allow conditional mutual information.
        # Two cc columns, delegate.
        lcols = self.get_lcols(bdb, genid)
        if colno0 in lcols and colno1 in lcols:
            cc_colnos = self.get_cc_colnos(bdb, genid, [colno0, colno1])
            return self.get_cc(bdb, genid).column_mutual_information(bdb,
                self.get_cc_id(bdb, genid), modelno, cc_colnos[0], cc_colnos[1],
                numsamples=numsamples)
        # Simple Monte Carlo by simulating from the joint
        # distributions, and computing the joint and marginal densities.
        Q, Y = [colno0, colno1], []
        samples = self.simulate(bdb, genid, modelno, Y, Q,
            numpredictions=numsamples)
        mi = logpx = logpy = logpxy = 0
        for s in samples:
            Qx = [(colno0, s[0])]
            logpx = self._joint_logpdf(bdb, genid, modelno, Qx, [])
            Qy = [(colno1, s[1])]
            logpy = self._joint_logpdf(bdb, genid, modelno, Qy, [])
            Qxy = Qx+Qy
            logpxy = self._joint_logpdf(bdb, genid, modelno, Qxy, [])
            mi += logpx - logpx - logpy
        # TODO: Use linfoot?
        return mi

    def column_value_probability(self, bdb, genid, modelno, colno, value,
            constraints):
        # XXX Aggregator only.
        p = []
        if modelno is None:
            n_model = 0
            while bayesdb_generator_has_model(bdb, genid, n_model):
                p.append(self._joint_logpdf(bdb, genid, n_model,
                    [(colno, value)], constraints))
                n_model += 1
            p = logmeanexp(p)
        else:
            p = self._joint_logpdf(bdb, genid, modelno, [(colno, value)],
                constraints)
        return math.exp(p)

    def _joint_logpdf(self, bdb, genid, modelno, Q, Y, n_samples=None):
        # XXX Computes the joint probability of query Q given evidence Y
        # for a single model. The function is a likelihood weighted
        # integrator.
        if modelno is None:
            raise ValueError('Invalid modelno argument for '
                'internal _joint_logpdf. An integer modelno '
                'is required, not None.')
        if n_samples is None:
            n_samples = 200
        # (Q,Y) marginal joint density.
        joint_samples, joint_weights = self._weighted_sample(bdb, genid,
            modelno, Q+Y, n_samples=n_samples)
        # Y marginal density.
        evidence_samples, evidence_weights = self._weighted_sample(bdb,
            genid, modelno, Q+Y, n_samples=n_samples)
        # XXX TODO Keep sampling until logpQY <= logpY
        logpQY = logmeanexp(joint_weights)
        logpY = logmeanexp(evidence_weights)
        return logpQY - logpY

    def predict_confidence(self, bdb, genid, modelno, colno, rowid,
            numsamples=None):
        # Predicts a value for the cell [rowid, colno] with a confidence metric.
        # XXX Prefer accuracy over speed for imputation.
        if numsamples is None:
            numsamples = 50
        # Obtain all values for all other columns.
        colnos = bayesdb_generator_column_numbers(bdb, genid)
        colnames = bayesdb_generator_column_names(bdb, genid)
        table = bayesdb_generator_table(bdb, genid)
        sql = '''
            SELECT {} FROM {} WHERE _rowid_ = ?
        '''.format(','.join(map(quote, colnames)), quote(table))
        cursor = bdb.sql_execute(sql, (rowid,))
        row = None
        try:
            row = cursor.next()
        except StopIteration:
            generator = bayesdb_generator_table(bdb, genid)
            raise BQLError(bdb, 'No such row in table {}'
                ' for generator {}: {}'.format(table, generator, rowid))
        # Account for multiple imputations if imputing parents.
        parent_conf = 1
        # Predicting lcol.
        if colno in self.get_lcols(bdb, genid):
            # Delegate to CC IFF
            # (lcol has no children OR all its children are None).
            children = [f for f in self.get_fcols(bdb, genid) if colno in
                    self.get_pcols(bdb, genid, f)]
            if len(children) == 0 or \
                    all(row[i] is None for i in xrange(len(row)) if i+1
                        in children):
                return self.get_cc(bdb, genid).predict_confidence(bdb,
                        self.get_cc_id(bdb, genid), modelno,
                        self.get_cc_colno(bdb, genid, colno), rowid)
            else:
                # Obtain likelihood weighted samples from posterior.
                Q = [colno]
                Y = [(c,v) for c,v in zip(colnos, row) if c != colno and v
                        is not None]
                samples = self.simulate(bdb, genid, modelno, Y, Q,
                    numpredictions=numsamples)
                samples = [s[0] for s in samples]
        # Predicting fcol.
        else:
            conditions = {c:v for c,v in zip(colnames, row) if
                bayesdb_generator_column_number(bdb, genid, c) in
                self.get_pcols(bdb, genid, colno)}
            for colname, val in conditions.iteritems():
                # Impute all missing parents.
                if val is None:
                    imp_col = bayesdb_generator_column_number(bdb, genid,
                        colname)
                    imp_val, imp_conf = self.predict_confidence(bdb, genid,
                        modelno, imp_col, rowid, numsamples=numsamples)
                    # XXX If imputing several parents, take the overall
                    # overall conf as min conf. If we define imp_conf as
                    # P[imp_val = correct] then we might choose to multiply
                    # the imp_confs, but we cannot assert that the imp_confs
                    # are independent so multiplying is extremely conservative.
                    parent_conf = min(parent_conf, imp_conf)
                    conditions[colname] = imp_val
            assert all(v is not None for c,v in conditions.iteritems())
            predictor = self.get_predictor(bdb, genid, colno)
            samples = predictor.simulate(numsamples, conditions)
        # Since foreign predictor does not know how to impute, imputation
        # shall occur here in the composer by simulate/logpdf calls.
        stattype = bayesdb_generator_column_stattype(bdb, genid, colno)
        if stattype == 'categorical':
            # imp_conf is most frequent.
            imp_val =  max(((val, samples.count(val)) for val in set(samples)),
                key=lambda v: v[1])[0]
            if colno in self.get_fcols(bdb, genid):
                imp_conf = math.exp(predictor.logpdf(imp_val, conditions))
            else:
                imp_conf = (np.array(samples)==imp_val) / len(samples)
        elif stattype == 'numerical':
            # XXX The definition of confidence is P[k=1] where
            # k=1 is the number of mixture componets (we need a distribution
            # over GPMM to answer this question). The confidence is instead
            # implemented as \max_i{p_i} where p_i are the weights of a
            # fitted DPGMM.
            imp_val = np.mean(samples)
            imp_conf = su.continuous_imputation_confidence(samples, None, None,
                n_steps=1000)
        else:
            raise ValueError('Unknown stattype {} for a foreign predictor '
                'column encountered in predict_confidence.'.format(stattype))
        return imp_val, imp_conf * parent_conf

    def simulate(self, bdb, genid, modelno, constraints, colnos,
            numpredictions=1):
        # Delegate to crosscat if colnos+constraints all lcols.
        all_cols = [c for c,v in constraints] + colnos
        if all(f not in all_cols for f in self.get_fcols(bdb, genid)):
            Y_cc = [(self.get_cc_colno(bdb, genid, c), v)
                for c, v in constraints]
            Q_cc = self.get_cc_colnos(bdb, genid, colnos)
            return self.get_cc(bdb, genid).simulate(bdb,
                self.get_cc_id(bdb, genid), modelno, Y_cc, Q_cc,
                numpredictions=numpredictions)
        # Solve inference problem by sampling-importance resampling.
        result = []
        for i in xrange(numpredictions):
            samples, weights = self._weighted_sample(bdb, genid, modelno,
                constraints)
            p = np.exp(np.asarray(weights) - np.max(weights))
            p /= np.sum(p)
            draw = np.nonzero(np.random.multinomial(1,p))[0][0]
            s = [samples[draw].get(col) for col in colnos]
            result.append(s)
        return result

    def row_similarity(self, bdb, genid, modelno, rowid, target_rowid,
            colnos):
        # XXX Delegate to CrossCat always.
        cc_colnos = self.get_cc_colnos(bdb, genid, colnos)
        return self.get_cc(bdb, genid).row_similarity(bdb,
            self.get_cc_id(bdb, genid), modelno, rowid, target_rowid, cc_colnos)

    def row_column_predictive_probability(self, bdb, genid, modelno,
            rowid, colno):
        raise NotImplementedError('PREDICTIVE PROBABILITY is being retired. '
            'Please use PROBABILITY OF <[...]> [GIVEN [...]] instead.')

    def _weighted_sample(self, bdb, genid, modelno, Y, n_samples=None):
        # Returns a list of [(sample, weight), ...]
        # Each `sample` is the a vector s=(X1,...,Xn) of values for all nodes in
        # the network. Y specifies evidence nodes: all returned samples have
        # constrained values at the evidence nodes.
        # `weight` is the likelihood of the evidence Y under s\Y.
        if n_samples is None:
            n_samples = 100
        # Create n_samples dicts, each entry is weighted sample from joint.
        samples = [{c:v for (c,v) in Y} for _ in xrange(n_samples)]
        weights = []
        w0 = 0
        # Assess likelihood of evidence at root.
        cc_evidence = [(c, v) for c,v in Y if c in self.get_lcols(bdb, genid)]
        if cc_evidence:
            w0 += self._joint_logpdf_cc(bdb, genid, modelno, cc_evidence, [])
        # Simulate latent CCs.
        cc_latent = [c for c in self.get_lcols(bdb, genid) if c not in
                samples[0]]
        cc_simulated = self.get_cc(bdb, genid).simulate(bdb,
            self.get_cc_id(bdb, genid), modelno, cc_evidence, cc_latent,
            numpredictions=n_samples)
        for k in xrange(n_samples):
            w = w0
            # Add simulated lcols.
            samples[k].update({c:v for c,v in zip(cc_latent, cc_simulated[k])})
            for fcol in self.get_topo(bdb, genid):
                pcols = self.get_pcols(bdb, genid, fcol)
                predictor = self.get_predictor(bdb, genid, fcol)
                # Assert all parents of FP known (evidence or simulated).
                assert pcols.issubset(set(samples[k]))
                conditions = {bayesdb_generator_column_name(bdb, genid, c):v
                    for c,v in samples[k].iteritems() if c in pcols}
                # f is evidence: compute likelihood weight.
                if fcol in samples[k]:
                    w += predictor.logpdf(samples[k][fcol], conditions)
                # f is latent: simulate from conditional distribution.
                else:
                    samples[k][fcol] = predictor.simulate(1, conditions)[0]
            weights.append(w)
        return samples, weights

    # TODO migrate to a reasonable place (ie sample_utils in CrossCat).
    def _joint_logpdf_cc(self, bdb, genid, modelno, Q, Y):
        # Ensure consistency for nodes in both query and evidence.
        lcols = self.get_lcols(bdb, genid)
        ignore = set()
        for (cq, vq), (cy, vy) in itertools.product(Q, Y):
            if cq not in lcols and cy not in lcols:
                raise ValueError('Foreign colno encountered in internal '
                    '_joint_logpdf_cc.')
            if cq == cy:
                if vq == vy:
                    ignore.add(cq)
                else:
                    return -float('inf')
        # Convert.
        Qi = []
        for (col, val) in Q:
            if col not in ignore:
                Qi.append((self.get_cc_colno(bdb, genid, col), val))
        Yi = []
        for (col, val) in Y:
            Yi.append((self.get_cc_colno(bdb, genid, col), val))
        # Chain rule.
        prob = 0
        for (col, val) in Qi:
            r = self.get_cc(bdb, genid).column_value_probability(bdb,
                    self.get_cc_id(bdb, genid), modelno, col, val, Yi)
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
        graph : list or dict
            Adjacency list or dict representing the graph, for example:
                graph_l = [(10, [8, 7]), (5, [8, 7, 9, 10, 11, 13, 15])]
                graph_d = {10: [8, 7], 5: [8, 7, 9, 10, 11, 13, 15]}

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

    def get_cc_colno(self, bdb, genid, colno):
        return self.get_cc_colnos(bdb, genid, [colno])[0]

    def get_cc_colnos(self, bdb, genid, colnos):
        lcolnames = [bayeslite.core.bayesdb_generator_column_name(bdb,
            genid, colno) for colno in colnos]
        return [bayeslite.core.bayesdb_generator_column_number(bdb,
            self.get_cc_id(bdb, genid), lcolname) for lcolname in lcolnames]

    def get_cc_id(self, bdb, genid):
        cursor = bdb.sql_execute('''
            SELECT crosscat_generator_id FROM bayesdb_composer_cc_id
                WHERE generator_id = ?
        ''', (genid,))
        return cursor.fetchall()[0][0]

    def get_cc(self, bdb, genid):
        return bayesdb_generator_metamodel(bdb, self.get_cc_id(bdb, genid))

    def get_lcols(self, bdb, genid):
        cursor = bdb.sql_execute('''
            SELECT colno FROM bayesdb_composer_column_owner
                WHERE generator_id = ? AND local = 1
                ORDER BY colno ASC
        ''', (genid,))
        return set([row[0] for row in cursor])

    def get_fcols(self, bdb, genid):
        cursor = bdb.sql_execute('''
            SELECT colno FROM bayesdb_composer_column_owner
                WHERE generator_id = ? AND local = 0
                ORDER BY colno ASC
        ''', (genid,))
        return set([row[0] for row in cursor])

    def get_pcols(self, bdb, genid, fcolno):
        cursor = bdb.sql_execute('''
            SELECT pcolno FROM bayesdb_composer_column_parents
                WHERE generator_id = ? AND fcolno = ?
                ORDER BY pcolno ASC
        ''', (genid, fcolno))
        return set([row[0] for row in cursor])

    def get_topo(self, bdb, genid):
        cursor = bdb.sql_execute('''
            SELECT colno FROM bayesdb_composer_column_toposort
                WHERE generator_id = ?
                ORDER BY position ASC
            ''', (genid,))
        return [row[0] for row in cursor]

    def get_predictor_name(self, bdb, genid, fcol):
        cursor = bdb.sql_execute('''
            SELECT predictor_name FROM bayesdb_composer_column_foreign_predictor
                WHERE generator_id = ? AND colno = ?
        ''', (genid, fcol))
        return cursor.fetchall()[0][0]

    def get_predictor(self, bdb, genid, fcol):
        if (genid, fcol) not in self.fp_cache:
            cursor = bdb.sql_execute('''
                SELECT predictor_name, predictor_binary
                    FROM bayesdb_composer_column_foreign_predictor
                    WHERE generator_id = ? AND colno = ?
            ''', (genid, fcol))
            name, binary = cursor.fetchall()[0]
            builder = self.fp_builders[name]
            self.fp_cache[(genid, fcol)] = builder.deserialize(binary)
        return self.fp_cache[(genid, fcol)]
