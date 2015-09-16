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

import bayeslite.core
import bayeslite.metamodel
import bayeslite.bqlfn

import bdbcontrib
from bdbcontrib.foreign.sat_orbital_mech import SatOrbitalMechanics
from bdbcontrib.foreign.sat_random_forest import SatRandomForest

class SatellitesMetamodel(bayeslite.metamodel.IBayesDBMetamodel):

    """A metamodel for the satellites dataset which supports two foreign
    predictors; a SatRandomForest targeting a single categorical column,
    and SatOrbitalMechanics. Other columns in the dataset are modeled using the
    default metamodel (ie CrosscatMetamodel).
    """

    def __init__(self):
        self.cc_id = None
        self.cc = None

    def name(self):
        """Return the name of the metamodel as a str."""
        return 'satcat'

    def register(self, bdb):
        # XXX Figure out what to do.
        # XXX Causes serialization problem, should have v0 of schema which
        # is only ever before the first time this metamodel is registered
        # in the history of the bdb.
        bdb.sql_execute("""
            INSERT INTO bayesdb_metamodel (name, version)
                VALUES ('satcat', 1);
            """)
        bdb.sql_execute("""
            CREATE TABLE bayesdb_satcat_crosscat_lookup (
                generator_id INTEGER NOT NULL PRIMARY KEY
                        REFERENCES bayesdb_generator(id),
                crosscat_generator_id INTEGER NOT NULL
                        REFERENCES bayesdb_generator(id)
            );
            """)

    def create_generator(self, bdb, table, schema, instantiate):
        # XXX Hardcoded schema forever.
        # XXX Get this name from the sche
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
            ('Class_of_orbit', 'CATEGORICAL'),  # Remove from internal CC
            ('Period_minutes', 'NUMERICAL')     # Remove from internal CC
            ]

        # First instantiate **this** generator.
        generator_id, _ = instantiate(schema)
        self.cc_name = bayeslite.core.bayesdb_generator_name(bdb,
            generator_id) + '_cc'

        # Create the internal crosscat generator.
        cc_schema = schema[:-2]
        txt = ','.join(['{} {}'.format(pair[0], pair[1]) for pair in cc_schema])
        bql = """
            CREATE GENERATOR {} FOR satellites USING crosscat(
                GUESS(*), Class_of_orbit IGNORE, Period_minutes IGNORE,
                {}
            );
        """.format(self.cc_name, txt)
        bdb.execute(bql)

        # Obtain the generator_id of satcat_cc. Should be stored in the database
        # table bayesdb_satcat_crosscat_lookup not as `self`.
        self.cc_id = bayeslite.core.bayesdb_get_generator(bdb, self.cc_name)
        # Obtain the crosscat metamodel object
        self.cc = bayeslite.core.bayesdb_generator_metamodel(bdb, self.cc_id)

    def drop_generator(self, bdb, generator_id):
        # Delegate
        bdb.execute('''DROP GENERATOR {}'''.format(self.cc_name))
        # self.cc.drop_generator(bdb, self.cc_id)

    def initialize_models(self, bdb, generator_id, modelnos, model_config):
        # Initialize Crosscat.
        bql = """
            INITIALIZE {} MODELS FOR {};
            """.format(len(modelnos), self.cc_name)
        bdb.execute(bql)

        # Obtain the table as a Pandas df.
        table = bayeslite.core.bayesdb_generator_table(bdb, generator_id)
        table_df = bdbcontrib.cursor_to_df(bdb.sql_execute(
            'SELECT * FROM {}'.format(table)))

        # Initialize the foreign predictors.
        self.sat_rf = SatRandomForest(table_df)
        self.sat_om = SatOrbitalMechanics(table_df)

        # Internal colno of RF target.
        self.sat_rf_target = bayeslite.core.bayesdb_generator_column_number(bdb,
            generator_id, self.sat_rf.get_targets()[0])
        # Internal col of RF conditions.
        self.sat_rf_conditions = \
            [bayeslite.core.bayesdb_generator_column_number(bdb,
                generator_id, cond) for cond in self.sat_rf.get_conditions()]
        # Internal col of OM target.
        self.sat_om_target = bayeslite.core.bayesdb_generator_column_number(bdb,
            generator_id, self.sat_om.get_targets()[0])
        # Internal col of OM conditions.
        self.sat_om_conditions = \
            [bayeslite.core.bayesdb_generator_column_number(bdb,
                generator_id, cond) for cond in self.sat_om.get_conditions()]
        # Colnos unmodeled by CrossCat.
        self.unmodeled = [self.sat_rf_target, self.sat_om_target]

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
                colno1):
        # Determine which object modeled colno0, colno1
        colno0_foreign = colno0 in self.unmodeled
        colno1_foreign = colno1 in self.unmodeled

        # If neither col is foreign, delegate to CrossCat
        if not (colno0_foreign or colno1_foreign):
            return self.cc.column_dependence_probability(bdb, self.cc_id,
                modelno,
                self._get_internal_cc_colno(bdb, generator_id, colno0),
                self._get_internal_cc_colno(bdb, generator_id, colno1))

        # If (colno0, colno1) form a (target, given) pair, then we explicitly
        # modeled them as dependent!
        # TODO: What if an FP determines it is not dependent on one of its
        # conditions? (ie regression with zero coefficient.)
        # -- check for RF
        elif ((colno0 == self.sat_rf_target and colno1 in self.sat_rf_conditions)
                or (colno0 in self.sat_rf_conditions and colno1 ==
                    self.sat_rf_target)):
            return 1
        # -- check for OM
        elif ((colno0 == self.sat_om_target and colno1 in self.sat_om_conditions)
                or (colno0 in self.sat_om_conditions and colno1 ==
                    self.sat_om_target)):
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
                        self._get_internal_cc_colno(bdb, generator_id, cc_col),
                        self._get_internal_cc_colno(bdb, generator_id, fc))
                # Otherwise fc is itself a target of the GPM, do the recursive
                # call. Will terminate iff the DAG of GPMs is acyclic.
                # TODO: Prove.
                else:
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
        cc_colnos = self._get_internal_cc_colnos(bdb, generator_id,
            [colno0, colno1])
        return self.cc.column_mutual_information(bdb, self.cc_id, modelno,
            cc_colnos[0], cc_colnos[1], numsamples=numsamples)

    def column_value_probability(self, bdb, generator_id, modelno, colno,
                value, constraints):
        # Delegate
        cc_colno = self._get_internal_cc_colno(bdb, generator_id, colno)
        return self.cc.column_value_probability(bdb, self.cc_id, modelno,
            cc_colno, value, constraints)

    def row_similarity(self, bdb, generator_id, modelno, rowid, target_rowid,
                colnos):
        # Delegate
        cc_colnos = self._get_internal_cc_colnos(bdb, generator_id, colnos)
        return self.cc.row_similarity(bdb, self.cc_id, modelno, rowid,
            target_rowid, cc_colnos)

    def row_column_predictive_probability(self, bdb, generator_id, modelno,
                rowid, colno):
        # Delegate
        cc_colno = self._get_internal_cc_colno(bdb, generator_id, colno)
        return self.cc.row_column_predictive_probability(bdb, self.cc_id,
            modelno, rowid, cc_colno)

    def predict_confidence(self, bdb, generator_id, modelno, colno, rowid,
                numsamples=None):
        # Delegate
        cc_colno = self._get_internal_cc_colno(bdb, generator_id, colno)
        return self.predict_confidence(bdb, self.cc_id, modelno, cc_colno,
            rowid, numsamples=numsamples)

    def simulate(self, bdb, generator_id, modelno, constraints, colnos,
                numpredictions=1):
        # Delegate
        cc_colnos = self._get_internal_cc_colnos(bdb, generator_id, colnos)
        return self.cc.simulate(bdb, self.cc_id, modelno, constraints, cc_colnos,
            numpredictions=numpredictions)

    def _set_cc_columns(self, columns):
        """
        Models `columns` using the default generator (crosscat).
        `columns` is a list<tuple> = [(colname), (stattype)]
        """
        self.cc_cols = columns

    def _set_rf_columns(self, target, conditions):
        """
        Models `target` column given `conditions` columns using SatRandomForest.
        `target` must be a categorical column.
        `conditions`
        """
        # Check target does not already have model.
        if target in self.cc_cols + self.om_cols:
            raise ValueError('Target column {} already has a '
                'model.'.format(target))
        # Check all the conditions have a model.
        for c in conditions:
            if c not in self.cc_cols + self.om_cols:
                raise ValueError('Condition column {} does not have a '
                    'model.'.format(c))

    def _set_om_columns(self):
        """
        Models `Period_minutes`|`Apogee_km`, `Perigee_`km` using Kepler's
        Third Law plus a Gaussian noise model.
        """
        # Check `Period_minutes` does not already have a model.
        if SatOrbitalMechanics.target[0] in self.cc_cols + self.rf_target_col:
            raise ValueError('Target column {} already has a '
                'model.'.format(SatOrbitalMechanics.target[0]))
        # Check `Apogee_km`, `Perigee_km` are modeling using default.
        for c in SatOrbitalMechanics.conditions:
            if c not in self.cc_cols:
                raise ValueError('Condition column {} does not have a '
                    'model. Use the default model.'.format(c))

    def _get_internal_cc_colno(self, bdb, generator_id, colno):
        return self._get_internal_cc_colnos(bdb, generator_id, colno)[0]

    def _get_internal_cc_colnos(self, bdb, generator_id, colnos):
        # First get the names from this generator.
        colnames = [bayeslite.core.bayesdb_generator_column_name(bdb,
            generator_id, no) for no in colnos]
        # Now do the inverse mapping.
        return [bayeslite.core.bayesdb_generator_column_number(bdb, self.cc_id,
            name) for name in colnames]

    def _get_condition_cols(self, col):
        if col == self.sat_rf_target:
            return self.sat_rf_conditions
        elif col == self.sat_om_target:
            return self.sat_om_conditions
        else:
            raise ValueError('Can only obtain the condition column numbers '
                'for columns modeled by a foreign predictor.')
