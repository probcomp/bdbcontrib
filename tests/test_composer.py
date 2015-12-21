# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2015, MIT Probabilistic Computing Project
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

import os
import pytest

import bayeslite
from bayeslite.sqlite3_util import sqlite3_quote_name as quote

import bdbcontrib
from bdbcontrib.metamodels.composer import Composer
from bdbcontrib.predictors import random_forest
from bdbcontrib.predictors import keplers_law
from bdbcontrib.predictors import multiple_regression


# Use satellites for all tests.
PATH_TESTS = os.path.dirname(os.path.abspath(__file__))
PATH_ROOT = os.path.dirname(PATH_TESTS)
PATH_EXAMPLES = os.path.join(PATH_ROOT, 'examples')
PATH_SATELLITES = os.path.join(PATH_EXAMPLES, 'satellites')
PATH_SATELLITES_CSV = os.path.join(PATH_SATELLITES, 'satellites.csv')

# ------------------------------------------------------------------------------
# The following live outside the TestComposer class since we do not need to
# reuse a bdb instance across tests.

def test_create_generator_schema():
    bdb = bayeslite.bayesdb_open()
    bayeslite.bayesdb_read_csv_file(bdb, 'satellites', PATH_SATELLITES_CSV,
        header=True, create=True)
    composer = Composer(n_samples=5)
    bayeslite.bayesdb_register_metamodel(bdb, composer)
    # Using crosscat and default to specify models should work.
    bdb.execute('''
        CREATE GENERATOR t1 FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                Apogee_km NUMERICAL, Eccentricity NUMERICAL
            ),
            crosscat (
                Anticipated_Lifetime NUMERICAL, Contractor CATEGORICAL
            )
        );''')
    assert bayeslite.core.bayesdb_has_generator(bdb, 't1_cc')
    # IGNORE and GUESS(*) are forbidden and should crash.
    with pytest.raises(Exception):
        bdb.execute('''
            CREATE GENERATOR t2 FOR satellites USING composer(
                default (
                    GUESS(*), Country_of_Operator IGNORE,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                )
            );''')
    # Test unregistered foreign predictor.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t3 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                ),
                random_forest (
                    Apogee_km NUMERICAL GIVEN Operator_Owner
                )
            );''')
    # Unregistered foreign predictor should crash.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t4 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                ),
                random_forest (
                    Apogee_km NUMERICAL GIVEN Operator_Owner
                )
            );''')
    # Registered foreign predictor should work.
    composer.register_foreign_predictor(random_forest.RandomForest)
    bdb.execute('''
        CREATE GENERATOR t5 FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                Eccentricity NUMERICAL
            ),
            random_forest (
                Apogee_km NUMERICAL GIVEN Operator_Owner
            )
        );''')
    # Wrong stattype in predictor should crash.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t6 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                ),
                random_forest (
                    Apogee_km RADIAL GIVEN Operator_Owner
                )
            );''')
    # Missing GIVEN keyword should crash.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t6 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                ),
                random_forest (
                    Apogee_km NUMERICAL, Operator_Owner
                )
            );''')
    # Missing conditions in random forest conditions should crash.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t7 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                ),
                random_forest (
                    Apogee_km NUMERICAL GIVEN Operator_Owner
                )
            );''')
    # Test duplicate declarations.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t7 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                ),
                random_forest (
                    Class_of_orbit CATEGORICAL GIVEN Operator_Owner
                )
            );''')
    # Arbitrary DAG with foreign predictors.
    composer.register_foreign_predictor(multiple_regression.MultipleRegression)
    bdb.execute('''
        CREATE GENERATOR t8 FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
            ),
            random_forest (
                Apogee_km NUMERICAL GIVEN Operator_Owner, Users
            ),
            multiple_regression (
                Eccentricity NUMERICAL GIVEN Apogee_km, Users, Perigee_km
            )
        );''')
    # Duplicate declarations in foreign predictors should crash.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t9 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                    Apogee_km NUMERICAL, Eccentricity NUMERICAL
                ),
                random_forest (
                    Perigee_km NUMERICAL GIVEN Purpose
                ),
                multiple_regression (
                    Perigee_km NUMERICAL GIVEN Operator_Owner
                )
            );''')
    # MML for default models should work.
    bdb.execute('''
        CREATE GENERATOR t10 FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Class_of_orbit CATEGORICAL, Apogee_km NUMERICAL
            )
            random_forest (
                Perigee_km NUMERICAL GIVEN Purpose
            )
            multiple_regression (
                Eccentricity NUMERICAL GIVEN Operator_Owner, Class_of_orbit
            )
            DEPENDENT(Apogee_km, Perigee_km, Purpose),
            INDEPENDENT(Country_of_Operator, Purpose)
        );''')
    # MML for foreign predictors should crash.
    with pytest.raises(ValueError):
        bdb.execute('''
            CREATE GENERATOR t11 FOR satellites USING composer(
                default (
                    Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                    Users CATEGORICAL, Purpose CATEGORICAL,
                    Class_of_orbit CATEGORICAL,
                    Apogee_km NUMERICAL
                ),
                random_forest (
                    Perigee_km NUMERICAL GIVEN Purpose
                ),
                multiple_regression (
                    Eccentricity NUMERICAL GIVEN Operator_Owner, Class_of_orbit
                )
                DEPENDENT(Apogee_km, Eccentricity, Country_of_Operator),
                INDEPENDENT(Perigee_km, Purpose)
            );''')
    # Test full generator.
    composer.register_foreign_predictor(keplers_law.KeplersLaw)
    bdb.execute('''
        CREATE GENERATOR t12 FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                Apogee_km NUMERICAL, Eccentricity NUMERICAL,
                Launch_Mass_kg NUMERICAL, Dry_Mass_kg NUMERICAL,
                Power_watts NUMERICAL, Date_of_Launch NUMERICAL,
                Contractor CATEGORICAL,
                Country_of_Contractor CATEGORICAL, Launch_Site CATEGORICAL,
                Launch_Vehicle CATEGORICAL,
                Source_Used_for_Orbital_Data CATEGORICAL,
                longitude_radians_of_geo NUMERICAL,
                Inclination_radians NUMERICAL,
            ),
            random_forest (
                Type_of_Orbit CATEGORICAL
                    GIVEN Apogee_km, Perigee_km,
                        Eccentricity, Period_minutes, Launch_Mass_kg,
                        Power_watts, Anticipated_Lifetime, Class_of_orbit
            ),
            keplers_law (
                Period_minutes NUMERICAL
                    GIVEN Perigee_km, Apogee_km
            ),
            multiple_regression (
                Anticipated_Lifetime NUMERICAL
                    GIVEN Dry_Mass_kg, Power_watts, Launch_Mass_kg, Contractor
            ),
            DEPENDENT(Apogee_km, Perigee_km, Eccentricity),
            INDEPENDENT(Country_of_Operator, longitude_radians_of_geo)
        );''')
    bdb.close()

def test_register():
    bdb = bayeslite.bayesdb_open()
    composer = Composer(n_samples=5)
    bayeslite.bayesdb_register_metamodel(bdb, composer)
    # Check if globally registered.
    try:
        bdb.sql_execute('''
            SELECT * FROM bayesdb_metamodel WHERE name={}
        '''.format(quote(composer.name()))).next()
    except StopIteration:
        pytest.fail('Composer not registered in bayesdb_metamodel.')
    # Check all tables/triggers.
    schema = [
        ('table', 'bayesdb_composer_cc_id'),
        ('table', 'bayesdb_composer_column_owner'),
        ('table', 'bayesdb_composer_column_toposort'),
        ('trigger', 'bayesdb_composer_column_toposort_check'),
        ('table', 'bayesdb_composer_column_parents'),
        ('table', 'bayesdb_composer_column_foreign_predictor'),
        ('trigger', 'bayesdb_composer_column_foreign_predictor_check')
    ]
    for kind, name in schema:
        try:
            bdb.sql_execute('''
                SELECT * FROM sqlite_master WHERE type={} AND name={}
            '''.format(quote(kind), quote(name))).next()
        except StopIteration:
            pytest.fail('Missing from Composer schema: {}'.format((kind,name)))
    bdb.close()

def test_register_foreign_predictor():
    bdb = bayeslite.bayesdb_open()
    composer = Composer(n_samples=5)
    bayeslite.bayesdb_register_metamodel(bdb, composer)
    # Register valid predictors.
    composer.register_foreign_predictor(random_forest.RandomForest)
    composer.register_foreign_predictor(multiple_regression.MultipleRegression)
    composer.register_foreign_predictor(keplers_law.KeplersLaw)
    # Register duplicates.
    with pytest.raises(ValueError):
        composer.register_foreign_predictor(keplers_law.KeplersLaw)
    with pytest.raises(ValueError):
        composer.register_foreign_predictor(
            multiple_regression.MultipleRegression)
    with pytest.raises(ValueError):
        composer.register_foreign_predictor(random_forest.RandomForest)
    # Register invalid predictors.
    with pytest.raises(AssertionError):
        composer.register_foreign_predictor(None)
    with pytest.raises(AssertionError):
        composer.register_foreign_predictor('bans')


def test_drop_generator():
    bdb = bayeslite.bayesdb_open()
    # Initialize the database
    bayeslite.bayesdb_read_csv_file(bdb, 'satellites', PATH_SATELLITES_CSV,
        header=True, create=True)
    composer = Composer(n_samples=5)
    bayeslite.bayesdb_register_metamodel(bdb, composer)
    composer.register_foreign_predictor(random_forest.RandomForest)
    composer.register_foreign_predictor(multiple_regression.MultipleRegression)
    composer.register_foreign_predictor(keplers_law.KeplersLaw)
    bdb.execute('''
        CREATE GENERATOR t1 FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                Apogee_km NUMERICAL, Eccentricity NUMERICAL,
                Launch_Mass_kg NUMERICAL, Dry_Mass_kg NUMERICAL,
                Power_watts NUMERICAL, Date_of_Launch NUMERICAL,
                Contractor CATEGORICAL,
                Country_of_Contractor CATEGORICAL, Launch_Site CATEGORICAL,
                Launch_Vehicle CATEGORICAL,
                Source_Used_for_Orbital_Data CATEGORICAL,
                longitude_radians_of_geo NUMERICAL,
                Inclination_radians NUMERICAL,
            ),
            random_forest (
                Type_of_Orbit CATEGORICAL
                    GIVEN Apogee_km, Perigee_km,
                        Eccentricity, Period_minutes, Launch_Mass_kg,
                        Power_watts, Anticipated_Lifetime, Class_of_orbit
            ),
            keplers_law (
                Period_minutes NUMERICAL
                    GIVEN Perigee_km, Apogee_km
            ),
            multiple_regression (
                Anticipated_Lifetime NUMERICAL
                    GIVEN Dry_Mass_kg, Power_watts, Launch_Mass_kg,
                    Contractor
            ),
            DEPENDENT(Apogee_km, Perigee_km, Eccentricity),
            DEPENDENT(Contractor, Country_of_Contractor),
            INDEPENDENT(Country_of_Operator, Date_of_Launch)
        );''')
    generator_id = bayeslite.core.bayesdb_get_generator(bdb, 't1')
    schema = [
        ('table', 'bayesdb_composer_cc_id'),
        ('table', 'bayesdb_composer_column_owner'),
        ('table', 'bayesdb_composer_column_toposort'),
        ('table', 'bayesdb_composer_column_parents'),
        ('table', 'bayesdb_composer_column_foreign_predictor'),
    ]
    # Iterate through tables before dropping.
    for _, name in schema:
        bdb.sql_execute('''
            SELECT * FROM {} WHERE generator_id=?
        '''.format(quote(name)), (generator_id,)).next()
    # Drop generator and ensure table lookups with generator_id throw error.
    bdb.execute('DROP GENERATOR t1')
    for _, name in schema:
        with pytest.raises(StopIteration):
            bdb.sql_execute('''
                SELECT * FROM {} WHERE generator_id=?
            '''.format(quote(name)), (generator_id,)).next()
    assert not bayeslite.core.bayesdb_has_generator(bdb, 't1')
    assert not bayeslite.core.bayesdb_has_generator(bdb, 't1_cc')
    bdb.close()

def test_composer_integration_slow():
    # But currently difficult to seperate these tests into smaller tests because
    # of their sequential nature. We will still test all internal functions
    # with different regimes of operation.

    # SETUP
    # -----
    # Dataset.
    bdb = bayeslite.bayesdb_open()
    bayeslite.bayesdb_read_csv_file(bdb, 'satellites', PATH_SATELLITES_CSV,
        header=True, create=True)
    bdbcontrib.nullify(bdb, 'satellites', 'NaN')
    # Composer.
    composer = Composer(n_samples=5)
    composer.register_foreign_predictor(
        multiple_regression.MultipleRegression)
    composer.register_foreign_predictor(keplers_law.KeplersLaw)
    composer.register_foreign_predictor(random_forest.RandomForest)
    # Use complex generator for interesting test cases.
    bayeslite.bayesdb_register_metamodel(bdb, composer)
    bdb.execute('''
        CREATE GENERATOR t1 FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
                Apogee_km NUMERICAL, Eccentricity NUMERICAL,
                Launch_Mass_kg NUMERICAL, Dry_Mass_kg NUMERICAL,
                Power_watts NUMERICAL, Date_of_Launch NUMERICAL,
                Contractor CATEGORICAL,
                Country_of_Contractor CATEGORICAL, Launch_Site CATEGORICAL,
                Launch_Vehicle CATEGORICAL,
                Source_Used_for_Orbital_Data CATEGORICAL,
                longitude_radians_of_geo NUMERICAL,
                Inclination_radians NUMERICAL,
            ),
            random_forest (
                Type_of_Orbit CATEGORICAL
                    GIVEN Apogee_km, Perigee_km,
                        Eccentricity, Period_minutes, Launch_Mass_kg,
                        Power_watts, Anticipated_Lifetime, Class_of_orbit
            ),
            keplers_law (
                Period_minutes NUMERICAL
                    GIVEN Perigee_km, Apogee_km
            ),
            multiple_regression (
                Anticipated_Lifetime NUMERICAL
                    GIVEN Dry_Mass_kg, Power_watts, Launch_Mass_kg,
                    Contractor
            ),
            DEPENDENT(Apogee_km, Perigee_km, Eccentricity),
            DEPENDENT(Contractor, Country_of_Contractor),
            INDEPENDENT(Country_of_Operator, Date_of_Launch)
        );''')


    # ----------------------
    # TEST INITIALIZE MODELS
    # ----------------------

    bdb.execute('INITIALIZE 2 MODELS FOR t1')
    # Check number of models.
    df = bdbcontrib.describe_generator_models(bdb, 't1')
    assert len(df) == 2
    df = bdbcontrib.describe_generator_models(bdb, 't1_cc')
    assert len(df) == 2

    # -------------------
    # TEST ANALYZE MODELS
    # -------------------

    bdb.execute('ANALYZE t1 FOR 2 ITERATIONS WAIT;')
    # Check number of iterations of composer.
    df = bdbcontrib.describe_generator_models(bdb, 't1')
    for index, modelno, iterations in df.itertuples():
        assert iterations == 2
    # Check number of iterations of composer_cc.
    df = bdbcontrib.describe_generator_models(bdb, 't1_cc')
    for index, modelno, iterations in df.itertuples():
        assert iterations == 2

    # ----------------------------------
    # TEST COLUMN DEPENDENCE PROBABILITY
    # ----------------------------------

    # Special 0/1 regimes.
    # Local with a INDEPENDENT local should be 0.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Date_of_Launch
            WITH Country_of_Operator FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 0
    # Local with a DEPENDENT local should be 1.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Perigee_km WITH Eccentricity
            FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Apogee_km WITH Eccentricity
            FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1
    # Foreign with a local parent should be 1.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Period_minutes WITH Apogee_km
            FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Anticipated_Lifetime WITH Power_watts
            FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1.
    # Foreign with a foreign parent should be 1.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Type_of_Orbit WITH
            Anticipated_Lifetime FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1.
    # Foreign with a local non-parent DEPENDENT with local parent should be 1.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Period_minutes WITH
            Eccentricity FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1.
    # Foreign with foreign sharing common direct ancestor should be 1.
    # Launch_Mass_kg is the common parent.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Anticipated_Lifetime WITH
            Type_of_Orbit FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1.
    # Foreign with a foreign sharing a common DEPENDENT ancestor should be 1.
    # Eccentricity is a parent of Type_of_orbit, and is dependent
    # with Period_minutes through DEPENDENT(Apogee_km, Perigee_km, Eccentricity)
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Period_minutes WITH
            Type_of_Orbit FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1.
    # Column with itself should be 1.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Anticipated_Lifetime WITH
            Anticipated_Lifetime FROM t1 LIMIT 1
    ''')
    assert curs.next()[0] == 1.

    # Unknown [0,1] regimes.
    # Foreign with a local of unknown relation with parents.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Anticipated_Lifetime WITH
            longitude_radians_of_geo FROM t1 LIMIT 1
    ''')
    assert 0 <= curs.next()[0] <= 1.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Period_minutes WITH
            longitude_radians_of_geo FROM t1 LIMIT 1
    ''')
    assert 0 <= curs.next()[0] <= 1.
    # Foreign with a foreign of unknown ancestry relation.
    curs = bdb.execute('''
        ESTIMATE DEPENDENCE PROBABILITY OF Anticipated_Lifetime WITH
            Period_minutes FROM t1 LIMIT 1
    ''')
    assert 0 <= curs.next()[0] <= 1.

    # ----------------------------------
    # TEST SIMULATE
    # ----------------------------------

    # Crash tests for various code paths. Quality of simulations ignored.
    # Joint local.
    curs = bdb.execute('''
        SIMULATE Power_watts, Launch_Mass_kg FROM t1 LIMIT 2;
    ''')
    assert len(curs.fetchall()) == 2
    # Forward simulate foreign.
    curs = bdb.execute('''
        SIMULATE Period_minutes FROM t1 GIVEN Apogee_km = 1000, Perigee_km = 980
            LIMIT 2;
    ''')
    assert len(curs.fetchall()) == 2
    # Forward simulate foreign with missing parents.
    curs = bdb.execute('''
        SIMULATE Anticipated_Lifetime FROM t1 GIVEN Dry_Mass_kg = 2894,
            Launch_Mass_kg = 1730 LIMIT 2;
    ''')
    assert len(curs.fetchall()) == 2
    # Joint simulate foreign with parents, and missing parents.
    curs = bdb.execute('''
        SIMULATE Type_of_Orbit, Eccentricity FROM t1 GIVEN Dry_Mass_kg = 2894,
            Launch_Mass_kg = 1730 LIMIT 2;
    ''')
    assert len(curs.fetchall()) == 2
    # Joint simulate foreign with non-parents.
    curs = bdb.execute('''
        SIMULATE Period_minutes, Eccentricity FROM t1 GIVEN Apogee_km = 38000
            LIMIT 2;
    ''')
    assert len(curs.fetchall()) == 2
    # Simulate joint local conditioned on two foreigns.
    curs = bdb.execute('''
        SIMULATE Country_of_Operator, Inclination_radians FROM t1
            GIVEN Period_minutes = 1432, Anticipated_Lifetime = 5 LIMIT 2;
    ''')
    assert len(curs.fetchall()) == 2
    # Simulate joint foreign conditioned on third foreign.
    curs = bdb.execute('''
        SIMULATE Period_minutes, Anticipated_Lifetime FROM t1
            GIVEN Type_of_Orbit = 'Deep Highly Eccentric' LIMIT 2
    ''')
    assert len(curs.fetchall()) == 2
    # Simulate foreign conditioned on itself.
    curs = bdb.execute('''
        SIMULATE Period_minutes, Apogee_km FROM t1
            GIVEN Period_minutes = 102 LIMIT 2
    ''')
    assert [s[0] for s in curs] == [102] * 2

    # -----------------------------
    # TEST COLUMN VALUE PROBABILITY
    # -----------------------------

    # Crash tests for various code path. Quality of logpdf ignored.
    # Conditional local.
    curs = bdb.execute('''
        ESTIMATE PROBABILITY OF Power_watts = 800 GIVEN (Perigee_km = 980,
            Launch_Mass_kg = 890) FROM t1 LIMIT 1;
    ''')
    assert 0. <= curs.next()[0]
    # Unconditional foreign
    curs = bdb.execute('''
        ESTIMATE PROBABILITY OF Period_minutes = 1020 FROM t1 LIMIT 1;
    ''')
    assert 0. <= curs.next()[0]
    # Conditional foreign on parent and non-parents.
    curs = bdb.execute('''
        ESTIMATE PROBABILITY OF Period_minutes = 1020 GIVEN
            (Apogee_km = 38000, Eccentricity = 0.03) FROM t1 LIMIT 1;
    ''')
    assert 0 <= curs.next()[0]
    # Conditional foriegn on foreign.
    curs = bdb.execute('''
        ESTIMATE PROBABILITY OF Anticipated_Lifetime = 4.09 GIVEN
            (Class_of_Orbit = 'LEO', Purpose='Astrophysics',
                Period_minutes = 1436) FROM t1 LIMIT 1;
    ''')
    assert 0. <= curs.next()[0]
    # Categorical foreign should be less than 1.
    curs = bdb.execute('''
        ESTIMATE PROBABILITY OF Type_of_Orbit = 'Polar' FROM t1 LIMIT 1;
    ''')
    assert curs.next()[0] <= 1.
    # Query inconsistent with evidence should be 0.
    curs = bdb.execute('''
        ESTIMATE PROBABILITY OF "Type_of_Orbit" = 'Polar'
            GIVEN ("Type_of_Orbit" = 'Deep Highly Eccentric') FROM t1 LIMIT 1;
    ''')
    assert curs.next()[0] == 0.
    # In theory, query consistent with evidence should be 1, but this is very
    # hard to ensure due to stochastic sampling giving different estimates of
    # P(Y), once in joint and once in marginal Monte Carlo estimation.

    # -----------------------
    # TEST MUTUAL INFORMATION
    # -----------------------

    # Two local columns.
    curs = bdb.execute('''
        ESTIMATE MUTUAL INFORMATION OF Country_of_Contractor WITH
            longitude_radians_of_geo USING 5 SAMPLES FROM t1 LIMIT 1;
    ''')
    # XXX Small sample sizes non-deterministically produce negative MI
    assert -1 <= curs.next()[0]
    # One local and one foreign column.
    curs = bdb.execute('''
        ESTIMATE MUTUAL INFORMATION OF Period_minutes WITH
            longitude_radians_of_geo USING 5 SAMPLES FROM t1 LIMIT 1;
    ''')
    # XXX This non-deterministically fails when sample sizes are small
    # assert 0. <= curs.next()[0]
    assert float("-inf") <= curs.next()[0]
    # Two foreign columns.
    curs = bdb.execute('''
        ESTIMATE MUTUAL INFORMATION OF Period_minutes WITH
            Anticipated_Lifetime USING 5 SAMPLES FROM t1 LIMIT 1;
    ''')
    # XXX This non-deterministically fails when sample sizes are small
    # assert 0. <= curs.next()[0]
    assert float("-inf") <= curs.next()[0]

    # -----------------------
    # TEST PREDICT CONFIDENCE
    # -----------------------

    # Continuous local column.
    curs = bdb.execute('''
        INFER EXPLICIT PREDICT Dry_Mass_kg CONFIDENCE c FROM t1 LIMIT 1;
    ''')
    assert curs.next()[1] >= 0.
    # Discrete local column with no children.
    curs = bdb.execute('''
        INFER EXPLICIT PREDICT Purpose CONFIDENCE c FROM t1 LIMIT 1;
    ''')
    assert 0 <= curs.next()[1] <= 1
    # Discrete local column with children.
    curs = bdb.execute('''
        INFER EXPLICIT PREDICT Contractor CONFIDENCE c FROM t1 LIMIT 1;
    ''')
    assert 0 <= curs.next()[1] <= 1
    # Continuous foreign columns.
    curs = bdb.execute('''
        INFER EXPLICIT PREDICT Period_minutes CONFIDENCE c FROM t1 LIMIT 1;
    ''')
    assert curs.next()[1] >= 0.
    # Discrete foreign column.
    curs = bdb.execute('''
        INFER EXPLICIT PREDICT Type_of_Orbit CONFIDENCE c FROM t1 LIMIT 1;
    ''')
    assert 0 <= curs.next()[1] <= 1

    bdb.close()

def test_topological_sort():
    # Acyclic graph should return a correct topo sort.
    graph = {1:[], 2:[1], 3:[2,6], 4:[2], 5:[4], 6:[4,1]}
    topo = Composer.topological_sort(graph)
    for i in xrange(len(topo)):
        node, parents = topo[i]
        for n in topo[i+1:]:
            assert n not in parents
    # Cyclic graph should throw an error.
    graph = {1:[], 2:[1], 3:[2,4,6], 4:[2], 5:[3,4], 6:[4,5,1]}
    with pytest.raises(ValueError):
        topo = Composer.topological_sort(graph)
