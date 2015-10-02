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

import time
import pytest

import bayeslite
from bayeslite.sqlite3_util import sqlite3_quote_name as quote

import bdbcontrib
from bdbcontrib.metamodels.composer import Composer
from bdbcontrib.predictors import random_forest
from bdbcontrib.predictors import keplers_law
from bdbcontrib.predictors import multiple_regression


# Use satellites for all tests.
satfile = '../../examples/satellites/data/satellites.csv'

# ------------------------------------------------------------------------------
# The following live outside the TestComposer class since we do not need to
# reuse a bdb instance across tests.

def test_create_generator_schema():
    bdb = bayeslite.bayesdb_open()
    bayeslite.bayesdb_read_csv_file(bdb, 'satellites', satfile, header=True,
        create=True)
    composer = Composer()
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
    # Missing conditions in randomf forest conditions should crash.
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
                    Class_of_orbit CATEGORICAL, Perigee_km NUMERICAL,
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
            INDEPENDENT(Country_of_Operator, )
        );''')
    bdb.close()

def test_register():
    bdb = bayeslite.bayesdb_open()
    composer = Composer()
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
    composer = Composer()
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


def test_composer_integration():
    # XXX Not really an integration test, all functions tested seperately.
    # But currently difficult to seperate these tests into smaller tests because
    # of their sequential nature. We will still test all internal functions
    # with different regimes of operation.

    # SETUP
    # -----
    # Dataset.
    import time
    bdb = bayeslite.bayesdb_open(str(time.time()).split('.')[0])
    bayeslite.bayesdb_read_csv_file(bdb, 'satellites', satfile, header=True,
        create=True)
    # Composer.
    composer = Composer()
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


    # TEST INITIALIZE MODELS
    # ----------------------
    bdb.execute('INITIALIZE 2 MODELS FOR t1')
    # Check number of models.
    curs = bdbcontrib.describe_generator_models(bdb, 't1')
    assert len(curs.fetchall()) == 2

    # TEST ANALYZE MODELS
    # -------------------
    bdb.execute('ANALYZE t1 FOR 2 ITERATIONS WAIT;')
    # Check number of iterations of composer.
    curs = bdbcontrib.describe_generator_models(bdb, 't1')
    for modelno, iterations in curs:
        assert iterations == 2
    # Check number of iterations of composer_cc.
    curs = bdbcontrib.describe_generator_models(bdb, 't1_cc')
    for modelno, iterations in curs:
        assert iterations == 2

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
    assert curs.next()[0] == 0.
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

# def test_column_mutual_information(self):
#     pass

# def test_column_value_probability(self):
#     pass

# def test_predict_confidence(self):
#     pass

# def test_simulate(self):
#     pass

# def test_row_column_predictive_probability(self):
#     pass
