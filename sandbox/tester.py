import os

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

bdb = bayeslite.bayesdb_open()
bayeslite.bayesdb_read_csv_file(bdb, 'satellites', PATH_SATELLITES_CSV,
    header=True, create=True)
composer = Composer(n_samples=5)
bayeslite.bayesdb_register_metamodel(bdb, composer)

# Registered foreign predictor should work.
composer.register_foreign_predictor(random_forest.RandomForest)
bdb.execute('''
    CREATE GENERATOR t5 FOR satellites USING composer(
        default (
            Apogee_km NUMERICAL, Perigee_km NUMERICAL
        ),
        random_forest (
            GENERATE (Users CATEGORICAL)
                GIVEN (Apogee_km, Perigee_km)
            IMPUTE (default)
        )
    );''')
