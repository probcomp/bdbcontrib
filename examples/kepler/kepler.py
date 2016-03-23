# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
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
import sys

import bayeslite

import bdbcontrib
from bdbcontrib.metamodels.composer import Composer
from bdbcontrib.predictors import random_forest
from bdbcontrib.predictors import keplers_law

# Get output directory.
if len(sys.argv) < 2:
    outdir = '.'
else:
    outdir = sys.argv[1]

# Find the satellites file.
PATH_KEPLER = os.path.dirname(os.path.abspath(__file__))
PATH_EXAMPLES = os.path.dirname(PATH_KEPLER)
PATH_SATELLITES = os.path.join(PATH_EXAMPLES, 'satellites')
PATH_SATELLITES_CSV = os.path.join(PATH_SATELLITES, 'satellites.csv')

composer = Composer()
composer.register_foreign_predictor(keplers_law.KeplersLaw)
composer.register_foreign_predictor(random_forest.RandomForest)

if os.path.exists(os.path.join(outdir, 'kepler.bdb')):
    os.remove(os.path.join(outdir, 'kepler.bdb'))

bdb = bayeslite.bayesdb_open(os.path.join(outdir, 'kepler.bdb'))
bayeslite.bayesdb_register_metamodel(bdb, composer)
bayeslite.bayesdb_read_csv_file(bdb, 'satellites', PATH_SATELLITES_CSV,
    header=True, create=True)

bdbcontrib.query(bdb, '''
    CREATE GENERATOR sat_kepler FOR satellites USING composer(
        default (
            Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
            Users CATEGORICAL, Purpose CATEGORICAL,
            Class_of_Orbit CATEGORICAL, Perigee_km NUMERICAL,
            Apogee_km NUMERICAL, Eccentricity NUMERICAL,
            Launch_Mass_kg NUMERICAL, Dry_Mass_kg NUMERICAL,
            Power_watts NUMERICAL, Date_of_Launch NUMERICAL,
            Anticipated_Lifetime NUMERICAL, Contractor CATEGORICAL,
            Country_of_Contractor CATEGORICAL, Launch_Site CATEGORICAL,
            Launch_Vehicle CATEGORICAL,
            Source_Used_for_Orbital_Data CATEGORICAL,
            longitude_radians_of_geo NUMERICAL, Inclination_radians NUMERICAL
        ),
        random_forest (
            Type_of_Orbit CATEGORICAL
                GIVEN Apogee_km, Perigee_km,
                    Eccentricity, Period_minutes, Launch_Mass_kg, Power_watts,
                    Anticipated_Lifetime, Class_of_orbit
        ),
        keplers_law (
            Period_minutes NUMERICAL
                GIVEN Perigee_km, Apogee_km
        )
    );''')

bdbcontrib.query(bdb, '''
    CREATE GENERATOR sat_default FOR satellites USING crosscat(
            Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
            Users CATEGORICAL, Purpose CATEGORICAL,
            Class_of_Orbit CATEGORICAL, Perigee_km NUMERICAL,
            Apogee_km NUMERICAL, Eccentricity NUMERICAL,
            Launch_Mass_kg NUMERICAL, Dry_Mass_kg NUMERICAL,
            Power_watts NUMERICAL, Date_of_Launch NUMERICAL,
            Anticipated_Lifetime NUMERICAL, Contractor CATEGORICAL,
            Country_of_Contractor CATEGORICAL, Launch_Site CATEGORICAL,
            Launch_Vehicle CATEGORICAL,
            Source_Used_for_Orbital_Data CATEGORICAL,
            longitude_radians_of_geo NUMERICAL, Inclination_radians NUMERICAL,
            Type_of_Orbit CATEGORICAL,
            Period_minutes NUMERICAL
    );''')

bdbcontrib.query(bdb, 'INITIALIZE 16 MODELS FOR sat_kepler')
bdbcontrib.query(bdb, 'INITIALIZE 16 MODELS FOR sat_default')

bdbcontrib.query(bdb, 'ANALYZE sat_kepler FOR 20 ITERATIONS WAIT')
bdbcontrib.query(bdb, 'ANALYZE sat_default FOR 20 ITERATIONS WAIT')

KC = bdbcontrib.query(bdb, '''
    SIMULATE Apogee_km, Perigee_km FROM sat_kepler
        GIVEN Period_minutes = 1436 LIMIT 100;''')

DC = bdbcontrib.query(bdb, '''
    SIMULATE Apogee_km, Perigee_km FROM sat_default
        GIVEN Period_minutes = 1436 LIMIT 100;''')

EC = bdbcontrib.query(bdb, '''
    SELECT Apogee_km, Perigee_km, Eccentricity FROM satellites
        WHERE Period_minutes BETWEEN 1430 AND 1440
            AND Apogee_km IS NOT NULL
            AND Perigee_km IS NOT NULL;''')

KJ = bdbcontrib.query(bdb, '''
    SIMULATE Apogee_km, Period_minutes FROM sat_kepler LIMIT 100;''')

DJ = bdbcontrib.query(bdb, '''
    SIMULATE Apogee_km, Period_minutes FROM sat_default LIMIT 100;''')

EJ = bdbcontrib.query(bdb, '''
    SELECT Apogee_km, Period_minutes FROM satellites
        WHERE Apogee_km IS NOT NULL AND Period_minutes IS NOT NULL;''')

if not os.path.exists(os.path.join(outdir, 'simulated')):
    os.mkdir(os.path.join(outdir, 'simulated'))

KC.to_csv(os.path.join(outdir, 'simulated', 'kc.csv'))
DC.to_csv(os.path.join(outdir, 'simulated', 'dc.csv'))
EC.to_csv(os.path.join(outdir, 'simulated', 'ec.csv'))
KJ.to_csv(os.path.join(outdir, 'simulated', 'kj.csv'))
DJ.to_csv(os.path.join(outdir, 'simulated', 'dj.csv'))
EJ.to_csv(os.path.join(outdir, 'simulated', 'ej.csv'))
