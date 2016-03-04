from datetime import datetime
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayeslite
import bdbcontrib

from bdbcontrib.metamodels.composer import Composer
from bdbcontrib.predictors import keplers_law
from bdbcontrib.predictors import random_forest
from bdbcontrib.predictors import multiple_regression

filename = 'resources/bdb/20160304-124521.bdb'
warnings.filterwarnings('ignore')

def create_bdb():
    # Load the bdb.
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    bdb = bayeslite.bayesdb_open('resources/bdb/%s.bdb' % timestamp)

    # Load satellites data.
    bayeslite.bayesdb_read_csv_file(bdb, 'satellites',
        'resources/satellites.csv', header=True, create=True)
    bdbcontrib.nullify(bdb, 'satellites', '')

    # Register MML models.
    composer = Composer()
    composer.register_foreign_predictor(keplers_law.KeplersLaw)
    composer.register_foreign_predictor(random_forest.RandomForest)
    composer.register_foreign_predictor(multiple_regression.MultipleRegression)
    bayeslite.bayesdb_register_metamodel(bdb, composer)
    return bdb

def load_bdb(filename):
    bdb = bayeslite.bayesdb_open(filename)
    composer = Composer()
    composer.register_foreign_predictor(keplers_law.KeplersLaw)
    composer.register_foreign_predictor(random_forest.RandomForest)
    composer.register_foreign_predictor(multiple_regression.MultipleRegression)
    bayeslite.bayesdb_register_metamodel(bdb, composer)
    return bdb

def initialize_analyse_satcomp(bdb):
    # Create compositor gpm.
    bdb.execute('''
        CREATE GENERATOR satcomp FOR satellites USING composer(
            default (
                Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
                Users CATEGORICAL, Purpose CATEGORICAL,
                Type_of_Orbit CATEGORICAL, Perigee_km NUMERICAL,
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
                Class_of_orbit CATEGORICAL
                    GIVEN Apogee_km, Perigee_km,
                        Eccentricity, Period_minutes, Launch_Mass_kg,
                        Power_watts, Anticipated_Lifetime, Type_of_Orbit
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
        );''')

    print 'Initializing'
    bdb.execute('INITIALIZE 1 MODEL FOR satcomp;')
    print 'Analyze'
    bdb.execute('ANALYZE satcomp FOR 1500 ITERATION WAIT;')

def initialize_analyse_satcc(bdb):
    # Create compositor gpm.
    bdb.execute('''
        CREATE GENERATOR satcc FOR satellites USING crosscat(
            Country_of_Operator CATEGORICAL, Operator_Owner CATEGORICAL,
            Users CATEGORICAL, Purpose CATEGORICAL,
            Type_of_Orbit CATEGORICAL, Perigee_km NUMERICAL,
            Apogee_km NUMERICAL, Eccentricity NUMERICAL,
            Launch_Mass_kg NUMERICAL, Dry_Mass_kg NUMERICAL,
            Power_watts NUMERICAL, Date_of_Launch NUMERICAL,
            Contractor CATEGORICAL,
            Country_of_Contractor CATEGORICAL, Launch_Site CATEGORICAL,
            Launch_Vehicle CATEGORICAL,
            Source_Used_for_Orbital_Data CATEGORICAL,
            longitude_radians_of_geo NUMERICAL,
            Inclination_radians NUMERICAL,
            Class_of_orbit CATEGORICAL,
            Period_minutes NUMERICAL,
            Anticipated_Lifetime NUMERICAL,
        );''')

    print 'Initializing'
    bdb.execute('INITIALIZE 1 MODEL FOR satcc;')
    print 'Analyze'
    bdb.execute('ANALYZE satcc FOR 1500 ITERATION WAIT;')

def query_period_perigee_given_purpose(bdb, gpm):
    tech_dev_bql = '''CREATE TABLE tech_dev_{gpm} AS
            SIMULATE period_minutes, perigee_km FROM
            {gpm} GIVEN purpose = "Technology Development" LIMIT 100;'''

    remote_sense_bql = '''CREATE TABLE remote_sense_{gpm} AS
            SIMULATE period_minutes, perigee_km FROM
            {gpm} GIVEN purpose = "Remote Sensing" LIMIT 100;'''

    comm_bql = '''CREATE TABLE comm_{gpm} AS
            SIMULATE period_minutes, perigee_km FROM
            {gpm} GIVEN purpose = "Communications" LIMIT 100;'''

    for bql in [tech_dev_bql, remote_sense_bql, comm_bql]:
        bdb.execute(bql.format(gpm=gpm))

def plot_period_perigee_given_purpose(bdb, gpm):
    "Plot Period, Perigee given Purpose."
    matplotlib.rcParams['legend.fontsize'] = 14
    matplotlib.rcParams['axes.titlesize'] = 18
    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16

    tech_dev = np.asarray(
        bdb.sql_execute(
            '''SELECT * FROM tech_dev_{}'''.format(gpm)).fetchall())
    remote_sense = np.asarray(
        bdb.sql_execute(
            '''SELECT * FROM remote_sense_{}'''.format(gpm)).fetchall())
    comm = np.asarray(
        bdb.sql_execute(
            '''SELECT * FROM comm_{}'''.format(gpm)).fetchall())

    _, ax = plt.subplots()
    ax.scatter(tech_dev[:,1], tech_dev[:,0], color='blue',
        label='Technology Development [Primarily LEO]')
    ax.scatter(remote_sense[:,1], remote_sense[:,0], color='green',
        label='Remote Sensing [Primarily LEO]')
    ax.scatter(comm[:,1], comm[:,0], color='red',
        label='Communications [Primarly GEO]')

    ax.set_title('SIMULATE Perigee_km, Period_minutes GIVEN Purpose LIMIT 100')
    ax.set_xlabel('Perigee [km]')
    ax.set_ylabel('Period [Minutes]')
    ax.set_xlim([-200, 48000])
    ax.set_ylim([-20, 1700])
    ax.grid()
    ax.legend(loc='upper left', framealpha=0, prop={'weight':'bold'})

def sklearn_analysis():
    # Load the dataset.
    dataset = pd.read_csv('resources/satellites.csv')

bdb = load_bdb(filename)
