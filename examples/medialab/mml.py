from datetime import datetime
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

    bdb.execute(tech_dev_bql.format(gpm='satcomp'))
    bdb.execute(remote_sense_bql.format(gpm='satcomp'))
    bdb.execute(comm_bql.format(gpm='satcomp'))

    bdb.execute(tech_dev_bql.format(gpm='satcc'))
    bdb.execute(remote_sense_bql.format(gpm='satcc'))
    bdb.execute(comm_bql.format(gpm='satcc'))

def plot_TP_given_Purpose(bdb, create=False):
    "Plot Period, Perigee given Purpose."
    matplotlib.rcParams['legend.fontsize'] = 14
    matplotlib.rcParams['axes.titlesize'] = 18
    matplotlib.rcParams['axes.labelsize'] = 18
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['xtick.labelsize'] = 16
    matplotlib.rcParams['ytick.labelsize'] = 16

    # else:
    #     tech_dev = np.loadtxt('resources/TPgPurpose/tech_dev_cc')
    #     remote_sense = np.loadtxt('resources/TPgPurpose/remote_sense_cc')
    #     communications = np.loadtxt('resources/TPgPurpose/communications_cc')

    # purpose_class_pairs = bdb.execute('''SELECT purpose, class_of_orbit,
    #     count(*) AS count FROM satellites GROUP BY purpose, class_of_orbit
    #     ORDER BY count DESC;''').fetchall()

    _, ax = plt.subplots()
    ax.scatter(tech_dev[:,1], tech_dev[:,0], color='blue',
        label='Technology Development [Primarily LEO]')
    ax.scatter(remote_sense[:,1], remote_sense[:,0], color='green',
        label='Remote Sensing [Primarily LEO]')
    ax.scatter(communications[:,1], communications[:,0], color='red',
        label='Communications [Primarly GEO]')

    ax.set_title('SIMULATE Perigee_km, Period_minutes GIVEN Purpose LIMIT 100')
    ax.set_xlabel('Perigee [km]')
    ax.set_ylabel('Period [Minutes]')
    ax.set_xlim([-200, 48000])
    ax.set_ylim([-20, 1700])
    ax.legend(loc='upper left', framealpha=0, prop={'weight':'bold'})
    ax.grid()

def plot_T_given_CO(bdb, create=False):
    "Plot Period given Class of Orbit."
    if create:
        leo = bdb.execute('''SELECT Period_minutes FROM satellites WHERE
            Class_of_orbit = "LEO"''').fetchall()
        geo = bdb.execute('''SELECT Period_minutes FROM satellites WHERE
            Class_of_orbit = "GEO"''').fetchall()
        meo = bdb.execute('''SELECT Period_minutes FROM satellites WHERE
            Class_of_orbit = "MEO"''').fetchall()
        elliptical = bdb.execute('''SELECT Period_minutes FROM satellites
            WHERE Class_of_orbit = Elliptical''').fetchall()
        samples_leo = bdb.execute('''SIMULATE Period_minutes FROM satcomp GIVEN
            Class_of_orbit = "LEO" LIMIT 100;''').fetchall()
        samples_geo = bdb.execute('''SIMULATE Period_minutes FROM satcomp GIVEN
            Class_of_orbit = "GEO" LIMIT 100;''').fetchall()
        samples_meo = bdb.execute('''SIMULATE Period_minutes FROM satcomp GIVEN
            Class_of_orbit = "MEO" LIMIT 100;''').fetchall()
        samples_elliptical = bdb.execute('''SIMULATE Period_minutes FROM satcomp
            GIVEN Class_of_orbit = "Elliptical" LIMIT 100;''').fetchall()
    else:
        leo = np.loadtxt('resources/TgCO/leo')
        geo = np.loadtxt('resources/TgCO/geo')
        meo = np.loadtxt('resources/TgCO/meo')
        elliptical = np.loadtxt('resources/TgCO/elliptical')
        samples_leo = list(np.loadtxt('resources/TgCO/samples_leo'))
        samples_geo = list(np.loadtxt('resources/TgCO/samples_geo'))
        samples_meo = list(np.loadtxt('resources/TgCO/samples_meo'))
        samples_elliptical = list(np.loadtxt('resources/TgCO/samples_elliptical'))

    fig, ax = plt.subplots(2,1)

    ax[0].hlines(leo, xmin=0, xmax=1, label='LEO', color='blue')
    ax[0].hlines(geo, xmin=1, xmax=2, label='MEO', color='red')
    ax[0].hlines(meo, xmin=2, xmax=3, label='GEO', color='green')
    ax[0].hlines(elliptical, xmin=3, xmax=4, label='Elliptical', color='black')
    ax[0].set_title('SELECT Period_minutes, Class_of_orbit FROM'
        ' satellites ORDER BY Class_of_orbit''', fontweight='bold',size=16)

    ax[1].hlines(samples_leo[:10]+samples_leo[-10:], xmin=0, xmax=1,
        label='LEO', color='blue')
    ax[1].hlines(samples_meo[:10]+samples_meo[-10:],
        xmin=1, xmax=2, label='MEO', color='red')
    ax[1].hlines(samples_geo[60:70] + samples_geo[80:90],
        xmin=2, xmax=3, label='GEO', color='green')
    ax[1].hlines(samples_elliptical[:10]+samples_elliptical[-10:],
        xmin=3, xmax=4, label='Elliptical', color='black')
    ax[1].set_title('''
        SIMULATE Period_minutes FROM satellites GIVEN Class_of_orbit''',
        fontweight='bold', size=16)

    for a in ax:
        a.set_xlim([0,4])
        a.set_xlim([0,4])
        a.set_ylim([0,4000])
        a.set_ylim([0,4000])
        a.set_xticks([0.5, 1.5, 2.5, 3.5])
        a.set_xticklabels(['LEO','MEO','GEO','Elliptical'])
        a.set_ylabel('Period (minutes)', fontweight='bold', size=16)
        a.grid()
        a.grid()
