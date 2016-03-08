# -*- coding: utf-8 -*-
import warnings
from datetime import datetime
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import bayeslite
import bdbcontrib

from bayeslite.metamodels.crosscat import CrosscatMetamodel
from bdbcontrib import plot_utils as pu

from bdbcontrib import query
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
    import crosscat.MultiprocessingEngine as ccme
    bdb = bayeslite.bayesdb_open(pathname=filename,
        builtin_metamodels=False)
    crosscat = ccme.MultiprocessingEngine(cpu_count=None)
    metamodel = CrosscatMetamodel(crosscat)
    bayeslite.bayesdb_register_metamodel(bdb, metamodel)
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
    bdb.execute('INITIALIZE 64 MODELS FOR satcc;')
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

def query_dry_mass_mutual_information(bdb):
    query(bdb, '''
        CREATE TABLE dry_mass_mi AS ESTIMATE c.name,
            MUTUAL INFORMATION WITH dry_mass_kg AS "MI(col:dry_mass) [linfoot]"
        FROM COLUMNS OF satcc;''')

def query_select_country_purpose_given_geo(bdb):
    country_purpose_select = query(bdb, '''
        CREATE TABLE IF NOT EXISTS country_purpose_select AS
            SELECT country_of_operator || "--" || purpose AS "Country-Purpose",
                COUNT("Country-Purpose") AS frequency
        FROM satellites
            WHERE Class_of_orbit = "GEO"
            GROUP BY "Country-Purpose"
            ORDER BY frequency DESC
        LIMIT 20
        ''')

def query_simulate_country_purpose_given_geo(bdb):
    query(bdb, '''
        CREATE TABLE country_purpose_sim_raw AS
            SIMULATE country_of_operator, purpose
        FROM satcc GIVEN Class_of_orbit = 'GEO' LIMIT 1000;
    ''');
    query(bdb,'''
        CREATE TABLE country_purpose_sim AS
            SELECT country_of_operator || "--" || purpose AS "Country-Purpose",
                COUNT("Country-Purpose") AS frequency
            FROM country_purpose_sim_raw
                GROUP BY "Country-Purpose"
                ORDER BY frequency DESC
            LIMIT 20;
    ''')

def query_simulate_country_purpose_given_geo_dm(bdb):
    query(bdb, '''
        CREATE TABLE country_purpose_dm_sim_raw AS
            SIMULATE country_of_operator, purpose
        FROM satcc GIVEN Class_of_orbit = "GEO", Dry_mass_kg = 500 LIMIT 1000;
    ''');
    query(bdb,'''
        CREATE TABLE country_purpose_dm_sim AS
            SELECT country_of_operator || "--" || purpose AS "Country-Purpose",
                COUNT("Country-Purpose") AS frequency
            FROM country_purpose_dm_sim_raw
                GROUP BY "Country-Purpose"
                ORDER BY frequency DESC
            LIMIT 20;
    ''')

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

def plot_apogee_perigee_colorby_class(bdb):
    return bdbcontrib.pairplot(bdb,
        'SELECT apogee_km, perigee_km, class_of_orbit FROM satellites;',
        colorby='class_of_orbit')

def plot_country_purpose_mass(bdb):
    bdb.sql_execute('''
            CREATE TEMP TABLE IF NOT EXISTS country_purpose_dm AS
                SELECT country_of_operator || "--" || purpose
                    AS "Country-Purpose", dry_mass_kg
                FROM satellites WHERE Class_of_orbit = "GEO" AND
                "Country-Purpose"
                    IN (SELECT "Country-Purpose" FROM country_purpose_select)
        ''')
    fig = bdbcontrib.pairplot(bdb, 'SELECT * FROM country_purpose_dm')
    ax = fig.get_axes()
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(90)
    fig.set_tight_layout(True)
    return fig

def plot_country_purpose_given_geo_all(bdb):
    # Compare country_purpose_select to country_purpose_simulate.
    cp_select = query(bdb, 'SELECT * FROM country_purpose_select;')
    cp_simulate = query(bdb, 'SELECT * FROM country_purpose_sim;')
    cp_simulate_dm = query(bdb, 'SELECT * FROM country_purpose_dm_sim;')

    # Normalize.
    cp_select['frequency'] /= 1167.
    cp_simulate['frequency'] /= 1000.
    cp_simulate_dm['frequency'] /= 1000.

    # Rename the columns.
    cp_select.columns = ['Country-Purpose', 'Probability']
    cp_simulate.columns = ['Country-Purpose', 'Probability']
    cp_simulate_dm.columns = ['Country-Purpose', 'Probability']

    # Histograms for raw data.
    barplot(cp_select.sort(columns=['Probability']), color='b',
        label='SELECT country-purpose GIVEN geo')
    barplot(cp_simulate.sort(columns=['Probability']), color='r',
        label='SIMULATE country-purpose GIVEN geo')
    barplot(cp_simulate_dm.sort(columns=['Probability']), color='g',
        label='SIMULATE country-purpose GIVEN geo, dry_mass_kg = 500')

    # Histogram for raw vs simulated.
    joined_select_sim = combine_dataframes(cp_select, cp_simulate)
    joined_select_sim.columns = ['Country-Purpose',
        'SELECT country-purpose GIVEN geo', 'SIMULATE country-purpose GIVEN geo']
    barplot_overlay(joined_select_sim, colors=['b','r'])

    # Histogram for raw vs simulated.
    joined_sim_dm = combine_dataframes(cp_simulate, cp_simulate_dm)
    joined_sim_dm.columns = ['Country-Purpose',
        'SIMULATE country-purpose GIVEN geo',
        'SIMULATE country-purpose GIVEN geo, dry_mass_kg = 500',]
    barplot_overlay(joined_sim_dm, colors=['r','g'])

def combine_dataframes(A, B):
    A_zero_uniq = set(A.ix[:,0])
    B_zero_uniq = set(B.ix[:,0])
    A = A.set_index(A.columns[0])
    B = B.set_index(B.columns[0])
    C = pd.DataFrame(columns=['a','b'],
        index=set.union(A_zero_uniq, B_zero_uniq))
    for k in C.index:
        a_val = A.loc[k].iloc[0] if k in A.index else 0
        b_val = B.loc[k].iloc[0] if k in B.index else 0
        C.loc[k] = [a_val, b_val]
    return C.sort(columns=['b'], ascending=True).reset_index()

def barplot(df, color='#333333', label=None):
    if df.shape[1] != 2:
        raise ValueError('Need two columns.')
    figure, ax = plt.subplots()
    ax.barh(range(df.shape[0]), df.ix[:,1].values,
        color=color, edgecolor=color, alpha=.7, label=label)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.ix[:,0].values, fontweight='bold')
    ax.set_ylabel(df.columns[0], fontweight='bold')
    ax.set_xlabel(df.columns[1], fontweight='bold')
    figure.set_tight_layout(True)
    ax.legend(loc='lower right')
    return figure

def barplot_overlay(df, colors=['g','r']):
    if df.shape[1] != 3:
        raise ValueError('Need three columns.')
    figure, ax = plt.subplots()
    ind = range(df.shape[0])
    width = .35
    ax.barh(ind, df.ix[:,1].values, width,
        color=colors[0], alpha=.7, label=df.columns[1])
    ax.barh([y + width for y in ind], df.ix[:,2].values, width,
        color=colors[1], alpha=.7, label=df.columns[2])
    ax.set_yticks([y + width for y in ind])
    ax.set_yticklabels(df.ix[:,0].values, fontweight='bold')
    ax.set_ylabel('Country-Purpose', fontweight='bold')
    ax.set_xlabel('Probability', fontweight='bold')
    ax.legend(loc='lower right')
    figure.set_tight_layout(True)
    return figure

bdb = load_bdb(filename)

# bdb.sql_execute('DROP TABLE IF EXISTS dry_mass_mi')
# bdb.sql_execute('DROP TABLE IF EXISTS country_purpose_select')
# bdb.sql_execute('DROP TABLE IF EXISTS country_purpose_sim_raw')
# bdb.sql_execute('DROP TABLE IF EXISTS country_purpose_sim')
# bdb.sql_execute('DROP TABLE IF EXISTS country_purpose_dm_sim_raw')
# bdb.sql_execute('DROP TABLE IF EXISTS country_purpose_dm_sim')

# query_dry_mass_mutual_information(bdb)
# query_select_country_purpose_given_geo(bdb)
# query_simulate_country_purpose_given_geo(bdb)
# query_simulate_country_purpose_given_geo_dm(bdb)
