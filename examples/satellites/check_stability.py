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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import math
import os
import re
import string
import warnings

import pandas as pd
import seaborn as sns

import sys
sys.path.append("..")
from aggregation import log, analyze_fileset

######################################################################
## Visualization                                                    ##
######################################################################

# results :: {(fname, model_ct, query_name) : tagged aggregated value}

plot_out_dir = "figures"

def num_replications(results):
    replication_counts = \
        set((len(l) for (qtype, l) in results.values()
             if qtype == 'num')).union(
        set((t+f for (qtype, l) in results.values()
             if qtype == 'bool'
             for (t, f) in [l])))
    replication_count = next(iter(replication_counts))
    if len(replication_counts) > 1:
        msg = "Non-rectangular results found; replication counts range " \
              "over %s, using %s" % (replication_counts, replication_count)
        warnings.warn(msg)
    return replication_count

def analysis_count_from_file_name(fname):
    return int(re.match(r'.*-[0-9]*m-([0-9]*)i.bdb', fname).group(1))

def plot_results_numerical(results, query):
    # results :: dict (file_name, model_ct, query_name) : tagged result_set
    data = ((analysis_count_from_file_name(fname), model_ct, value)
        for ((fname, model_ct, qname), (qtype, values)) in results.iteritems()
        if qname == query and qtype == 'num'
        for value in values)
    cols = ["num iterations", "n_models", "value"]
    df = pd.DataFrame.from_records(data, columns=cols) \
                     .sort(["num iterations", "n_models"])
    g = sns.FacetGrid(df, col="n_models", size=5, col_wrap=3)
    g.map(sns.violinplot, "num iterations", "value")
    return g

def plot_results_boolean(results):
    # results :: dict (file_name, model_ct, query_name) : tagged result_set
    data = ((analysis_count_from_file_name(fname), model_ct, qname, float(t)/(t+f))
        for ((fname, model_ct, qname), (qtype, value)) in results.iteritems()
        if qtype == 'bool'
        for (t, f) in [value])
    cols = ["num iterations", "n_models", "query", "freq"]
    df = pd.DataFrame.from_records(data, columns=cols) \
                     .sort(["query", "num iterations", "n_models"])
    g = sns.FacetGrid(df, col="n_models", hue="query", size=5, col_wrap=3)
    g.map(plt.plot, "num iterations", "freq").add_legend()
    return g

def plot_results(results, ext=".png"):
    if not os.path.exists(plot_out_dir):
        os.makedirs(plot_out_dir)
    replications = num_replications(results)
    queries = sorted(set((qname, qtype)
                         for ((_, _, qname), (qtype, _)) in results.iteritems()))
    for query, qtype in queries:
        if not qtype == 'num': continue
        grid = plot_results_numerical(results, query)
        grid.fig.suptitle(query + ", %d replications" % replications)
        # XXX Actually shell quote the query name
        figname = string.replace(query, " ", "-").replace("/", "") + ext
        savepath = os.path.join(plot_out_dir, figname)
        grid.savefig(savepath)
        plt.close(grid.fig)
        log("Query '%s' results saved to %s" % (query, savepath))
    grid = plot_results_boolean(results)
    grid.fig.suptitle("Boolean queries, %d replications" % replications)
    figname = "boolean-queries" + ext
    savepath = os.path.join(plot_out_dir, figname)
    grid.savefig(savepath)
    plt.close(grid.fig)
    log("Boolean query results saved to %s" % (savepath,))

######################################################################
## Queries                                                          ##
######################################################################

def country_purpose_queries(bdb):
    bdb.execute('''
        CREATE TEMP TABLE satellite_purpose AS
        SIMULATE country_of_operator, purpose FROM satellites_cc
        GIVEN Class_of_orbit = 'GEO', Dry_mass_kg = 500
        LIMIT 1000;
    ''')
    usa_ct_sql = '''
        SELECT COUNT(*) FROM satellite_purpose
        WHERE country_of_operator = 'USA'
          AND purpose = 'Communications'
    '''
    (usa_ct,) = bdb.execute(usa_ct_sql).next()
    hist_sql = '''
        SELECT country_of_operator, purpose, COUNT(*) AS frequency
        FROM satellite_purpose
        GROUP BY country_of_operator, purpose
        ORDER BY frequency DESC
        LIMIT 2
    '''
    results = bdb.execute(hist_sql).fetchall()
    usa_top = results[0][0] == 'USA' and results[0][1] == 'Communications'
    by_much = results[0][2] > 2*results[1][2]
    return [ ("USA-Communications count", "num", usa_ct),
             ("USA-Communications is top", "bool", usa_top),
             ("USA-Communications is top by 2x", "bool", by_much),
         ]

def unlikely_periods_queries(bdb):
    bdb.execute('''
        CREATE TEMP TABLE unlikely_periods AS
        ESTIMATE name, class_of_orbit, period_minutes,
        PREDICTIVE PROBABILITY OF period_minutes
            AS "Relative Probability of Period"
        FROM satellites_cc;
    ''')
    def prob_of(name):
        query = '''
            SELECT "Relative Probability of Period"
            FROM unlikely_periods
            WHERE name LIKE '%s'
        '''
        (ans,) = bdb.sql_execute(query % name).next()
        return ans
    top_ten = bdb.execute('''
        SELECT name FROM unlikely_periods
        WHERE class_of_orbit = 'GEO' AND period_minutes IS NOT NULL
        ORDER BY "Relative Probability of Period" ASC LIMIT 10;
    ''').fetchall()
    def in_top_ten(name):
        return any((n.startswith(name) for (n,) in top_ten))
    def in_top_five(name):
        return any((n.startswith(name) for (n,) in top_ten[0:5]))
    return [ ("DSP 20 period prob",    'num', prob_of('DSP 20 (USA 149)%')),
             ("SDS III-6 period prob", 'num', prob_of('SDS III-6%')),
             ("SDS III-7 period prob", 'num', prob_of('SDS III-7%')),
             ("Orion 6 period prob",   'num', prob_of('Advanced Orion 6%')),
             ("DSP 20 in top 10 unlikely periods",
              'bool', in_top_ten('DSP 20 (USA 149)')),
             ("SDS III-6 in top 10 unlikely periods",
              'bool', in_top_ten('SDS III-6')),
             ("SDS III-7 in top 10 unlikely periods",
              'bool', in_top_ten('SDS III-7')),
             ("Orion 6 in top 10 unlikely periods",
              'bool', in_top_ten('Advanced Orion 6')),
             ("DSP 20 in top 5 unlikely periods",
              'bool', in_top_five('DSP 20 (USA 149)')),
             ("SDS III-6 in top 5 unlikely periods",
              'bool', in_top_five('SDS III-6')),
             ("SDS III-7 in top 5 unlikely periods",
              'bool', in_top_five('SDS III-7')),
             ("Orion 6 in top 5 unlikely periods",
              'bool', in_top_five('Advanced Orion 6')),
         ]

def orbit_type_imputation_queries(bdb):
    bdb.execute('''
        CREATE TEMP TABLE inferred_orbit AS
        INFER EXPLICIT
        anticipated_lifetime, perigee_km, period_minutes, class_of_orbit,
        PREDICT type_of_orbit
            AS inferred_orbit_type
            CONFIDENCE inferred_orbit_type_conf
        FROM satellites_cc
        WHERE type_of_orbit IS NULL;
    ''')
    query = '''
        SELECT i.class_of_orbit, i.inferred_orbit_type,
            COUNT(*) AS count, AVG(i.inferred_orbit_type_conf) AS mean,
            AVG((i.inferred_orbit_type_conf - sub.answer) * (i.inferred_orbit_type_conf - sub.answer)) as variance
        FROM inferred_orbit AS i,
            (SELECT i.class_of_orbit, i.inferred_orbit_type,
                AVG(i.inferred_orbit_type_conf) AS answer
            FROM inferred_orbit as i
            GROUP BY i.class_of_orbit, i.inferred_orbit_type) AS sub
        WHERE i.class_of_orbit == sub.class_of_orbit
            AND i.inferred_orbit_type == sub.inferred_orbit_type
        GROUP BY i.class_of_orbit, i.inferred_orbit_type
        ORDER BY i.class_of_orbit, i.inferred_orbit_type
    '''
    answers = dict(((cl, tp), (ct, avg, var))
                   for (cl, tp, ct, avg, var)
                   in bdb.execute(query).fetchall())
    def query_gen():
        for cl in ["Elliptical", "LEO"]:
            for tp in ["Deep Highly Eccentric",
                       "Intermediate",
                       "Molniya",
                       "Sun-Synchronous",
                       "Cislunar",
                       "N/A"]:
                (ct, avg, var) = answers.get((cl, tp), (0,0,0))
                yield ("%s %s ct imputed instances" % (cl, tp),
                       'num', float(ct))
                yield ("%s %s mean inference confidence" % (cl, tp),
                       'num', avg)
                yield ("%s %s inference confidence stddev" % (cl, tp),
                       'num', math.sqrt(var))
    return list(query_gen())

######################################################################
## Driver                                                           ##
######################################################################

import cPickle as pickle # json doesn't like tuple dict keys

import glob

def save_query_results(filename):
    # files = glob.glob("output/*i.bdb")
    files = ["output/satellites-2015-09-30-axch-60m-4i.bdb",
             "output/satellites-2015-09-30-axch-60m-8i.bdb"]
    results = analyze_fileset(
        files,
        [country_purpose_queries,
         unlikely_periods_queries,
         orbit_type_imputation_queries],
        model_schedule = [1,3,6],
        n_replications = 10)

    with open(filename, "w") as f:
        pickle.dump(results, f)
    log("Saved query results to %s" % filename)

def plot_query_results(filename):
    log("Loading query results from %s" % filename)
    with open(filename, "r") as f:
        results = pickle.load(f)
    plot_results(results)

save_query_results("results.pkl")
plot_query_results("results.pkl")
