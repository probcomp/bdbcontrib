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

import contextlib
import math
import re
import string
import time
import warnings

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import bayeslite

######################################################################
## Query running and result accumulation                            ##
######################################################################

model_schedule = [1,3,6]
model_skip = model_schedule[-1]
n_replications = 10
# TODO Expect the number of models in the file to be at least
# model_skip * n_replications; excess models are wasted.

then = time.time()
def log(msg):
    print "At %3.2fs" % (time.time() - then), msg

def analyze_fileset(files):
    """Aggregate all the queries over all the given bdb files.

    Do seed and model count variation within each file by varying
    model specs; assume the files represent analysis iteration
    variation and report accordingly.

    Alternative: could just assume the file name describes the file
    opaquely.

    """
    # Keys are (file_name, model_ct, name); values are aggregated results
    results = {}
    for fname in files:
        log("processing file %s" % fname)
        with bayeslite.bayesdb_open(fname) as bdb:
            res = [((fname, model_ct, name), ress)
                   for ((model_ct, name), ress)
                   in analyze_queries(bdb).iteritems()]
            incorporate(results, res)
    return results

def model_specs():
    def spec_at(location, size):
        """Return a 0-indexed model range
        inclusive of lower bound and exclusive of upper bound."""
        return (location * model_skip, location * model_skip + size)
    return [spec_at(location, size)
            for location in range(n_replications)
            for size in model_schedule]

@contextlib.contextmanager
def model_restriction(bdb, gen_name, spec):
    (low, high) = spec
    model_count = 60 # TODO !
    with bdb.savepoint_rollback():
        if low > 0:
            bdb.execute('''DROP MODELS 0-%d FROM %s''' % (low-1, gen_name))
        if model_count > high:
            bdb.execute('''DROP MODELS %d-%d FROM %s'''
                        % (high, model_count-1, gen_name))
        yield

def incorporate(running, new):
    for (key, more) in new:
        if key in running:
            running[key] = merge_one(running[key], more)
        else:
            running[key] = more

# Type-dependent aggregation of intermediate values (e.g., counting
# yes and no for booleans vs accumulating lists of reals).
def inc_singleton(qtype, val):
    if qtype == 'num':
        return ('num', [val])
    elif qtype == 'bool':
        if val:
            return ('bool', (1, 0))
        else:
            return ('bool', (0, 1))
    else:
        raise Exception("Unknown singleton type %s for %s" % (qtype, val))

def merge_one(so_far, val):
    (tp1, item1) = so_far
    (tp2, item2) = val
    if tp1 == 'num' and tp2 == 'num':
        item1.extend(item2)
        return ('num', item1)
    elif tp1 == 'bool' and tp2 == 'bool':
        (t1, f1) = item1
        (t2, f2) = item2
        return ('bool', (t1+t2, f1+f2))
    else:
        raise Exception("Type mismatch in merge into %s of %s" % (so_far, val))

def analyze_queries(bdb):
    results = {} # Keys are (model_ct, name); values are aggregated results
    for spec in model_specs():
        (low, high) = spec
        model_ct = high - low
        with model_restriction(bdb, "satellites_cc", spec):
            log("querying models %d-%d" % (low, high-1))
            for queryset in [country_purpose_queries,
                             unlikely_periods_queries,
                             orbit_type_imputation_queries]:
                qres = [((model_ct, name), inc_singleton(qtype, res))
                        for (name, qtype, res) in queryset(bdb)]
                incorporate(results, qres)
    return results

######################################################################
## Visualization                                                    ##
######################################################################

# results :: {(fname, model_ct, query_name) : tagged aggregated value}

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
    g = sns.FacetGrid(df, col="n_models")
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
                     .sort(["num iterations", "n_models"])
    g = sns.FacetGrid(df, col="n_models", hue="query")
    g.map(plt.plot, "num iterations", "freq").add_legend()
    return g

def plot_results(results, basename="fig", ext=".png"):
    replications = num_replications(results)
    queries = sorted(set((qname, qtype)
                         for ((_, _, qname), (qtype, _)) in results.iteritems()))
    for query, qtype in queries:
        if not qtype == 'num': continue
        grid = plot_results_numerical(results, query)
        grid.fig.suptitle(query + ", %d replications" % replications)
        figname = basename + "-" + string.replace(query, " ", "-") + ext
        grid.savefig(figname)
        log("Query '%s' results saved to %s" % (query, figname))
    grid = plot_results_boolean(results)
    grid.fig.suptitle("Boolean queries, %d replications" % replications)
    figname = basename + "-boolean-queries" + ext
    grid.savefig(figname)
    log("Boolean query results saved to %s" % (figname,))

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
    def query_gen():
        for (cl, tp, _, avg, var) in bdb.execute(query).fetchall():
            yield ("%s %s mean inference confidence" % (cl, tp), 'num', avg)
            yield ("%s %s inference confidence stddev" % (cl, tp),
                   'num', math.sqrt(var))
    return list(query_gen())

######################################################################
## Driver                                                           ##
######################################################################

import cPickle as pickle # json doesn't like tuple dict keys

import glob

def save_query_results(filename):
    files = glob.glob("output/*i.bdb")
    # files = ["output/satellites-2015-09-30-axch-60m-4i.bdb"]
    results = analyze_fileset(files)

    with open(filename, "w") as f:
        pickle.dump(results, f)
    log("Saved query results to %s" % filename)

def plot_query_results(filename):
    log("Loading query results from %s" % filename)
    with open(filename, "r") as f:
        results = pickle.load(f)
    plot_results(results)

save_query_results("results.json")
plot_query_results("results.json")
