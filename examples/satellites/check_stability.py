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
import string
import time

import pandas as pd
import seaborn as sns

import bayeslite

######################################################################
## Query running and result accumulation                            ##
######################################################################

model_schedule = [1,2]
model_skip = model_schedule[-1]
n_replications = 2
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
            incorporate(results,
                [((fname, model_ct, name), ress)
                 for ((model_ct, name), ress) in analyze_queries(bdb).iteritems()])
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
    model_count = 4 # TODO !
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

# These two functions are placeholders for type-dependent aggregation
# of intermediate values (e.g., counting yes and no for booleans vs
# accumulating lists of reals).
def inc_singleton(val):
    return [val]

def merge_one(so_far, val):
    so_far.extend(val)
    return so_far

def analyze_queries(bdb):
    results = {} # Keys are (model_ct, name); values are aggregated results
    for spec in model_specs():
        (low, high) = spec
        model_ct = high - low
        with model_restriction(bdb, "satellites_cc", spec):
            log("querying models %d-%d" % (low, high-1))
            for queryset in [country_purpose_queries]:
                incorporate(results, [((model_ct, name), inc_singleton(res))
                                      for (name, res) in queryset(bdb)])
    return results

######################################################################
## Visualization                                                    ##
######################################################################

def plot_results_q(results, query):
    # results :: dict (file_name, model_ct, query_name) : result_set
    data = ((fname, model_ct, value)
        for ((fname, model_ct, qname), values) in results.iteritems()
        if qname == query
        for value in values)
    df = pd.DataFrame.from_records(data, columns=["file", "n_models", "value"]).replace([False, True], [0,1])
    g = sns.FacetGrid(df, col="n_models")
    g.map(sns.violinplot, "file", "value")
    return g

def plot_results(results, basename="fig", ext=".png"):
    queries = sorted(set(qname for ((_, _, qname), _) in results.iteritems()))
    for query in queries:
        grid = plot_results_q(results, query)
        grid.fig.suptitle(query)
        figname = basename + "-" + string.replace(query, " ", "-") + ext
        grid.savefig(figname)
        log("Query '%s' results saved to %s" % (query, figname))

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
    return [ ("USA-Communications count", usa_ct),
             ("USA-Communications is top", usa_top),
             ("USA-Communications is top by 2x", by_much),
         ]

######################################################################
## Driver                                                           ##
######################################################################

plot_results(analyze_fileset(["output/satellites-2015-09-24-axch-4m-%di.bdb" % j
                              for j in range(4)]))
