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

"""Aggregation of probe results over bdb files and model numbers."""

import contextlib
import time

import bayeslite

then = time.time()
def log(msg):
    print "At %3.2fs" % (time.time() - then), msg

def analyze_fileset(files, probes, model_schedule=None,
                    n_replications=None):
    """Aggregate all the queries over all the given bdb files.

    Do seed and model count variation within each file by varying
    model specs; assume the files represent analysis iteration
    variation and report accordingly.

    Alternative: could just assume the file name describes the file
    opaquely.

    """
    # TODO Expect the number of models in the file to be at least
    # model_skip * n_replications; excess models are wasted.
    # TODO Default the model schedule and n_replications based on the
    # square root of the number of models in the first file
    model_skip = model_schedule[-1]
    specs = model_specs(model_schedule, model_skip, n_replications)
    return do_analyze_fileset(files, probes, specs)

def do_analyze_fileset(files, probes, specs):
    # Keys are (file_name, model_ct, name); values are aggregated results
    results = {}
    for fname in files:
        log("processing file %s" % fname)
        with bayeslite.bayesdb_open(fname) as bdb:
            res = [((fname, model_ct, name), ress)
                   for ((model_ct, name), ress)
                   in analyze_queries(bdb, probes, specs).iteritems()]
            incorporate(results, res)
    return results

def model_specs(model_schedule, model_skip, n_replications):
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

def analyze_queries(bdb, probes, specs):
    results = {} # Keys are (model_ct, name); values are aggregated results
    for spec in specs:
        (low, high) = spec
        model_ct = high - low
        with model_restriction(bdb, "satellites_cc", spec):
            log("querying models %d-%d" % (low, high-1))
            for queryset in probes:
                qres = [((model_ct, name), inc_singleton(qtype, res))
                        for (name, qtype, res) in queryset(bdb)]
                incorporate(results, qres)
    return results
