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

"""Aggregation of probe results over bdb files and model numbers.

A single probe is a typed result of some computation over a BayesDB,
whose variation with model count, analysis iteration, or other setup
one is interested in studying.

A probe function is a function that accepts a bdb and computes a list
of probe results (more than one so that they can share intermediate
computations).

A probe result is represented as a 3-tuple: the name of the probe, the
type of the result (as a string), and the value of the result.

Two result types are currently supported:

- 'num' results are aggregated by maintaining a list of all of them.
- 'bool' results are aggregated by maintaining the count of True and
  False results seen.

"""

import contextlib
import logging
import random
import time

import bayeslite
from bayeslite.core import bayesdb_get_generator
from bayeslite.metamodels.crosscat import CrosscatMetamodel
from crosscat.LocalEngine import LocalEngine as CrosscatLocalEngine

start_time = time.time()
def log(msg, *irritants):
    logging.info("At %3.2fs " % (time.time() - start_time) + msg, *irritants)

def probe_fileset(files, generator, probes, model_schedule=None,
                    n_replications=None, seed=0):
    """Aggregate all the probes over all the given bdb files.

    Measure seed and model count variation within each file by varying
    model sets to which probes are applied.

    `files` is a list of filenames of saved BayesDBs to aggregate over.

    `generator` is a string naming the generator the probes read.

    `probes` is a list of probe functions.

    `model_schedule` is a list of numbers of models to probe.

    `n_replications` is the number of replications to do.

    `seed` is the initial entropy for any randomness embedded in
    BayesDB queries performed by the probe functions.  Default: 0.

    Each saved bdb must have the named generator, and that generator
    must have at least as many models as `n_replications *
    max(model_schedule)`.

    Probe functions are expected to utilize all available models to
    compute their results; application to different sets of models is
    accomplished by temporarily dropping unwanted models.

    Returns a dict mapping conditions to aggregated probe results.
    Each condition is a 3-tuple: Name of bdb file, number of models
    probed, name of probe.  The results are aggregated over
    replications: different sets of the desired number of models
    within each file.

    """
    # TODO Expect the number of models in the file to be at least
    # model_skip * n_replications; excess models are wasted.
    # TODO Default the model schedule and n_replications based on some
    # heuristic like the square root of the number of models in the
    # first file.
    model_skip = max(model_schedule)
    specs = model_specs(model_schedule, model_skip, n_replications)
    return do_probe_fileset(files, generator, probes, specs, seed)

def do_probe_fileset(files, generator, probes, specs, seed):
    # Keys are (file_name, model_ct, name); values are aggregated results
    results = {}
    prng = random.Random(seed)
    for fname in files:
        log("processing file %s" % fname)
        with bayeslite.bayesdb_open(fname, builtin_metamodels=False) as bdb:
            bayeslite.bayesdb_register_metamodel(bdb, new_cc_metamodel(prng))
            res = [((fname, model_ct, name), ress)
                   for ((model_ct, name), ress)
                   in run_probes(bdb, generator, probes, specs).iteritems()]
            incorporate(results, res)
    return results

def new_cc_metamodel(prng):
    return CrosscatMetamodel(CrosscatLocalEngine(seed=prng.randint(0, 2**31)))

def run_probes(bdb, generator, probes, specs):
    results = {} # Keys are (model_ct, name); values are aggregated results
    ct = num_models_in_bdb(bdb, generator)
    for spec in specs:
        (low, high) = spec
        model_ct = high - low
        with model_restriction(bdb, generator, spec, ct):
            log("probing models %d-%d" % (low, high-1))
            for probeset in probes:
                pres = [((model_ct, name), inc_singleton(ptype, res))
                        for (name, ptype, res) in probeset(bdb)]
                incorporate(results, pres)
    return results

def num_models_in_bdb(bdb, generator):
    generator_id = bayesdb_get_generator(bdb, generator)
    q = '''SELECT COUNT(*), MAX(modelno)
        FROM bayesdb_generator_model WHERE generator_id = ?'''
    (ct, maximum) = bdb.execute(q, (generator_id,)).next()
    assert ct == maximum+1, "Expected models to be numbered sequentially"
    return ct

def model_specs(model_schedule, model_skip, n_replications):
    def spec_at(location, size):
        """Return a 0-indexed model range
        inclusive of lower bound and exclusive of upper bound."""
        return (location * model_skip, location * model_skip + size)
    return [spec_at(location, size)
            for location in range(n_replications)
            for size in model_schedule]

@contextlib.contextmanager
def model_restriction(bdb, gen_name, spec, model_count):
    (low, high) = spec
    assert model_count >= high, "Not enough models in bdb"
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
def inc_singleton(ptype, val):
    if ptype == 'num':
        return ('num', [val])
    elif ptype == 'bool':
        if val:
            return ('bool', (1, 0))
        else:
            return ('bool', (0, 1))
    else:
        raise Exception("Unknown singleton type %s for %s" % (ptype, val))

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
