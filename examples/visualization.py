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

import os
import re
import string
import warnings

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

import sys
sys.path.append(".")
from aggregation import log

# results :: {(fname, model_ct, probe_name) : tagged aggregated value}

plot_out_dir = "figures"

def num_replications(results):
    replication_counts = \
        set((len(l) for (ptype, l) in results.values()
             if ptype == 'num')).union(
        set((t+f for (ptype, l) in results.values()
             if ptype == 'bool'
             for (t, f) in [l])))
    replication_count = next(iter(replication_counts))
    if len(replication_counts) > 1:
        msg = "Non-rectangular results found; replication counts range " \
              "over %s, using %s" % (replication_counts, replication_count)
        warnings.warn(msg)
    return replication_count

def analysis_count_from_file_name(fname):
    return int(re.match(r'.*-[0-9]*m-([0-9]*)i.bdb', fname).group(1))

def plot_results_numerical(results, probe):
    # results :: dict (file_name, model_ct, probe_name) : tagged result_set
    data = ((analysis_count_from_file_name(fname), model_ct, value)
        for ((fname, model_ct, pname), (ptype, values)) in results.iteritems()
        if pname == probe and ptype == 'num'
        for value in values)
    cols = ["num iterations", "n_models", "value"]
    df = pd.DataFrame.from_records(data, columns=cols) \
                     .sort(["num iterations", "n_models"])
    g = sns.FacetGrid(df, col="n_models", size=5, col_wrap=3)
    g.map(sns.violinplot, "num iterations", "value")
    return g

def plot_results_boolean(results):
    # results :: dict (file_name, model_ct, probe_name) : tagged result_set
    data = ((analysis_count_from_file_name(fname), model_ct, pname, float(t)/(t+f))
        for ((fname, model_ct, pname), (ptype, value)) in results.iteritems()
        if ptype == 'bool'
        for (t, f) in [value])
    cols = ["num iterations", "n_models", "probe", "freq"]
    df = pd.DataFrame.from_records(data, columns=cols) \
                     .sort(["probe", "num iterations", "n_models"])
    g = sns.FacetGrid(df, col="n_models", hue="probe", size=5, col_wrap=3)
    g.map(plt.plot, "num iterations", "freq").add_legend()
    return g

def plot_results(results, ext=".png"):
    if not os.path.exists(plot_out_dir):
        os.makedirs(plot_out_dir)
    replications = num_replications(results)
    probes = sorted(set((pname, ptype)
                        for ((_, _, pname), (ptype, _)) in results.iteritems()))
    for probe, ptype in probes:
        if not ptype == 'num': continue
        grid = plot_results_numerical(results, probe)
        grid.fig.suptitle(probe + ", %d replications" % replications)
        # XXX Actually shell quote the probe name
        figname = string.replace(probe, " ", "-").replace("/", "") + ext
        savepath = os.path.join(plot_out_dir, figname)
        grid.savefig(savepath)
        plt.close(grid.fig)
        log("Probe '%s' results saved to %s" % (probe, savepath))
    grid = plot_results_boolean(results)
    grid.fig.suptitle("Boolean probes, %d replications" % replications)
    figname = "boolean-probes" + ext
    savepath = os.path.join(plot_out_dir, figname)
    grid.savefig(savepath)
    plt.close(grid.fig)
    log("Boolean probe results saved to %s" % (savepath,))
