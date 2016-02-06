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

"""Visualize results of probing across iteration and model count."""

import os
import string
import warnings

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from bdbcontrib.experiments.probe import log

# results :: [((probe_name, model_ct, iters), tagged aggregated value)]

def plot_results(results, outdir="figures", ext=".png"):
    """Plot the aggregate results of probing.

    `results` is a list of pairs giving probe conditions and
    aggregated probe results.

    `outdir` is the name of a directory to which to write the visualizations.
    Default: "figures".

    `ext` is the file extension for visualizations, which determines
    the image format used.  Default ".png".

    Each probe condition is expected to be a 3-tuple: probe name,
    model count, analysis iteration count.  Each result is expected to
    be a tagged aggregate (see aggregation.py).

    Each numerical probe produces one plot, named after the probe.
    The plot facets over the model count, displays the iteration count
    on the x-axis, and a violin plot of the results on the y axis.

    All boolean probes are aggregated into one plot named
    "boolean-probes", whose y axis is the frequency of a "True"
    result.  Each probe is a line giving the relationship of the
    frequency to the number of analysis iterations.

    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    replications = num_replications(results)
    probes = sorted(set((pname, ptype)
                        for ((pname, _, _), (ptype, _)) in results))
    for probe, ptype in probes:
        if not ptype == 'num': continue
        grid = plot_results_numerical(results, probe)
        grid.fig.suptitle(probe + ", %d replications" % replications)
        # XXX Actually shell quote the probe name
        figname = string.replace(probe, " ", "-").replace("/", "") + ext
        savepath = os.path.join(outdir, figname)
        grid.savefig(savepath)
        plt.close(grid.fig)
        log("Probe '%s' results saved to %s" % (probe, savepath))
    grid = plot_results_boolean(results)
    grid.fig.suptitle("Boolean probes, %d replications" % replications)
    figname = "boolean-probes" + ext
    savepath = os.path.join(outdir, figname)
    grid.savefig(savepath)
    plt.close(grid.fig)
    log("Boolean probe results saved to %s" % (savepath,))

def plot_results_numerical(results, probe):
    # results :: [((probe_name, model_ct, iters), tagged result_set)]
    data = ((iters, model_ct, value)
            for ((pname, model_ct, iters), (ptype, values)) in results
        if pname == probe and ptype == 'num'
        for value in values)
    cols = ["num iterations", "n_models", "value"]
    df = pd.DataFrame.from_records(data, columns=cols) \
                     .sort(["num iterations", "n_models"])
    g = sns.FacetGrid(df, col="n_models", size=5, col_wrap=3)
    g.map(sns.violinplot, "num iterations", "value")
    return g

def plot_results_boolean(results):
    # results :: [((probe_name, model_ct, iters), tagged result_set)]
    data = ((iters, model_ct, pname, float(t)/(t+f))
            for ((pname, model_ct, iters), (ptype, value)) in results
        if ptype == 'bool'
        for (t, f) in [value])
    cols = ["num iterations", "n_models", "probe", "freq"]
    df = pd.DataFrame.from_records(data, columns=cols) \
                     .sort(["probe", "num iterations", "n_models"])
    g = sns.FacetGrid(df, col="n_models", hue="probe", size=5, col_wrap=3)
    g.map(plt.plot, "num iterations", "freq").add_legend()
    return g

def num_replications(results):
    replication_counts = \
        set((len(l) for (_, (ptype, l)) in results
             if ptype == 'num')).union(
        set((t+f for (_, (ptype, l)) in results
             if ptype == 'bool'
             for (t, f) in [l])))
    replication_count = next(iter(replication_counts))
    if len(replication_counts) > 1:
        msg = "Non-rectangular results found; replication counts range " \
              "over %s, using %s" % (replication_counts, replication_count)
        warnings.warn(msg)
    return replication_count
