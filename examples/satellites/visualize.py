#!/usr/bin/env python
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

import matplotlib
matplotlib.use('Agg')

import argparse
import cPickle as pickle # json doesn't like tuple dict keys
import logging
import re

from bdbcontrib.experiments.probe import log
from bdbcontrib.experiments.visualization import plot_results

def analysis_count_from_file_name(fname):
    return int(re.match(r'.*-[0-9]*m-([0-9]*)i.bdb', fname).group(1))

def plot_probe_results(filename, outdir):
    log("Loading probe results from %s" % filename)
    with open(filename, "r") as f:
        results = pickle.load(f)
    results = [((probe, n_models, analysis_count_from_file_name(fname)), val)
               for ((fname, n_models, probe), val) in results.iteritems()]
    plot_results(results, outdir=outdir)

parser = argparse.ArgumentParser(
    description='Process a collection of saved probe results')
parser.add_argument('-i', '--infile',
                    help="Read probe results from the given file.")
parser.add_argument('-o', '--outdir',
                    help="Write polts to the given output directory.")

def main():
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    plot_probe_results(args.infile, args.outdir)

if __name__ == '__main__':
    main()
