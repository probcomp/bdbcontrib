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

import argparse
import cPickle as pickle # json doesn't like tuple dict keys
import logging
import math
import os

from bdbcontrib.experiments.probe import log, probe_fileset

######################################################################
## Probes                                                          ##
######################################################################

def country_purpose_probes(bdb):
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

def unlikely_periods_probes(bdb):
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

def orbit_type_imputation_probes(bdb):
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
    def probe_gen():
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
    return list(probe_gen())

######################################################################
## Driver                                                           ##
######################################################################

def doit(files, outfile, model_schedule, n_replications):
    out_dir = os.path.dirname(outfile)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    results = probe_fileset(
        files, "satellites_cc",
        [country_purpose_probes,
         unlikely_periods_probes,
         orbit_type_imputation_probes],
        model_schedule = model_schedule,
        n_replications = n_replications)

    with open(outfile, "w") as f:
        pickle.dump(results, f)
    log("Saved probe results to %s" % outfile)

parser = argparse.ArgumentParser(
    description='Probe a collection of Satellites .bdb files')
parser.add_argument('-o', '--outfile',
                    help="Save results to the given file")
parser.add_argument('-m', '--n_models', nargs="+", type=int,
                    help="Count of models to probe")
parser.add_argument('-n', '--n_replications', type=int,
                    help="Number of replications to probe")
parser.add_argument('files', nargs="+", help=".bdb files to probe")

def main():
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    doit(args.files, args.outfile, args.n_models, args.n_replications)

if __name__ == '__main__':
    main()
