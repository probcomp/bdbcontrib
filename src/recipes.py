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

from collections import defaultdict
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import sys
import traceback

import bayeslite
import bayeslite.core
from bayeslite.loggers import BqlLogger, logged_query
from bayeslite.exception import BayesLiteException as BLE
import bdbcontrib
import bdbcontrib.plot_utils # To helpsub from it.
from bdbcontrib.population import Population
from bdbcontrib.population_method import population_method
from py_utils import helpsub

BqlRecipes = Population

@population_method(population=0)
def quick_explore_vars(self, varnames, plotfile=None, nsimilar=20):
  """Show dependence probabilities and neighborhoods based on those.

  varnames: list of strings
      At least two column names to look at dependence probabilities of,
      and to explore neighborhoods of.
  nsimilar: positive integer
      The size of the neighborhood to explore.
  """
  if len(varnames) < 2:
    raise BLE(ValueError('Need to explore at least two variables.'))
  self.pairplot_vars(varnames)
  query_columns = '''"%s"''' % '''", "'''.join(varnames)
  deps = self.query('''ESTIMATE DEPENDENCE PROBABILITY
                       FROM PAIRWISE COLUMNS OF %s
                       FOR %s;''' % (self.generator_name, query_columns))
  deps.columns = ['genid', 'name0', 'name1', 'value']
  self.logger.plot(plotfile, self.heatmap(deps))
  deps.columns = ['genid', 'name0', 'name1', 'value']
  triangle = deps[deps['name0'] < deps['name1']]
  triangle = triangle.sort_values(ascending=False, by=['value'])
  self.logger.result("Pairwise dependence probability for: %s\n%s\n\n",
                     query_columns, triangle)

  for col in varnames:
    neighborhood = self.query(
    '''ESTIMATE *, DEPENDENCE PROBABILITY WITH "%s"
       AS "Probability of Dependence with %s"
       FROM COLUMNS OF %s
       ORDER BY "Probability of Dependence with %s"
       DESC LIMIT %d;'''
       % (col, col, self.generator_name, col, nsimilar))
    neighbor_columns = ('''"%s"''' %
                        '''", "'''.join(neighborhood["name"].tolist()))
    deps = self.query('''ESTIMATE DEPENDENCE PROBABILITY
        FROM PAIRWISE COLUMNS OF %s
        FOR %s;''' % (self.generator_name, neighbor_columns))
    deps.columns = ['genid', 'name0', 'name1', 'value']
    self.logger.plot(plotfile, self.heatmap(deps))
    self.logger.result("Pairwise dependence probability of %s with its " +
                       "strongest dependents:\n%s\n\n", col, neighborhood)

@population_method(population=0)
def quick_similar_rows(self, identify_row_by, nsimilar=10):
  """Explore rows similar to the identified one.

  identify_row_by : dict
      Dictionary of column names to their values. These will be turned into
      a WHERE clause in BQL, and must identify one unique row.
  nsimilar : positive integer
      The number of similar rows to retrieve.
  """
  import hashlib
  table_name = 'tmptbl_' + hashlib.md5('\x00'.join(
      [repr(identify_row_by), str(self.status)])).hexdigest()
  column_name = 'similarity_to_' + "__".join(
      re.sub(r'\W', '_', str(val)) for val in identify_row_by.values())
  query_params = []
  query_columns = []
  for k, v in identify_row_by.iteritems():
    query_columns.append('''%s = ? ''' % bayeslite.bql_quote_name(k))
    query_params.append(v)
  query_attrs = ' and '.join(query_columns)

  with self.bdb.savepoint():
    row_exists = self.query('SELECT COUNT(*) FROM %s WHERE %s;' %
                            (self.name, query_attrs))
    if row_exists.ix[0][0] != 1:
      raise BLE(NotImplementedError(
          'identify_row_by found %d rows instead of exactly 1 in %s.' %
          (row_exists.ix[0][0], self.csv_path)))
    creation_query = ('''CREATE TEMP TABLE IF NOT EXISTS %s AS ESTIMATE *,
                         SIMILARITY TO (%s) AS %s FROM %%g LIMIT %d;''' %
                      (table_name, query_attrs, column_name, nsimilar))
    self.query(creation_query, query_params)
    result = self.query('''SELECT * FROM %s ORDER BY %s DESC;''' %
                        (table_name, column_name))
  return result

quickstart = bdbcontrib.quickstart
