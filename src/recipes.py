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

import bayeslite
import bayeslite.core
import bayeslite.guess
from bayeslite.loggers import BqlLogger, logged_query
from bayeslite.exception import BayesLiteException as BLE
import bayeslite.metamodels.crosscat
import bdbcontrib
import bdbcontrib.plot_utils
from py_utils import helpsub

from collections import defaultdict
import crosscat
import crosscat.LocalEngine
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import sys
import traceback


class BqlRecipes(object):
  def __init__(self, name, csv_path=None, bdb_path=None, df=None, logger=None,
               session_capture_name=None):
    """A set of shortcuts for common ways to use BayesDB.

    name : str  REQUIRED.
        The name of dataset, should use letters and underscores only.
        This will also be used as a table name, and %t in queries will be
        replaced by this name.
    csv_path : str
        The path to a comma-separated values file. If specified, will be used
        to populate the bdb. It must exist and be both readable and non-empty.
    df : pandas.DataFrame
        If specified, these data will be used to populate the bdb, superseding
        any csv_path. It must not be empty.
    bdb_path : str
        If specified, store data and analysis at this location. If no other
        data source (csv or df) is specified, then it must already have been
        populated. If not specified, we will use a volatile in-memory bdb.
    logger : object
        Something on which we can call .info or .warn
        By default a bayeslite.loggers.BqlLogger, but could be QuietLogger
        (only results), SilentLogger (nothing), IpyLogger, CaptureLogger,
        or LoggingLogger to use those modules, or anything else that
        implements the BqlLogger interface.
    session_capture_name : String
        Signing up with your name and email and sending your session details
        to the MIT Probabilistic Computing Group helps build a community of
        support and helps improve your user experience. You can save your choice
        in a file called 'bayesdb-session-capture-opt.txt' in the directory
        where you run the software, or any parent directory. This option
        overrides any setting in such a file. Any string is interpreted as
        opting in to sending session details. False is interpreted as opting
        out. You must choose. If you choose to use an organization name or
        email, then please send a note to bayesdb@mit.edu to help us connect
        your sessions to you.

        If you encounter a bug, or something surprising, please include your
        session capture name in your report.

        If you opt out, you still allow us to count how often users opt out.

        DO NOT USE THIS SOFTWARE FOR HIPAA-COVERED, PERSONALLY IDENTIFIABLE,
        OR SIMILARLY SENSITIVE DATA! Opting out does not guarantee security.
    """
    assert re.match(r'\w+', name)
    assert df is not None or csv_path or bdb_path
    self.name = name
    self.generator_name = name + '_cc'
    self.csv_path = csv_path
    self.df = df
    self.bdb_path = bdb_path
    if logger is None:
      if 'IPython' in sys.modules:
        from bdbcontrib.loggers import IPYTHON_LOGGER as ipy
        self.logger = ipy
      else:
        self.logger = BqlLogger()
    else:
      self.logger = logger
    self.bdb = None
    self.status = None
    self.session_capture_name = None
    self.generators = None
    self.initialize_session_capture(session_capture_name)
    self.initialize()

  def initialize_session_capture(self, name):
    if self.session_capture_name is not None:
      return
    if name is not None:
      if name == False:
        with logged_query('count-beacon', None, name='single-opt-out'):
          pass
      self.session_capture_name = name
      return
    # Search for a session-capture name or opt-out saved as a file:
    filename = "bayesdb-session-capture-opt.txt"
    searchdir = os.getcwd()
    while searchdir != os.path.dirname(searchdir):  # While not at root.
      try:
        with open(os.path.join(searchdir, filename), 'r') as optinfile:
          self.session_capture_name = optinfile.read()
          if self.session_capture_name == 'False':
            with logged_query('count-beacon', None, name='saved-opt-out'):
              self.session_capture_name = False
          break
      except IOError:
        pass
      searchdir = os.path.dirname(searchdir)
    # No init option specified, no choice file found. Force the choice.
    if self.session_capture_name is None:
      raise BLE(ValueError(
        "Please set session_capture_name option to quickstart\n"
        "  to either opt-in or opt-out of sending details of your usage of\n"
        "  this software to the MIT Probabilistic Computing Group.\n\n"
        "If you see this in one of our example notebooks,\n"
        "  return to the starting page, the Index.ipynb, to\n"
        "  make that choice."))  # TODO: Index.ipynb is a promise.

  def initialize(self):
    if self.bdb:
      return
    self.bdb = bayeslite.bayesdb_open(self.bdb_path)
    if not bayeslite.core.bayesdb_has_table(self.bdb, self.name):
      if self.df is not None:
        bayeslite.read_pandas.bayesdb_read_pandas_df(
          self.bdb, self.name, self.df, create=True, ifnotexists=True)
      elif self.csv_path:
        bayeslite.bayesdb_read_csv_file(
          self.bdb, self.name, self.csv_path,
          header=True, create=True, ifnotexists=True)
      else:
        raise BLE(ValueError("No data sources specified, and an empty bdb."))
    self.generators = self.query('''SELECT * FROM bayesdb_generator''')
    if len(self.generators) == 0:
      size = self.query('''SELECT COUNT(*) FROM %t''').ix(0, 0)
      assert 0 < size
      self.query('''
        CREATE GENERATOR %g IF NOT EXISTS FOR %t USING crosscat( GUESS(*) )''')

  def check_representation(self):
    assert self.bdb, "Did you initialize?"
    assert self.session_capture_name is not None

  def reset(self):
    self.check_representation()
    self.query('drop generator if exists %s' % self.generator_name)
    self.query('drop table if exists %s' % self.name)
    self.bdb = None
    self.initialize()

  @helpsub('bdbcontrib_describe', bdbcontrib.describe_generator_columns.__doc__)
  def quick_describe_columns(self):
    '''Wraps bdbcontrib.describe_generator_columns with bdb and generator name.

    bdbcontrib_describe'''
    self.check_representation()
    return bdbcontrib.describe_generator_columns(self.bdb, self.generator_name)

  help_for_query = (
      """Query the database. Use %t for the data table and %g for the generator.

      %t and %g work only with word boundaries. E.g., 'LIKE "%table%"' is fine.

      Returns a pandas.DataFrame with the results, rather than the cursor that
      the underlying bdb would return, so LIMIT your queries if you need to.
      """)

  @helpsub(r'help_for_query', help_for_query)
  def query(self, query_string, *bindings):
    '''Basic querying without session capture or reporting.

    help_for_query'''
    self.check_representation()
    query_string = re.sub(r'(^|(?<=\s))%t\b',
                          bayeslite.bql_quote_name(self.name),
                          re.sub(r'(^|(?<=\s))%g\b',
                                 bayeslite.bql_quote_name(self.generator_name),
                                 query_string))
    self.logger.info("BQL [%s] [%r]", query_string, bindings)
    with self.bdb.savepoint():
      try:
        res = self.bdb.execute(query_string, bindings)
        assert res is not None and res.description is not None
        self.logger.debug("BQL [%s] [%r] has returned a cursor." %
                          (query_string, bindings))
        df = bdbcontrib.cursor_to_df(res)
        self.logger.debug("BQL [%s] [%r] has created a dataframe." %
                            (query_string, bindings))
        return df
      except:
        self.logger.exception("")

  @helpsub(r'help_for_query', help_for_query)
  def q(self, query_string, *bindings):
    '''help_for_query'''
    with logged_query(query_string, bindings, name=self.session_capture_name):
      return self.query(query_string, *bindings)

  @helpsub('bdbcontrib_nullify_doc', bdbcontrib.nullify.__doc__)
  def nullify(self, value):
    """Wraps bdbcontrib.nullify by passing bdb and name.

    bdbcontrib_nullify_doc"""
    bdbcontrib.nullify(self.bdb, self.name, value)

  def analyze(self, models=100, minutes=0, iterations=0, checkpoint=0):
    '''Run analysis.

    models : integer
        The number of models bounds the accuracy of predictive probabilities.
        With ten models, then you get one decimal digit of interpretability,
        with a hundred models, you get two, and so on.
    minutes : integer
        How long you want to let it run.
    iterations : integer
        How many iterations to let it run.

    Returns:
        A report indicating how many models have seen how many iterations,
        and other info about model stability.
    '''
    self.check_representation()
    with logged_query(query_string='recipes.analyze',
                      name=self.session_capture_name):
      if models > 0:
        self.query('INITIALIZE %d MODELS IF NOT EXISTS FOR %s' %
              (models, self.generator_name))
        assert minutes == 0 or iterations == 0
      else:
        models = self.analysis_status().sum()
      if minutes > 0:
        if checkpoint == 0:
          checkpoint = max(1, int(minutes * models / 200))
        analyzer = ('ANALYZE %s FOR %d MINUTES CHECKPOINT %d ITERATION WAIT' %
                    (self.generator_name, minutes, checkpoint))
        with logged_query(query_string=analyzer,
                          name=self.session_capture_name,
                          bindings=self.query('SELECT * FROM %t')):
          self.query(analyzer)
      elif iterations > 0:
        if checkpoint == 0:
          checkpoint = max(1, int(iterations / 20))
        self.query(
            '''ANALYZE %s FOR %d ITERATIONS CHECKPOINT %d ITERATION WAIT''' % (
              self.generator_name, iterations, checkpoint))
      else:
        raise BLE(NotImplementedError('No default analysis strategy yet.'
                                      ' Please specify minutes or iterations.'))
    # itrs = self.per_model_analysis_status()
    # models_with_fewest_iterations =
    #    itrs[itrs['iterations'] == itrs.min('index').head(0)[0]].index.tolist()
    # TODO(gremio): run each model with as many iterations as it needs to get
    # up to where it needs to get to, if that's larger?
    # Nope. Vikash said there's no reason to think that's a good idea. Perhaps
    # even better to have some young models mixed in with the old ones.
    # I still think we should make some recommendation that scales for what
    # "the right thing" is, where that's something that at least isn't known to
    # suck.

    return self.analysis_status()

  # TODO: remove import plot_utils from the __init__.py file -- make it empty.
  # If you want to use it, you should import bdbcontrib.foo.

  def per_model_analysis_status(self):
    """Return the number of iterations for each model."""
    # XXX Move this to bdbcontrib/src/bql_utils.py ?
    try:
      return self.query('''SELECT iterations FROM bayesdb_generator_model
                           WHERE generator_id = (
                            SELECT id FROM bayesdb_generator WHERE name = ?)''',
                        self.generator_name)
    except ValueError:
      # Because, e.g. there is no generator yet, for an empty db.
      return None

  def analysis_status(self):
    """Return the count of models for each number of iterations run."""
    itrs = self.per_model_analysis_status()
    if itrs is None or len(itrs) == 0:
      emt = pd.DataFrame(columns=['count of models'])
      emt.index.name = 'iterations'
      return emt
    vcs = pd.DataFrame(itrs['iterations'].value_counts())
    vcs.index.name = 'iterations'
    vcs.columns = ['count of models']
    self.status = vcs
    return vcs

  @helpsub('bdbcontrib_pairplot', bdbcontrib.plot_utils.pairplot.__doc__)
  def pairplot(self, cols, plotfile=None, colorby=None, **kwargs):
    """Wrap bdbcontrib.plot_utils.pairplot to show the given columns.

    Specifies bdb, query with the given columns, and generator_name:
    bdbcontrib_pairplot
    """
    if len(cols) < 1:
        raise BLE(ValueError('Pairplot at least one variable.'))
    qcols = cols if colorby is None else set(cols + [colorby])
    query_columns = '''"%s"''' % '''", "'''.join(qcols)
    with logged_query(query_string='pairplot cols=?', bindings=(query_columns,),
                      name=self.session_capture_name):
      self.logger.plot(plotfile,
                       bdbcontrib.pairplot(self.bdb, '''SELECT %s FROM %s''' %
                                             (query_columns, self.name),
                                           generator_name=self.generator_name,
                                           colorby=colorby,
                                           **kwargs))

  def heatmap(self, deps, selectors=None, plotfile=None, **kwargs):
    '''Show heatmaps for the given dependencies

    Parameters
    ----------
    deps : pandas.DataFrame(columns=['generator_id', 'name0', 'name1', 'value'])
        The result of a .q('ESTIMATE ... PAIRWISE ...')
        E.g., DEPENDENCE PROBABILITY, MUTUAL INFORMATION, COVARIANCE, etc.

    selectors : {str: lambda name --> bool}
        Rather than plot the full NxN matrix all together, make separate plots
        for each combination of these selectors, plotting them in sequence.
        If selectors are specified, the actual selector functions are values of
        a dict, and the keys are their names, for purposes of plot legends and
        filenames.
        E.g.,
          {'A-E': lambda x: bool(re.search(r'^[a-eA-E]', x[0])),
           'F-O': lambda x: bool(re.search(r'^[f-oF-O]', x[0])),
           'P-Z': lambda x: bool(re.search(r'^[p-zP-Z]', x[0]))}

    plotfile : str
        If a plotfile is specified, savefig to that file. If selectors are also
        specified, savefig to name1.name2.plotfile.

    **kwargs : dict
        Passed to zmatrix: vmin, vmax, row_ordering, col_ordering
    '''
    with logged_query(query_string='heatmap(deps, selectors)',
                      bindings=(str(deps), repr(selectors)),
                      name=self.session_capture_name):
      hmap = plt.figure()
      if selectors is None:
        cmap = bdbcontrib.plot_utils.heatmap(self.bdb, df=deps, **kwargs)
        self.logger.plot(plotfile, cmap)
      else:
        selfns = [selectors[k] for k in sorted(selectors.keys())]
        reverse = dict([(v, k) for (k, v) in selectors.items()])
        for (cmap, sel1, sel2) in bdbcontrib.plot_utils.selected_heatmaps(
             self.bdb, df=deps, selectors=selfns, **kwargs):
          self.logger.plot("%s.%s.%s" % (
              reverse[sel1], reverse[sel2], plotfile),
                           cmap)
      return hmap

  def quick_explore_cols(self, cols, nsimilar=20, plotfile='explore_cols'):
    """Show dependence probabilities and neighborhoods based on those.

    cols: list of strings
        At least two column names to look at dependence probabilities of,
        and to explore neighborhoods of.
    nsimilar: positive integer
        The size of the neighborhood to explore.
    plotfile: string pathname
        Where to save plots, if not displaying them on console.
    """
    if len(cols) < 2:
      raise BLE(ValueError('Need to explore at least two columns.'))
    with logged_query(query_string='quick_explore_cols', bindings=(cols,),
                      name=self.session_capture_name):
      self.pairplot(cols)
      query_columns = '''"%s"''' % '''", "'''.join(cols)
      deps = self.query('''ESTIMATE DEPENDENCE PROBABILITY
                           FROM PAIRWISE COLUMNS OF %s
                           FOR %s;''' % (self.generator_name, query_columns))
      deps.columns = ['genid', 'name0', 'name1', 'value']
      self.heatmap(deps, plotfile=plotfile)
      deps.columns = ['genid', 'name0', 'name1', 'value']
      triangle = deps[deps['name0'] < deps['name1']]
      triangle = triangle.sort_values(ascending=False, by=['value'])
      self.logger.result("Pairwise dependence probability for: %s\n%s\n\n",
                         query_columns, triangle)

      for col in cols:
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
        self.heatmap(deps, plotfile=(plotfile + "-" + col))
        self.logger.result("Pairwise dependence probability of %s with its " +
                           "strongest dependents:\n%s\n\n", col, neighborhood)

  def column_type(self, col):
    """The statistical type of the given column in the current model."""
    descriptions = self.quick_describe_columns()
    return descriptions[descriptions['name'] == col]['stattype'].iloc[0]

  def sql_tracing(self, turn_on=True):
    """Trace underlying SQL, for debugging."""
    # Always turn off:
    self.bdb.sql_untrace(self.bdb.sql_tracer)
    if turn_on:
      printer = lambda query, bindings: self.logger.info(
          "query: [%s] bindings: [%s]\n\n", query, bindings)
      self.bdb.sql_trace(printer)

  def quick_similar_rows(self, identify_row_by, nsimilar=10):
    """Explore rows similar to the identified one.

    identify_row_by : dict
        Dictionary of column names to their values. These will be turned into
        a WHERE clause in BQL, and must identify one unique row.
    nsimilar : positive integer
        The number of similar rows to retrieve.
    """
    with logged_query(query_string='quick_similar_rows(id_by, n)',
                      bindings=(identify_row_by, nsimilar),
                      name=self.session_capture_name):
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

@helpsub('__init__', BqlRecipes.__init__.__doc__)
def quickstart(*args, **kwargs):
  '''__init__'''
  return BqlRecipes(*args, **kwargs)
