
import pandas as pd
import bayeslite
import bayeslite.crosscat
import bayeslite.bql as bql
from bdbcontrib import draw_cc_state
from crosscat.LocalEngine import LocalEngine
from crosscat.MultiprocessingEngine import MultiprocessingEngine


def cursor_to_df(cursor):
    """ Converts SQLite3 cursor to a pandas DataFrame """
    df = pd.DataFrame.from_records(cursor.fetchall(), coerce_float=True)
    df.columns = [desc[0] for desc in cursor.description]
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

    return df


# XXX: This allows for the following syntax:
# df = do_query(bdb, 'SELECT foo FROM baz..').as_df()
# which I like better than
# df = cursor_to_df(do_query(bdb, 'SELECT foo FROM baz...'))
# The user can also do
# df = do_query(bdb, 'SELECT foo FROM baz..').df
class BQLQueryResult(object):
    """
    Does the BQL query and stores the result.

    Attributes:
    - bql_query (str): The BQL query supplied
    """
    def __init__(self, bdb, bql_query):
        self.bql_query = bql_query
        self._df = None
        self._bdb = bdb

        phrases = bayeslite.parse.parse_bql_string(bql_query)
        phrase = phrases.next()
        done = None
        try:
            phrases.next()
            done = False
        except StopIteration:
            done = True
        if done is not True:
            raise ValueError('>1 phrase: %s' % (bql_query,))
        self._cursor = bql.execute_phrase(bdb, phrase)

    def as_df(self):
        """ Returns the query result as a dataframe."""
        if self._df is None:
            with self._bdb.savepoint():
                self._df = cursor_to_df(self._cursor)
        return self._df

    def as_cursor(self):
        """ Returns the query result as a SQLite cursor"""
        return self._cursor

    @property
    def df(self):
        return self.as_df()

    @property
    def cursor(self):
        return self.as_cursor()


def do_query(bdb, bql_query):
    print '--> %s' % (bql_query.strip(),)
    return BQLQueryResult(bdb, bql_query)


# ``````````````````````````````````````````````````````````````````````````````````````````````````
class BayesDBClient(object):
    """
    """
    def __init__(self, bdb_filename, no_mp=False):
        if no_mp:
            self.engine = bayeslite.crosscat.CrosscatEngine(LocalEngine())
        else:
            self.engine = bayeslite.crosscat.CrosscatEngine(MultiprocessingEngine())

        if bdb_filename is None:
            print "WARNING: bdb_filename is None, all analyses will be conducted in memory"

        self.bdb = bayeslite.BayesDB(bdb_filename)
        bayeslite.bayesdb_register_metamodel(self.bdb, 'crosscat', self.engine)
        bayeslite.bayesdb_set_default_metamodel(self.bdb, 'crosscat')

    def __call__(self, bql_query_str):
        """ Wrapper for query """
        return self.query(bql_query_str)

    @classmethod
    def from_csv(cls, bdb_filename, btable_name, csv_filename, codebook_filename=None,
                 no_mp=False, column_types=None, ifnotexists=False):
        """
        Initilize table using a csv file.
        """
        self = cls(bdb_filename, no_mp=no_mp)
        self.add_table_from_csv(btable_name, csv_filename, column_types=column_types,
                                ifnotexists=False)
        if codebook_filename is not None:
            self.add_codebook_to_table(btable_name, codebook_filename)
        return self

    @classmethod
    def from_pandas(cls, bdb_filename, btable_name, pandas_df, codebook_filename=None, no_mp=False,
                    column_types=None, ifnotexists=False):
        """
        Initialize table using a pandas df.
        """
        self = cls(bdb_filename, no_mp=no_mp)
        self.add_table_from_pandas(btable_name, pandas_df, column_types=column_types,
                                   ifnotexists=False)
        if codebook_filename is not None:
            self.add_codebook_to_table(btable_name, codebook_filename)
        return self

    def add_table_from_csv(self, btable_name, csv_filename, codebook_filename=None,
                           column_types=None, ifnotexists=False):
        bayeslite.bayesdb_import_csv_file(self.bdb, btable_name, csv_filename,
                                          column_types=column_types, ifnotexists=ifnotexists)

    def add_table_from_pandas(self, btable_name, pandas_df, codebook_filename=None,
                              column_types=None, ifnotexists=False):
        bayeslite.bayesdb_import_pandas_df(self.bdb, btable_name, pandas_df,
                                           column_types=column_types)

    def add_codebook_to_table(self, btable_name, codebook_filename):
        bayeslite.bayesdb_import_codebook_csv_file(self.bdb, btable_name, codebook_filename)

    def query(self, bql_query):
        """ Do query; return BQLQueryResult """
        return do_query(self.bdb, bql_query)

    def plot_state(self, btable_name, modelno, **kwargs):
        """ Render a visualization of the crosscat state of a given model """
        return draw_cc_state.draw_state(self.bdb, btable_name, modelno, **kwargs)
