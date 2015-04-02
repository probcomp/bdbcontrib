
import json
import pandas as pd
import bayeslite
import bayeslite.crosscat
import bayeslite.guess
import bayeslite.read_pandas
import bayeslite.bql as bql
import general_utils as gu
from bayeslite.sqlite3_util import sqlite3_quote_name as sqlquote
from bdbcontrib import draw_cc_state
from crosscat.LocalEngine import LocalEngine
from crosscat.MultiprocessingEngine import MultiprocessingEngine


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
                self._df = gu.cursor_to_df(self._cursor)
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
            self.engine = bayeslite.crosscat.CrosscatMetamodel(LocalEngine())
        else:
            self.engine = bayeslite.crosscat.CrosscatMetamodel(MultiprocessingEngine())

        if bdb_filename is None:
            print "WARNING: bdb_filename is None, all analyses will be conducted in memory"

        self.bdb = bayeslite.BayesDB(bdb_filename)
        bayeslite.bayesdb_register_metamodel(self.bdb,  self.engine)

    def __call__(self, bql_query_str):
        """ Wrapper for query """
        return self.query(bql_query_str)

    @classmethod
    def from_csv(cls, bdb_filename, btable_name, csv_filename, codebook_filename=None,
                 generator_name=None, no_mp=False, create=True, header=True,
                 columns_types=None):
        """
        Initilize table using a csv file.
        """
        if generator_name is None:
            generator_name = btable_name + '_cc'

        self = cls(bdb_filename, no_mp=no_mp)
        self.add_table_from_csv(btable_name, csv_filename, header=header, create=create)
        self.create_generator(btable_name, generator_name, columns_types)
        if codebook_filename is not None:
            self.add_codebook_to_table(btable_name, codebook_filename)

        return self

    @classmethod
    def from_pandas(cls, bdb_filename, btable_name, pandas_df, codebook_filename=None,
                    generator_name=None, no_mp=False, create=True, header=True,
                    column_types=None):
        """
        Initialize table using a pandas df.
        """
        if generator_name is None:
            generator_name = btable_name + '_cc'

        self = cls(bdb_filename, no_mp=no_mp)
        self.add_table_from_pandas(btable_name, pandas_df, create=create)
        self.create_generator(btable_name, generator_name, column_types)
        if codebook_filename is not None:
            self.add_codebook_to_table(btable_name, codebook_filename)
        return self

    def add_table_from_csv(self, btable_name, csv_filename, header=True, create=True):
        bayeslite.bayesdb_read_csv_file(self.bdb, btable_name, csv_filename, header=header,
                                        create=create)

    def add_table_from_pandas(self, btable_name, pandas_df, create=True):
        bayeslite.read_pandas.bayesdb_read_pandas_df(self.bdb, btable_name, pandas_df,
                                                     create=create)

    def create_generator(self, btable_name, generator_name, column_types=None):
        if column_types is not None:
            qg = sqlquote(generator_name)
            qt = sqlquote(btable_name)
            bql_query = "CREATE GENERATOR %s FOR %s USING crosscat" % (qg, qt)
            bql_query += '(' + ", ".join([sqlquote(cn) + ' ' + sqlquote(ct)
                                         for cn, ct in column_types]) + ')'
            self.bdb.execute(bql_query)
        else:
            bayeslite.guess.bayesdb_guess_generator(self.bdb, generator_name, btable_name,
                                                    'crosscat')

    def add_codebook_to_table(self, btable_name, codebook_filename):
        bayeslite.bayesdb_load_codebook_csv_file(self.bdb, btable_name, codebook_filename)

    def query(self, bql_query):
        """ Do query; return BQLQueryResult """
        return do_query(self.bdb, bql_query)

    def drawstate(self, btable_name, generator_name, modelno, **kwargs):
        """ Render a visualization of the crosscat state of a given model """
        with self.bdb.savepoint():
            return draw_cc_state.draw_state(self.bdb, btable_name, generator_name, modelno, **kwargs)