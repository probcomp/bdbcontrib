from bayeslite.shell.hook import bayesdb_shell_cmd
from bayeslite.shell.pretty import pp_list
from bayeslite.sqlite3_util import sqlite3_quote_name as quote
import bdbcontrib.general_utils as utils

import os
import markdown2
import shutil

ROOTDIR = os.path.dirname(os.path.abspath(__file__))
READTOHTML_CSS = os.path.join(ROOTDIR, 'readtohtml.css')


@bayesdb_shell_cmd('readtohtml')
def render_bql_as_html(self, argin):
    '''Reads a bql file and outputs to html and markdown.
    <bql_file> <output_directory>

    Example:
    bayeslite> .readtohtml myscript.bql analyses/myanalsis
    '''
    args = argin.split()
    bql_file = args[0]
    output_dir = args[1]

    head = '''
    <html>
        <head>
            <link rel="stylesheet" type="text/css" href="style.css">
        </head>
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(bql_file) as f:
        mdstr = utils.mdread(f, output_dir, self)

    html = head + markdown2.markdown(mdstr) + '</html>'

    htmlfilename = os.path.splitext(os.path.basename(bql_file))[0]
    htmlfilename = os.path.join(output_dir, htmlfilename + '.html')
    with open(htmlfilename, 'w') as f:
        f.write(html)
    shutil.copyfile(READTOHTML_CSS, os.path.join(output_dir, 'style.css'))

    # Because you want to know when it's done:
    self.stdout.write(utils.unicorn())
    self.stdout.write("Output saved to %s\n" % (output_dir,))


@bayesdb_shell_cmd('nullify')
def nullify(self, argin):
    '''replaces a user specified missing value with NULL
    <table> <value>

    Example:
    bayeslite> .nullify mytable NaN
    bayeslite> .nullify mytable ''
    '''
    args = argin.split()
    table = args[0]
    value = args[1]
    utils.nullify(self._bdb, table, value)

@bayesdb_shell_cmd('cardinality')
def cardinality(self, argin):
    '''display the cardinality of columns in a table
    <table> [<column> <column> ...]

    Example:
    bayeslite> .cardinality mytable
    bayeslite> .cardinality mytable col1 col2 col3
    '''
    args = argin.split()
    table = args.pop(0)
    if len(args) > 0:
        cols = args
    else:
        sql = 'PRAGMA table_info(%s)' % (quote(table),)
        res = self._bdb.sql_execute(sql)
        cols = [r[1] for r in res.fetchall()]
    counts = []
    for col in cols:
        sql = '''
            SELECT COUNT (DISTINCT %s) FROM %s
        ''' % (quote(col), quote(table))
        res = self._bdb.sql_execute(sql)
        counts.append((col, res.next()[0]))

    pp_list(self.stdout, counts, ['column', 'cardinality'])
