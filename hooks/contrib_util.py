import markdown2
import os
import shutil
import shlex

from bayeslite.shell.hook import bayesdb_shell_cmd
from bayeslite.shell.pretty import pp_list
from bayeslite.sqlite3_util import sqlite3_quote_name as quote

import bdbcontrib.general_utils as utils
from bdbcontrib.general_utils import ArgparseError, ArgumentParser


ROOTDIR = os.path.dirname(os.path.abspath(__file__))
READTOHTML_CSS = os.path.join(ROOTDIR, 'readtohtml.css')


@bayesdb_shell_cmd('readtohtml')
def render_bql_as_html(self, argin):
    """
    Read BQL file and output to HTML and markdown.
    USAGE: .readtohtml <bql_file> <output_directory>

    Example:
    bayeslite> .readtohtml myscript.bql analyses/myanalsis
    """
    parser = ArgumentParser(prog='.readtohtml')
    parser.add_argument('bql-file', type=str,
        help='Name of the file containing the bql script.')
    parser.add_argument('output-dir', type=str,
        help='Name of the output directory.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql_file = args.bql_file
    output_dir = args.output_dir

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
    """
    Replace a user-specified missing value with NULL
    USAGE: .nu<table> <value>

    Example:
    bayeslite> .nullify mytable NaN
    bayeslite> .nullify mytable ''
    """
    parser = ArgumentParser(prog='.nullify')
    parser.add_argument('table', type=str,
        help='Name of the table.')
    parser.add_argument('value', type=str,
        help='Target string to nullify.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    table = args.table
    value = args.value

    utils.nullify(self._bdb, table, value)

@bayesdb_shell_cmd('cardinality')
def cardinality(self, argin):
    """
    Display the cardinality of columns in a table
    USAGE: .cardinality <table> [<column> <column> ...]

    Example:
    bayeslite> .cardinality mytable
    bayeslite> .cardinality mytable col1 col2 col3
    """
    parser = ArgumentParser(prog='.cardinality')
    parser.add_argument('table', type=str,
        help='Name of the table.')
    parser.add_argument('cols', type=str, nargs='*',
        help='Target columns for which to compute cardinality.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    table = args.table
    # If no target columns specified, use all.
    if args.cols:
        cols = args.cols
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
