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

import markdown2
import os
import shutil

from bayeslite.shell.hook import bayesdb_shell_cmd
from bayeslite.shell.pretty import pp_list
from bayeslite.sqlite3_util import sqlite3_quote_name as quote

import bdbcontrib.general_utils as utils


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
    """
    Replace a user-specified missing value with NULL
    USAGE: .nullify <table> <value>

    Example:
    bayeslite> .nullify mytable NaN
    bayeslite> .nullify mytable ''
    """
    args = argin.split()
    table = args[0]
    value = args[1]
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
