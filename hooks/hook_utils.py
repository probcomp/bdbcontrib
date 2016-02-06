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

import markdown2
import os
import shutil
import shlex

from bayeslite.shell.hook import bayesdb_shell_cmd
from bayeslite.shell.pretty import pp_list

import bdbcontrib
import bdbcontrib.shell_utils as utils


ROOTDIR = os.path.dirname(os.path.abspath(__file__))
READTOHTML_CSS = os.path.join(ROOTDIR, 'readtohtml.css')


@bayesdb_shell_cmd('readtohtml')
def render_bql_as_html(self, argin):
    """read BQL file and output to HTML and markdown
    <bql_file> <output_directory>

    Example:
    bayeslite> .readtohtml myscript.bql analyses/myanalsis
    """
    parser = utils.ArgumentParser(prog='.readtohtml')
    parser.add_argument('bql_file', metavar='bql-file', type=str,
        help='Name of the file containing the bql script.')
    parser.add_argument('output_dir', metavar='output-dir', type=str,
        help='Name of the output directory.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except utils.ArgparseError as e:
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
    """replace user-specified missing value with NULL
    <table> <value>

    Example:
    bayeslite> .nullify mytable NaN
    bayeslite> .nullify mytable ''
    """
    parser = utils.ArgumentParser(prog='.nullify')
    parser.add_argument('table', type=str,
        help='Name of the table.')
    parser.add_argument('value', type=str,
        help='Target string to nullify.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except utils.ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bdbcontrib.nullify(self._bdb, args.table, args.value)


@bayesdb_shell_cmd('cardinality')
def cardinality(self, argin):
    """show cardinality of columns in table
    <table> [<column> <column> ...]

    Example:
    bayeslite> .cardinality mytable
    bayeslite> .cardinality mytable col1 col2 col3
    """
    parser = utils.ArgumentParser(prog='.cardinality')
    parser.add_argument('table', type=str,
        help='Name of the table.')
    parser.add_argument('cols', type=str, nargs='*',
        help='Target columns for which to compute cardinality.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except utils.ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    counts = bdbcontrib.cardinality(self._bdb, args.table, cols=args.cols)
    pp_list(self.stdout, counts, ['column', 'cardinality'])
