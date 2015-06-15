
from bayeslite.shell.hook import bayesdb_shell_cmd
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
        mdstr, err = utils.mdread(f, output_dir, self)

    html = head + markdown2.markdown(mdstr) + '</html>'

    htmlfilename = os.path.splitext(os.path.basename(bql_file))[0]
    htmlfilename = os.path.join(output_dir, htmlfilename + '.html')
    with open(htmlfilename, 'w') as f:
        f.write(html)
    shutil.copyfile(READTOHTML_CSS, os.path.join(output_dir, 'style.css'))

    if err:
        self.stdout.write('The build failed. Check output for details.\n')
    else:
        self.stdout.write(utils.unicorn())

    self.stdout.write("Output saved to %s\n" % (output_dir,))


@bayesdb_shell_cmd('nullify')
def nullify(self, argin):
    '''Replaces a user specified missing value with NULL
    <table> <value>

    Example:
    bayeslite> .nullify mytable NaN
    bayeslite> .nullify mytable ''
    '''
    args = argin.split()
    table = args[0]
    value = args[1]
    utils.nullify(self._bdb, table, value)
