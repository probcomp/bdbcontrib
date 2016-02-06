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
import os
import string
from cStringIO import StringIO


PLOTTING_COMMANDS = ['.heatmap', '.histogram', '.show',
                     '.chainplot', '.ccstate', '.bar',
                     '.mihist']
BPROMPT = 'bayeslite> '
CPROMPT = '      ...> '


# Kludgey workaround for idiotic Python argparse module which exits
# the process on argument parsing failure.  We override the exit
# method of ArgumentParser so that it raises an exception instead of
# exiting the process, which we then catch around parser.parse_args()
# in order to report the message and return to the command loop.
class ArgparseError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message


class ArgumentParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        raise ArgparseError(status, message)

def is_dot_command(line):
    if is_blank_line(line):
        return False
    return line[0] == '.'


def is_continuation(line):
    if is_blank_line(line):
        return False
    return line[0] in string.whitespace


def is_plotting_command(line):
    if not is_dot_command(line):
        return False
    return line.split()[0] in PLOTTING_COMMANDS


def is_blank_line(line):
    return len(line) == 0 or line.isspace()


def is_comment(line):
    if len(line) < 2:
        return False
    return line[:2] == '--'


def get_line_type(line):
    if is_comment(line):
        return 'comment'
    elif is_blank_line(line):
        return 'blank'
    else:
        return 'code'


def do_and_cap(shell, cmd):
    assert len(cmd) > 0
    shell.stderr.write("Performing ")
    shell.stderr.write(cmd)
    shell.stderr.write("\n")
    shell.stderr.flush()

    backup = shell.stdout
    stream = StringIO()
    shell.stdout = stream

    shell.onecmd(cmd)

    shell.stdout = backup
    output = stream.getvalue()
    stream.close()

    return output


def clean_cmd_filename(cmd, fignum, output_dir):
    if ' -f ' not in cmd and ' --filename ' not in cmd:
        figname = 'fig_' + str(fignum) + '.png'
        filename = os.path.join(output_dir, figname)
        cmd += ' --filename ' + filename
    else:
        raise ValueError
    return cmd, figname


def exec_and_cap_cmd(cmd, fignum, shell, mdstr, output_dir):
    plotting_cmd = is_plotting_command(cmd)
    if plotting_cmd:
        cmd, figfile = clean_cmd_filename(cmd, fignum, output_dir)
        fignum += 1
    # do the comand and grab the output, and close the code markup
    output = do_and_cap(shell, cmd)
    output = '\n'.join(['    ' + s for s in output.split('\n')])
    mdstr += '\n' + output
    if plotting_cmd:
        mdstr += "\n![{}]({})\n".format(cmd, figfile)
    cmd = ''
    return cmd, mdstr, fignum


def mdread(f, output_dir, shell):
    """Reads a .bql file and converts it to markdown.

    Captures text and figure output and saves all code and image assets to
    a directory. Using `mdread` requires some special care on behalf of the
    user. See `writing a BQL script`.

    Parameters
    ----------
    f : file
        The .bql file to convert.
    output_dir : string
        The name of an output directory where all assets will be saved.
    shell : bayeslite.Shell
        The shell object from which the function was called.

    Returns
    -------
    mdstr : string
        A string of markdown

    Notes
    -----
    When writing a script for mdread, do not use ``--filename`` arguments for
    plotting commands. `mdread` will generate filenames that ensure assets are
    saved alongside the markdown.
    """
    if not isinstance(f, file):
        raise TypeError('f should be a file.')
    lines = f.read().split('\n')

    cont = False
    cmd = ''
    mdstr = ''
    last_type = None
    fignum = 0

    # XXX: The first three chracters are stripped from comments, because it is
    # assumed that a space immediately follows '--'. I should probably change
    # this in future.
    line = lines[0]
    last_type = get_line_type(line)
    if last_type == 'code':
        mdstr += "\n"
        cmd = line.strip()
        mdstr += '    ' + line
    elif last_type == 'comment':
        mdstr += line[3:].rstrip() + '\n'
    else:
        mdstr += line

    for i, line in enumerate(lines[1:]):
        linetype = get_line_type(line)
        cont = is_continuation(line)
        if cont and linetype != last_type:
            raise ValueError
        if cont:
            if linetype == 'code':
                cmd += ' ' + line.strip()
                mdstr += '\n' + '    ' + CPROMPT + line
        else:
            if last_type == 'code':
                mdstr += '\n'
                cmd, mdstr, fignum = exec_and_cap_cmd(cmd, fignum, shell,
                                                      mdstr, output_dir)
            if linetype == 'comment':
                mdstr += line[3:].rstrip() + '\n'
            elif linetype == 'code':
                cmd = line
                mdstr += '\n    ' + BPROMPT + line
                if i == len(lines)-1:
                    last_cmd, mdstr, fignum = exec_and_cap_cmd(
                        cmd, fignum, shell, mdstr, output_dir)
            else:
                mdstr += '\n'
        last_type = linetype

    if last_type == 'code' and cont:
        last_cmd, mdstr, fignum = exec_and_cap_cmd(cmd, fignum, shell, mdstr,
                                                   output_dir)

    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(mdstr)

    return mdstr


def unicorn():
    """It's not a unicorn at all!"""
    unicorn = """
                                 ,-""   `.
                               ,'  _   e )`-._
                              /  ,' `-._<.===-'
                             /  /
                            /  ;
                _          /   ;
   (`._    _.-"" ""--..__,'    |
   <_  `-""                     \\
    <`-                          :
     (__   <__.                  ;
       `-.   '-.__.      _.'    /
          \\      `-.__,-'    _,'
           `._    ,    /__,-'
              ""._\\__,'< <____
                   | |  `----.`.
                   | |        \\ `.
                   ; |___      \\-``
                   \\   --<
                    `.`.<
               hjw    `-'
    """
    return unicorn
