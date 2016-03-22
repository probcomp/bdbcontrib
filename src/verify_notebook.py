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

import re
import json
import subprocess
import sys

def get_out_and_err(notebook_path=None, outfile=None, errfile=None):
  output = None
  error = None
  if notebook_path is not None:
    cmd = 'runipy --matplotlib --stdout "%s.ipynb"' % (notebook_path,)
    p = subprocess.Popen(cmd, shell=True, stdin=None,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         close_fds=True)
    output = p.stdout.read()
    error = p.stderr.read()
  elif outfile is not None and errfile is not None:
    with open(outfile, 'r') as out:
      output = out.read()
    with open(errfile, 'r') as err:
      error = err.read()
  elif len(sys.argv) == 3:
    return get_out_and_err(sys.argv[1], sys.argv[2])
  else:
    raise ValueError("Specify a notebook path or out and err files to verify.")
  return (output, error)


def check_results(results, warnings_are_errors=False, content_tester=None):
  (output, error) = results
  try:
    notebook = json.loads(output)
  except:
    print error
    raise
  cells = notebook['worksheets'][0]['cells']
  for cell in cells:
    if cell['cell_type'] in ('markdown', 'heading'):
      pass
    elif cell['cell_type'] == 'code':
      for output in cell['outputs']:
        if output['output_type'] == 'pyerr':
          raise ValueError(str(output))
        elif output['output_type'] == 'stream':
          for msg in output['text']:
            if re.search(r'/(.*):\d+:.*(warn(ings?)?)\W', msg, re.I):
              if warnings_are_errors:
                raise ValueError(msg)
              else:
                print "WARNING: ", msg
        elif output['output_type'] == 'pyout':
          if content_tester:
            content_tester(cell)  # Content-tester should raise on error.
          else:
            pass
        elif output['output_type'] == 'display_data':
          pass  # Assume they're good.
        else:
          raise ValueError(str(output))
    else:
      raise ValueError(str(output))
  if 'exception' in error or 'nonzero exit status' in error:
    raise ValueError(error)

def run_and_verify_notebook(notebook_path, **kwargs):
  '''runipy it and verify it.'''
  check_results(get_out_and_err(notebook_path), **kwargs)
