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

from contextlib import contextmanager
import json
import os
import re
import subprocess
import sys
import traceback

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

def ident(cell_i, cell, msglimit=None):
  return "Cell[%s]: %s" % (cell_i, repr(cell)[:msglimit])

def assert_markdown_matches(rxp, **kwargs):
  def assert_markdown_matches_tester(cell_i, cell, dscr):
    assert cell['cell_type'] in ('markdown', 'heading'), \
        "Expected markdown cell: " + dscr
    assert 'source' in cell, dscr
    full_source = '\n'.join(cell['source'])
    assert re.search(rxp, full_source, **kwargs), \
        "Expected /%s/\nFound: %s\nFor cell: %s" % (rxp, full_source, dscr)
    return True
  return assert_markdown_matches_tester

def assert_markdown_not(rxp, **kwargs):
  def assert_markdown_not_matches_tester(cell_i, cell, dscr):
    assert cell['cell_type'] in ('markdown', 'heading'), \
        "Expected markdown cell: " + dscr
    assert 'source' in cell, dscr
    full_source = '\n'.join(cell['source'])
    assert not re.search(rxp, full_source, **kwargs), \
        ("Expected not to find /%s/\nFound: %s\nFor cell: %s" %
         (rxp, full_source, dscr))
    return True
  return assert_markdown_not_matches_tester

def make_warning_tester(rxp, optional=False, **kwargs):
  def assert_warns_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert 'outputs' in cell, dscr
    found = False
    for output in cell['outputs']:
      assert 'output_type' in output, dscr
      if 'stream' != output['output_type']:
        continue
      assert 'text' in output, dscr
      for msg_i, msg in enumerate(output['text']):
        if re.search(r'/(.*):\d+:.*(warn(ings?)?)\W', msg, re.I):
          if re.search(rxp, msg, **kwargs):
            # Found it!
            # Mark it so we don't die when checking for unchecked warnings:
            output['text'][msg_i] += ' __verify_notebook_warning_ok__'
            return True
    assert optional or found, \
      "No matching warning: /%s/ in cell %s" % (rxp, dscr)
    return True
  return assert_warns_tester

def assert_warns(rxp, **kwargs):
  return make_warning_tester(rxp, optional=False, **kwargs)

def allow_warns(rxp, **kwargs):
  return make_warning_tester(rxp, optional=True, **kwargs)

def assert_raises(rxp, **kwargs):
  def assert_raises_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert 'outputs' in cell, dscr
    found = False
    for output in cell['outputs']:
      assert 'output_type' in output, dscr
      if 'pyerr' != output['output_type']:
        continue
      assert 'evalue' in output, dscr
      if re.search(rxp, output['evalue'], **kwargs):
        # Found it!
        # Mark it so we don't die when checking for unchecked warnings:
        output['evalue'] += ' __verify_notebook_error_ok__'
        return True
    assert found, "No matching error: /%s/ in cell %s" % (rxp, dscr)
  return assert_raises_tester

def assert_no_pyout():
  def assert_no_pyout_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert ('outputs' not in cell or
            [] == cell['outputs']), "Expected no output from cell: " + dscr
    return True
  return assert_no_pyout_tester

def assert_stream_matches(rxp, **kwargs):
  def assert_stream_matches_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert 'outputs' in cell, "No outputs in cell: " + dscr
    found = False
    for output in cell['outputs']:
      assert 'output_type' in output, "No output type: " + dscr
      if 'stream' != output['output_type']:
        continue
      assert 'text' in output, "No text in output: " + dscr
      for msg_i, msg in enumerate(output['text']):
        if re.search(rxp, msg, **kwargs):
          return True
    assert found, "No matching stream text: /%s/ in cell %s" % (rxp, dscr)
  return assert_stream_matches_tester

def assert_stream_not(rxp, **kwargs):
  def assert_stream_not_matches_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert 'outputs' in cell, "No outputs in code cell: " + dscr
    for output in cell['outputs']:
      assert 'output_type' in output, "No output_type in cell: " + dscr
      if 'stream' != output['output_type']:
        continue
      assert 'text' in output, "No text in stream: " + dscr
      for msg_i, msg in enumerate(output['text']):
        assert not re.search(rxp, msg, **kwargs), \
          "Expected not to find /%s/ but found it in cell: %s" % (rxp, dscr)
    return True
  return assert_stream_not_matches_tester

def assert_pyout_matches(rxp, **kwargs):
  def assert_pyout_matches_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert 'outputs' in cell, "No outputs in cell: " + dscr
    found = False
    for output in cell['outputs']:
      assert 'output_type' in output, "No output type: " + dscr
      if 'pyout' != output['output_type']:
        continue
      assert 'text' in output, "No text in output: " + dscr
      for msg_i, msg in enumerate(output['text']):
        if re.search(rxp, msg, **kwargs):
          return True
    assert found, "No matching output: /%s/ in cell %s" % (rxp, dscr)
  return assert_pyout_matches_tester

def assert_pyout_not(rxp, **kwargs):
  def assert_pyout_not_matches_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert 'outputs' in cell, "No outputs in code cell: " + dscr
    for output in cell['outputs']:
      assert 'output_type' in output, "No output_type in cell: " + dscr
      if 'pyout' != output['output_type']:
        continue
      assert 'text' in output, "No text in pyout: " + dscr
      for msg_i, msg in enumerate(output['text']):
        assert not re.search(rxp, msg, **kwargs), \
          "Expected not to find /%s/ but found it in cell: %s" % (rxp, dscr)
    return True
  return assert_pyout_not_matches_tester

def assert_has_png():
  def assert_has_png_tester(cell_i, cell, dscr):
    assert 'code' == cell['cell_type'], "Expected code cell: " + dscr
    assert 'outputs' in cell, dscr
    found = False
    for output in cell['outputs']:
      assert 'output_type' in output, dscr
      if 'display_data' == output['output_type'] and 'png' in output:
        output['png'] = '__verify_notebook_png_ok__'
        return True
    assert found, "Expected png for cell: " + dscr
  return assert_has_png_tester


def check_results(results,
                  exhaustive=None,
                  required=None,
                  callback=None,
                  warnings_are_errors=True,
                  msglimit=None,
                  errstream=sys.stderr):
  """Check that the ipynb ran successfully and that they match given patterns.

  exhaustive: [ (input cell regex, [ ANDed output cell testers ] ) ]
      * In the same order as the notebook,
      * All cells must be listed,
      * A cell tester is a fn(cell_i, cell, description) --> boolean
      * If a cell tester is not callable, it is assumed to be a regex r that
        is meant to be short for assert_markdown_matches or assert_pyout_matches
        as appropriate for the cell type.
      * It may raise or return False or None to fail, must return True to pass.
      * An empty cell pattern list is okay (checks success as usual).
      * Note the helpers defined in this module:
        + assert_markdown_matches(rxp, **kwargs_to_re_search)
        + assert_markdown_not(rxp, **kwargs_to_re_search)
        + assert_warns(rxp, **kwargs_to_re_search)
        + assert_raises(rxp, **kwargs_to_re_search)
        + assert_no_pyout()
        + assert_pyout_matches(rxp, **kwargs_to_re_search)
        + assert_pyout_not(rxp, **kwargs_to_re_search)
        + assert_has_png()
      * If you use assert_warns or assert_raises, then the messages found by
        those will not cause the overall correctness check to fail.

  required: [ (input cell regex, [ ANDed output cell testers ] ) ]
      * As before, all given testers must pass for a cell that matches the
        input cell regex, but
      * Not all notebook cells need be listed, and they need not be in order.
      * Even so, each input regex can still only match once, and
      * All input regexes must still match.

  callback: fn(cell_i, cell, description) --> boolean
      * Called on each cell.
      * Like testers above, may raise, or return False or None to fail,
        and must return True to pass.

  warnings_are_errors: boolean (default=True)
      Fail if a cell issues a warning (except as marked by assert_warns).

  msglimit: int
      Show this many characters of a notebook cell in an error message. Useful
      to make errors on very long cells more readable.
      Example: msglimit=(None if pytest.config.options.verbose else 1000)

  errstream: StringIO (default sys.stderr) or list or None.
      Where to write individual AssertionErrors (other errors propagate).
      Pass a list to collect them and let you do what you please.
      Pass None to raise immediately rather than collecting all failures.
  """
  exh = exhaustive[:] if exhaustive is not None else []  # Mutable copy.
  req = required[:] if required is not None else []  # Mutable copy.
  (output, error) = results
  try:
    notebook = json.loads(output)
  except:
    print error
    raise
  cells = notebook['worksheets'][0]['cells']

  errors = []
  @contextmanager
  def each_error():
    try:
      yield
    except AssertionError as e:
      if errstream is None:
        raise
      elif isinstance(errstream, list):
        # Pylint can't tell that isinstance above protected errstream from
        # being a file.
        errstream.append(e)  # pylint: disable=no-member
      else:
        errors.append(e)
        traceback.print_exc(limit=1, file=errstream)
        print >>errstream, "\n"

  for cell_i, cell in enumerate(cells):
    with each_error():
      dscr = "Cell[%s]: %s" % (cell_i, repr(cell)[:msglimit])
      assert cell['cell_type'] in ('markdown', 'heading', 'code'), dscr
      if callback:
        assert callback(cell_i, cell, dscr)

      # Find the tests to run:
      input = '\n'.join(cell['input'] if cell['cell_type'] == 'code'
                        else cell['source'])
      cell_tests = []
      if exhaustive is not None:
        assert cell_i < len(exh), "Unexpected cell at end: " + dscr
        spec = exh[cell_i]
        assert 2 == len(spec), \
            ("Expected an input pattern and a list of tests,\n"
             "Found %s\nFor cell %s" % (spec, dscr))
        pttn, cell_tests = spec
        assert re.search(pttn, input), \
          "Expected /%s/\nFound: %s\nFor cell: %s" % (pttn, input, dscr)

      for i, spec in enumerate(req):
        assert 2 == len(spec), \
            ("Expected an input pattern and a list of tests,\n"
             "Found %s\nFor cell %s" % (spec, dscr))
        pttn, tests = spec
        if re.search(pttn, input):
          cell_tests += tests
          req.pop(i)

      for test in cell_tests:
        if not hasattr(test, '__call__'):
          if cell['cell_type'] == 'code':
            test = assert_pyout_matches(test)
          else:
            test = assert_markdown_matches(test)
        ans = test(cell_i, cell, dscr)
        assert ans, "Tester %r returned %r on cell: %s" % (test, ans, dscr)

      # Standard tests:
      if cell['cell_type'] == 'code':
        for output in cell['outputs']:
          assert output['output_type'] in (
            'pyerr', 'stream', 'pyout', 'display_data'), dscr
          if (output['output_type'] == 'pyerr'):
            assert 'evalue' in output
            full_output = output['evalue']
            assert re.search('__verify_notebook_error_ok__', full_output), dscr
          elif output['output_type'] == 'stream':
            full_output = '\n'.join(output['text'])
            if (warnings_are_errors and
                re.search(r'/(.*):\d+:.*(warn(ings?)?)\W', full_output, re.I)):
              assert re.search(r'__verify_notebook_warning_ok__', full_output),\
                  "Undeclared warning [%s] in cell: %s" % (full_output, dscr)
          elif output['output_type'] == 'display_data':
            if exhaustive is not None and 'png' in output:
              assert '__verify_notebook_png_ok__' == output['png'], \
                "Undeclared png in cell: " + dscr
          else:
            pass  # No automatic tests for successful code.
      else:
        pass  # No automatic tests for markdown

  if callback:
    with each_error():
      assert callback(None, None, "At end.")

  for spec in exh[len(cells):] + req:
    with each_error():
      assert 2 == len(spec), \
          ("Expected an input pattern and a list of tests,\n"
           "Found %s\nFor cell %s" % (spec, dscr))
      pttn, _test = spec
      assert False, "Expected pattern /%s/ not found" % (pttn,)

  assert 0 == len(errors), "See the error stream for individual error reports."

def run_and_verify_notebook(notebook_path_basename, **kwargs):
  '''runipy it and verify it. Adds ".ipynb" to the path.

  If the DEBUG_TESTS environment variable is set, cache notebook results in
  basename.out-ipynb and basename.err. That avoids rerunning the notebook
  when first writing tests for an existing notebook, or when debugging the
  tests themselves.
  '''
  if "DEBUG_TESTS" not in os.environ:
    check_results(get_out_and_err(notebook_path_basename), **kwargs)
  else:
    outfile=notebook_path_basename + ".out-ipynb"
    errfile=notebook_path_basename + ".err"
    if os.path.exists(outfile) and os.path.exists(errfile):
      check_results(get_out_and_err(outfile=outfile, errfile=errfile), **kwargs)
    else:
      (out, err) = get_out_and_err(notebook_path_basename)
      with open(outfile, 'w') as outfd:
        outfd.write(out)
      with open(errfile, 'w') as errfd:
        errfd.write(err)
      check_results((out, err), **kwargs)
