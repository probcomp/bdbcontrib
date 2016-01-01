import re
import json
import subprocess
import sys

def get_out_and_err():
  output = None
  if len(sys.argv) == 3:
    with open(sys.argv[1], 'r') as out:
      output = out.read()
    with open(sys.argv[2], 'r') as err:
      error = err.read()
  else:
    cmd = 'runipy --matplotlib --stdout Satellites.ipynb'
    p = subprocess.Popen(cmd, shell=True, stdin=None,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         close_fds=True)
    output = p.stdout.read()
    error = p.stderr.read()
  return (output, error)


def check_results(results, warnings_are_errors=False, content_tester=None):
  (output, error) = results
  notebook = json.loads(output)
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
            content_tester(cell)
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

        

def main():
  check_results(get_out_and_err())

if __name__ == "__main__":
  main()
