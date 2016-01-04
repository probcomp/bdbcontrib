# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2015, MIT Probabilistic Computing Project
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

import base64
import bayeslite
import contextlib
import ed25519
import getopt
import json
import os
import os.path
import requests
import sys
import zlib

from ..version import __version__

DEMO_URI = 'http://probcomp.csail.mit.edu/bayesdb/demo/current'
PUBKEY = '\x93\xca\x8f\xedds\x934B\xf8\xac\xee\x91A\x1d\xa9-\xf5\xfb\xe3\xbf\xe4\xea\xba\nG\xa5>z=\xc4\x8b'

short_options = 'hu:v'
long_options = [
    'help',
    'uri=',
    'version',
]

def main():
    # Parse options.
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_options, long_options)
    except getopt.GetoptError as e:
        sys.stderr.write('%s: %s\n' % (progname(), str(e)))
        usage(sys.stderr)
        sys.exit(2)
    demo_uri = DEMO_URI
    pubkey = PUBKEY
    for o, a in opts:
        if o in ('-v', '--version'):
            version(sys.stdout)
            sys.exit(0)
        elif o in ('-h', '--help'):
            usage(sys.stdout)
            sys.exit(0)
        elif o in ('-u', '--uri'):
            demo_uri = a
        else:
            assert False, 'invalid option %s' % (o,)

    # Select command.
    fetch_p = False
    launch_p = False
    if len(args) == 0:
        fetch_p = True
        launch_p = True
    elif args[0] == 'help':
        usage(sys.stdout)
        sys.exit(0)
    elif args[0] == 'version':
        version(sys.stdout)
        sys.exit(0)
    elif args[0] == 'fetch':
        if 1 < len(args):
            sys.stderr.write('%s: excess arguments\n' % (progname(),))
        fetch_p = True
    elif args[0] == 'launch':
        if 1 < len(args):
            sys.stderr.write('%s: excess arguments\n' % (progname(),))
        launch_p = True
    else:
        sys.stderr.write('%s: invalid command: %s\n' % (progname(), args[0]))
        usage(sys.stderr)
        sys.exit(2)

    # Fetch demo if requested.
    if fetch_p:
        if 0 < len(os.listdir(os.curdir)):
            sys.stderr.write('%s: enter empty directory first\n' %
                (progname(),))
            sys.exit(2)
        nretry = 3
        while 0 < nretry:
            try:
                demo = download_demo(demo_uri, pubkey)
            except Exception as e:
                sys.stderr.write('%s: %s\n' % (progname(), str(e)))
                nretry -= 1
                if nretry == 0:
                    sys.exit(1)
                else:
                    sys.stderr.write('Retrying %d more time%s.\n' %
                        (nretry, 's' if nretry > 1 else ''))
            else:
                break
        extract_demo(demo)

    # Launch demo if requested.
    if launch_p:
        try:
            os.execlp('ipython', 'ipython', 'notebook')
        except Exception as e:
            sys.stderr.write('%s: %s\n' % (progname(), str(e)))
            sys.stderr.write('%s: failed to launch ipython\n' % (progname(),))
            sys.exit(1)

def usage(out):
    out.write('Usage: %s [-hv] [-u <uri>]\n' % (progname(),))
    out.write('       %s [-hv] [-u <uri>] fetch\n' % (progname(),))
    out.write('       %s [-hv] [-u <uri>] launch\n' % (progname(),))

def progname():
    return os.path.basename(sys.argv[0])

def version(out):
    out.write('bdbcontrib %s\n' % (__version__,))
    out.write('bayeslite %s\n' % (bayeslite.__version__,))

class Fail(Exception):
    def __init__(self, string):
        self._string = string
    def __str__(self):
        return 'failed to download demo: %s' % (self._string,)

def fail(s):
    raise Fail(s)
def bad(s):
    # XXX Distinguish MITM on network from local failure?
    fail(s)

def selftest():
    payload = 'x\x9c\xab\xae\x05\x00\x01u\x00\xf9'
    try:
        if zlib.decompress(payload) != '{}':
            raise Exception
    except Exception:
        fail('compression self-test failed')
    sig = 'R6i&2\x911)\xce9Y\x0b&\xd2\xb0-<\xa5\rw\xc4)\xd6\xd4\x89\x03\x10\x8a;\x1e)\xfe\xb0\x92\xca?\xc3\x17\x0c\xc1\x84\xdd\xe6\xb2\xbfDZ\xe7Z\xd6*y\xe99\x9fk\x1e\xb9\x0f`\x07\xc0\x83\x08'
    try:
        ed25519.checkvalid(sig, payload, PUBKEY)
    except:
        fail('crypto self-test failed')

def download_demo(demo_uri, pubkey):
    with note('Requesting') as progress:
        headers = {
            'User-Agent': 'bdbcontrib demo'
        }
        r = requests.get(demo_uri, headers=headers, stream=True)
        r.raise_for_status()
        try:
            content_length = int(r.headers['content-length'])
        except Exception:
            bad('bad content-length')
        if content_length > 64*1024*1024:
            bad('demo too large')
        try:
            sig = r.iter_content(chunk_size=64).next()
        except Exception:
            bad('invalid signature')
        try:
            chunks = []
            so_far = 64
            for chunk in r.iter_content(chunk_size=65536):
                if content_length - so_far < len(chunk):
                    raise Exception
                so_far += len(chunk)
                progress(so_far, content_length)
                chunks.append(chunk)
            # Doesn't matter if content-length overestimates.
            payload = ''.join(chunks)
        except Exception:
            bad('invalid payload')
    with note('Verifying'):
        selftest()
        try:
            ed25519.checkvalid(sig, payload, pubkey)
        except Exception:
            bad('signature verification failed')
    with note('Decompressing'):
        try:
            demo_json = zlib.decompress(payload)
        except Exception:
            fail('decompression failed')
    with note('Parsing'):
        try:
            demo = json.loads(demo_json)
        except Exception:
            fail('parsing failed')
        if 'compatible' not in demo:
            fail('no compatibility information in demo')
        if not isinstance(demo['compatible'], list):
            fail('invalid compatible list')
        if 1 not in demo['compatible']:
            fail('demo too new, please upgrade software!')
        if 'files' not in demo:
            fail('no files in demo')
        if not isinstance(demo['files'], dict):
            fail('invalid files in demo')
        if not all(isinstance(k, unicode) for k in demo['files'].iterkeys()):
            fail('invalid file names in demo')
        if not all(k == os.path.basename(k) for k in demo['files'].iterkeys()):
            fail('invalid file base names in demo')
        if not all(isinstance(v, unicode) for v in demo['files'].itervalues()):
            fail('invalid file data in demo')
    return demo

def extract_demo(demo):
    for filename, data_b64 in sorted(demo['files'].iteritems()):
        with note('Decoding %s' % (filename,)):
            try:
                data = base64.b64decode(data_b64)
            except Exception:
                fail('failed to decode file: %s' % (filename,))
        with note('Extracting %s' % (filename,)):
            try:
                with open(filename, 'wb') as f:
                    f.write(data)
            except Exception:
                fail('failed to write file: %s' % (filename,))

@contextlib.contextmanager
def note(head):
    start = '%s...' % (head,)
    sys.stdout.write(start)
    sys.stdout.flush()
    def progress(n, d):
        if os.isatty(sys.stdout.fileno()):
            sys.stdout.write('\r%s %d/%d' % (start, n, d))
            sys.stdout.flush()
    try:
        yield progress
        sys.stdout.write(' done\n')
        sys.stdout.flush()
    except Exception:
        sys.stdout.write(' failed\n')
        sys.stdout.flush()
        raise
