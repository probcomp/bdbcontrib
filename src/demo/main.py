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

import base64
import contextlib
import ed25519
import json
import requests
import os
import sys
import zlib

def main():
    fetch_p = len(sys.argv) < 2 or sys.argv[1] == 'fetch'
    launch_p = len(sys.argv) < 2 or sys.argv[1] == 'launch'
    if fetch_p:
        if 0 < len(os.listdir(os.curdir)):
            print >>sys.stderr, 'Please enter an empty directory first!'
            sys.exit(1)
        nretry = 3
        last_error = None
        while 0 < nretry:
            try:
                download_demo()
            except Exception as e:
                last_error = e
                nretry -= 1
            else:
                break
        if last_error is not None:
            print >>sys.stderr, last_error
            sys.exit(1)
    if launch_p:
        try:
            os.execlp('ipython', 'ipython', 'notebook')
        except Exception as e:
            print >>sys.stderr, e
            print >>sys.stderr, 'Failed to launch ipython!'
            sys.exit(1)

class Fail(Exception):
    def __init__(self, string):
        self._string = string
    def __str__(self):
        return 'Failed to download demo: %s' % (self._string,)

def fail(s):
    raise Fail(s)
def bad(s):
    # XXX Distinguish MITM on network from local failure?
    fail(s)

#DEMO_URI = 'http://probcomp.csail.mit.edu/bayesdb/demo/current.json'
DEMO_URI = 'http://127.0.0.1:12345/'
PUBKEY = '\x93\xca\x8f\xedds\x934B\xf8\xac\xee\x91A\x1d\xa9-\xf5\xfb\xe3\xbf\xe4\xea\xba\nG\xa5>z=\xc4\x8b'

def selftest(sig0, payload0):
    payload = 'x\x9c\xab\xae\x05\x00\x01u\x00\xf9'
    try:
        if zlib.decompress(payload) != '{}':
            raise Exception
    except Exception:
        fail('Compression self-test failed!')
    sig = 'R6i&2\x911)\xce9Y\x0b&\xd2\xb0-<\xa5\rw\xc4)\xd6\xd4\x89\x03\x10\x8a;\x1e)\xfe\xb0\x92\xca?\xc3\x17\x0c\xc1\x84\xdd\xe6\xb2\xbfDZ\xe7Z\xd6*y\xe99\x9fk\x1e\xb9\x0f`\x07\xc0\x83\x08'
    try:
        ed25519.checkvalid(sig, payload, PUBKEY)
    except:
        fail('Crypto self-test failed!')

def download_demo():
    with note('Requesting') as progress:
        headers = {
            'User-Agent': 'bdbcontrib demo'
        }
        r = requests.get(DEMO_URI, headers=headers, stream=True)
        try:
            content_length = int(r.headers['content-length'])
        except Exception:
            bad('Bad content-length!')
        if content_length > 64*1024*1024:
            bad('Demo too large!')
        try:
            sig = r.iter_content(chunk_size=64, decode_unicode=False).next()
        except Exception:
            bad('Invalid signature!')
        try:
            chunks = []
            so_far = 64
            for chunk in r.iter_content(chunk_size=1024):
                if content_length - so_far < len(chunk):
                    raise Exception
                so_far += len(chunk)
                progress(so_far, content_length)
                chunks.append(chunk)
            # Doesn't matter if content-length overestimates.
            payload = ''.join(chunks)
        except Exception:
            bad('Invalid payload!')
    with note('Verifying'):
        selftest(sig, payload)
        try:
            ed25519.checkvalid(sig, payload, PUBKEY)
        except Exception:
            with open('/tmp/riastradh/bad.pack', 'wb') as f:
                f.write(sig)
                f.write(payload)
            bad('Signature verification failed!')
    with note('Decompressing'):
        try:
            demo_json = zlib.decompress(payload)
        except Exception:
            fail('Decompression failed!')
    with note('Parsing'):
        try:
            demo = json.loads(demo_json)
        except Exception:
            fail('Parsing failed!')
    for filename, data_b64 in sorted(demo.iteritems()):
        with note('Decoding %s' % (filename,)):
            try:
                data = base64.b64decode(data_b64)
            except Exception:
                fail('Failed to decode file: %s' % (filename,))
        with note('Extracting %s' % (filename,)):
            try:
                with open(filename, 'wb') as f:
                    f.write(data)
            except Exception:
                fail('Failed to write file: %s' % (filename,))

@contextlib.contextmanager
def note(head):
    start = '%s...' % (head,)
    sys.stdout.write(start)
    sys.stdout.flush()
    def progress(n, d):
        if os.isatty(sys.stdout.fileno()):
            sys.stdout.write('\r%s %d/%d' % (start, n, d))
            sys.stdout.flush()
    yield progress
    sys.stdout.write(' done\n')
    sys.stdout.flush()

if __name__ == '__main__':
    main()
