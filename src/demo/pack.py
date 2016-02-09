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

import base64
import ed25519
import hashlib
import json
import os
import sys
import zlib

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s <magic> <seckey>\n' % (progname(),))
        sys.exit(1)
    if sys.argv[1] != 'I am a human on a single-user machine':
        sys.stderr.write('%s: This program is vulnerable'
            ' to timing side channels.\n' % (progname(),))
        sys.exit(1)
    fd = os.open(sys.argv[2], os.O_RDONLY)
    try:
        if os.fstat(fd).st_size != 64:
            raise IOError('secret key file must be 64 bytes')
        seckey = os.read(fd, 32)
        cksum = os.read(fd, 32)
        assert os.lseek(fd, 0, os.SEEK_CUR) == 64
        pubkey = ed25519.publickey(seckey)
        cksum_ctx = hashlib.sha256()
        cksum_ctx.update(seckey)
        cksum_ctx.update(pubkey)
        cksum_ctx.update('This is a bayesdb-demo version 2 signing key.')
        if cksum != cksum_ctx.digest():
            raise IOError('bad checksum')
    except Exception as e:
        sys.stderr.write('%s: Error reading secret key file: %s\n' %
            (progname(), str(e)))
        sys.exit(1)
    finally:
        os.close(fd)
    demo = {}
    demo['compatible'] = [1]
    demo['files'] = {}
    for filename in os.listdir(os.curdir):
        with open(filename, 'rb') as f:
            data = f.read()
        demo['files'][filename] = base64.b64encode(data)
    demo_json = json.dumps(demo, sort_keys=True)
    payload = zlib.compress(demo_json)
    sig = ed25519.signature(payload, seckey, pubkey)
    ed25519.checkvalid(sig, payload, pubkey)
    assert len(sig) == 64
    size = len(sig) + len(payload)
    if size > 64*1024*1024:
        sys.stderr.write('%s: Pack is too large after compression: %d bytes,'
            ' limit is %d bytes\n' %
            (progname(), size, 64*1024*1024))
        sys.exit(1)
    sys.stdout.write(sig)
    sys.stdout.write(payload)
    sys.stdout.flush()

def progname():
    return os.path.basename(sys.argv[0])

if __name__ != '__main__':
    raise Exception('pack is standalone only')
main()
