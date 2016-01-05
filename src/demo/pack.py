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

# Intended use pattern:
#
# Be in the directory that you want to pack. This will have at least
#    Satellites.ipynb and satellites.bdb in it, for the current demo.
#    It will NOT have any subdirectories. We pack files only.
# Do the packing.
#    python ~/pc/bdbcontrib/src/demo/pack.py \
#      'I am a human on a single-user machine' \
#      ~/pc/privhome/pubring.gpg > ~/packed_demo
#    (Need to create pubring.gpg? http://wooledge.org/~greg/crypto/node41.html )
# Note the public key it used! The one you gave it.
#    This program puts the hex form on stderr when done.
#    Add that to the list in main.py if it's not already there.
# Copy it to a test location on probcomp.csail:
#    pcwww=probcomp-4.csail.mit.edu:/afs/csail/proj/probcomp/www/data
#    scp ~/packed_demo $pcwww/bayesdb/demo/test.pack
# Check that it works:
#    mkdir test || rm -fr test/*
#    cd test
#    bayesdb-demo -u http://probcomp.csail.mit.edu/bayesdb/demo/test.pack
# or if you've modified the set of public keys, or other stuff in main.py:
#    cd ~/pc/bdbcontrib
#    python setup.py build
#    ~/pc/bdbcontrib/pythenv.sh \
#      ~/pc/bdbcontrib/build/scripts-2.7/bayesdb-demo \
#      -u http://probcomp.csail.mit.edu/bayesdb/demo/test.pack
# When that works, mv test.pack to a dated file (see others there) and update
# the "current" symlink when e.g. you've updated pypi distros to ones that can
# handle the new demos.

import base64
import ed25519
import json
import os
import sys
import zlib

def main():
    if len(sys.argv) < 3:
        sys.stderr.write('Usage: %s <magic> <seckey>\n' % (sys.argv[0],))
        sys.exit(1)
    if sys.argv[1] != 'I am a human on a single-user machine':
        sys.stderr.write(
            'This program is vulnerable to timing side channels.')
        sys.exit(1)
    with open(sys.argv[2], 'rb') as f:
        seckey = f.read()
    pubkey = ed25519.publickey(seckey)
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
    ed25519.checkvalid(sig, payload, (pubkey,))
    assert len(sig) == 64
    size = len(sig) + len(payload)
    if size > 64*1024*1024:
        sys.stderr.write('%s: pack too large after compression: %d bytes,'
            ' limit is %d bytes\n' %
            (sys.argv[0], size, 64*1024*1024))
        sys.exit(1)
    sys.stdout.write(sig)
    sys.stdout.write(payload)
    sys.stdout.flush()
    hexkey=''.join( [ "\\x%02X" % ord( x ) for x in pubkey ] ).strip()
    sys.stderr.write("wrote [%s] bytes with pubkey [%s]\n" % (size, hexkey))

if __name__ == '__main__':
    main()
