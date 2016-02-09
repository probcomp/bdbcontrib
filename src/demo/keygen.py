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

import ed25519
import hashlib
import os
import sys

def main():
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s <seckey> <pubkey>\n' % (progname(),))
        sys.exit(1)
    seckey_path = sys.argv[1]
    pubkey_path = sys.argv[2]
    close = []
    unlink = []
    try:
        seckey_fd = os.open(seckey_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        close.append(seckey_fd)
        unlink.append(seckey_path)
        pubkey_fd = os.open(pubkey_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
        close.append(pubkey_fd)
        unlink.append(pubkey_path)
        seckey = os.urandom(32)
        assert len(seckey) == 32
        pubkey = ed25519.publickey(seckey)
        assert len(pubkey) == 32
        cksum_ctx = hashlib.sha256()
        cksum_ctx.update(seckey)
        cksum_ctx.update(pubkey)
        cksum_ctx.update('This is a bayesdb-demo version 2 signing key.')
        cksum = cksum_ctx.digest()
        assert len(cksum) == 32
        os.write(seckey_fd, seckey)
        os.write(seckey_fd, cksum)
        assert os.lseek(seckey_fd, 0, os.SEEK_CUR) == 64
        assert os.fstat(seckey_fd).st_size == 64
        os.write(pubkey_fd, '%s\n' % (repr(pubkey),))
    except Exception as e:
        for fd in close:
            try:
                os.close(fd)
            except:
                pass
        for path in unlink:
            try:
                os.unlink(path)
            except:
                pass
        sys.stderr.write('%s: %s\n' % (progname(), str(e)))

def progname():
    return os.path.basename(sys.argv[0])

if __name__ != '__main__':
    raise Exception('bayesdb-demo pack keygen is standalone only')
main()
