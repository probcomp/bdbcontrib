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

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('VERSION', 'rU') as f:
    version = f.readline().strip()

# Append the Git commit id if this is a development version.
if version.endswith('+'):
    tag = 'v' + version[:-1]
    try:
        import subprocess
        desc = subprocess.check_output([
            'git', 'describe', '--dirty', '--match', tag,
        ])
    except Exception:
        version += 'unknown'
    else:
        assert desc.startswith(tag)
        version = desc[1:].strip()

try:
    with open('src/version.py', 'rU') as f:
        version_old = f.readlines()
except IOError:
    version_old = None
version_new = ['__version__ = %s\n' % (repr(version),)]
if version_old != version_new:
    with open('src/version.py', 'w') as f:
        f.writelines(version_new)

setup(
    name='bdbcontrib',
    version=version,
    description='Hodgepodge library of extras for bayeslite',
    url='http://probcomp.csail.mit.edu/bayesdb',
    author='MIT Probabilistic Computing Project',
    author_email='bayesdb@mit.edu',
    license='Apache License, Version 2.0',
    install_requires=[
        'ipython[notebook]>=3',
        'markdown2',
        'matplotlib',
        'numpy',
        'numpydoc',
        'pandas',
        'seaborn==0.5.1',
        'sphinx',
    ],
    packages=[
        'bdbcontrib',
        'bdbcontrib.demo',
        'bdbcontrib.demo.ed25519',
    ],
    package_dir={
        'bdbcontrib': 'src',
    },
    scripts=[
        'scripts/bayesdb-demo',
    ],
)
