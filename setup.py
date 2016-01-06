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

try:
    from setuptools import setup
    from setuptools.command.test import test as TestCommand
except ImportError:
    from distutils.core import setup
    from distutils.cmd import Command
    class TestCommand(Command):
        user_options = []
        def initialize_options(self): pass
        def finalize_options(self): pass
        def run(self): self.run_tests()

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
        import re
        match = re.match(r'v([^-]*)-([0-9]+)-(.*)$', desc)
        if match is None:       # paranoia
            version += 'unknown'
        else:
            ver, rev, local = match.groups()
            version = '%s.post%s+%s' % (ver, rev, local.replace('-', '.'))
            assert '-' not in version

try:
    with open('src/version.py', 'rU') as f:
        version_old = f.readlines()
except IOError:
    version_old = None
version_new = ['__version__ = %s\n' % (repr(version),)]
if version_old != version_new:
    with open('src/version.py', 'w') as f:
        f.writelines(version_new)

# XXX Several horrible kludges here to make `python setup.py test' work:
#
# - Standard setputools test command searches for unittest, not
#   pytest.
#
# - pytest suggested copypasta assumes . works in sys.path; we
#   deliberately make . not work in sys.path and require ./build/lib
#   instead, in order to force a clean build.
#
# - Must set PYTHONPATH too for shell tests, which fork and exec a
#   subprocess which inherits PYTHONPATH but not sys.path.
#
# - build command's build_lib variable is relative to source
#   directory, so we must assume os.getcwd() gives that.
#
# (The shell test issues are irrelevant for the moment in bdbcontrib,
# but there's no harm in setting PYTHONPATH anyway.)
class cmd_pytest(TestCommand):
    def __init__(self, *args, **kwargs):
        TestCommand.__init__(self, *args, **kwargs)
        self.test_suite = 'tests'
        self.build_lib = None
    def finalize_options(self):
        TestCommand.finalize_options(self)
        # self.build_lib = ...
        self.set_undefined_options('build', ('build_lib', 'build_lib'))
    def run_tests(self):
        import pytest
        import os
        import os.path
        import sys
        sys.path = [os.path.join(os.getcwd(), self.build_lib)] + sys.path
        os.environ['BAYESDB_WIZARD_MODE'] = '1'
        os.environ['BAYESDB_DISABLE_VERSION_CHECK'] = '1'
        os.environ['PYTHONPATH'] = ':'.join(sys.path)
        sys.exit(pytest.main(['tests']))

setup(
    name='bdbcontrib',
    version=version,
    description='Hodgepodge library of extras for bayeslite',
    url='http://probcomp.csail.mit.edu/bayesdb',
    author='MIT Probabilistic Computing Project',
    author_email='bayesdb@mit.edu',
    license='Apache License, Version 2.0',
    install_requires=[
        'bayeslite>=0.1.4',
        'ipython[notebook]>=3',
        'markdown2',
        'matplotlib==1.4.3',  # See bdbcontrib/issues/94
        'numpy==1.8.2',       # See bdbcontrib/issues/94
        'numpydoc',
        'pandas',
        'seaborn>=0.6',
        'scipy==0.15.1',      # See bdbcontrib/issues/94
        'sklearn',
        'sklearn-pandas',
        'sphinx',
        'tornado>=4.0',
    ],
    tests_require=[
        'mock',
        'pillow',
        'pytest',
    ],
    packages=[
        'bdbcontrib',
        'bdbcontrib.predictors',
        'bdbcontrib.metamodels',
        'bdbcontrib.demo',
        'bdbcontrib.demo.ed25519',
        'bdbcontrib.experiments',
    ],
    package_dir={
        'bdbcontrib': 'src',
    },
    scripts=[
        'scripts/bayesdb-demo',
    ],
    cmdclass={
        'test': cmd_pytest,
    },
)
