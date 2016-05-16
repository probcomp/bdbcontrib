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

try:
    from setuptools import setup
    from setuptools.command.build_py import build_py
    from setuptools.command.sdist import sdist
    from setuptools.command.test import test as TestCommand
except ImportError:
    from distutils.core import setup
    from distutils.cmd import Command
    from distutils.command.build_py import build_py
    from distutils.command.sdist import sdist

    class TestCommand(Command):
        user_options = []
        def initialize_options(self): pass
        def finalize_options(self): pass
        def run(self): self.run_tests()

def get_version():
    with open('VERSION', 'rb') as f:
        version = f.read().strip()

    # Append the Git commit id if this is a development version.
    if version.endswith('+'):
        import re
        import subprocess
        version = version[:-1]
        tag = 'v' + version
        desc = subprocess.check_output([
            'git', 'describe', '--dirty', '--long', '--match', tag,
        ])
        match = re.match(r'^v([^-]*)-([0-9]+)-(.*)$', desc)
        assert match is not None
        verpart, revpart, localpart = match.groups()
        assert verpart == version
        # Local part may be g0123abcd or g0123abcd-dirty.  Hyphens are
        # not kosher here, so replace by dots.
        localpart = localpart.replace('-', '.')
        full_version = '%s.post%s+%s' % (verpart, revpart, localpart)
    else:
        full_version = version

    # Strip the local part if there is one, to appease pkg_resources,
    # which handles only PEP 386, not PEP 440.
    if '+' in full_version:
        pkg_version = full_version[:full_version.find('+')]
    else:
        pkg_version = full_version

    # Sanity-check the result.  XXX Consider checking the full PEP 386
    # and PEP 440 regular expressions here?
    assert '-' not in full_version, '%r' % (full_version,)
    assert '-' not in pkg_version, '%r' % (pkg_version,)
    assert '+' not in pkg_version, '%r' % (pkg_version,)

    return pkg_version, full_version

pkg_version, full_version = get_version()

def write_version_py(path):
    try:
        with open(path, 'rb') as f:
            version_old = f.read()
    except IOError:
        version_old = None
    version_new = '__version__ = %r\n' % (full_version,)
    if version_old != version_new:
        print 'writing %s' % (path,)
        with open(path, 'wb') as f:
            f.write(version_new)

from distutils.command.install import INSTALL_SCHEMES
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']
example_files = {}
import os
import re
with open('MANIFEST.in', 'r') as manifest:
    for line in manifest:
        line = line.strip()
        if line.startswith('include examples'):
            line = line[len("include "):]
            dirname = os.path.join('bdbcontrib', os.path.dirname(line))
            if dirname not in example_files:
                example_files[dirname] = []
            example_files[dirname].append(line)

class local_build_py(build_py):
    def run(self):
        write_version_py(version_py)
        build_py.run(self)

# Make sure the VERSION file in the sdist is exactly specified, even
# if it is a development version, so that we do not need to run git to
# discover it -- which won't work because there's no .git directory in
# the sdist.
class local_sdist(sdist):
    def make_release_tree(self, base_dir, files):
        import os
        sdist.make_release_tree(self, base_dir, files)
        version_file = os.path.join(base_dir, 'VERSION')
        print('updating %s' % (version_file,))
        # Write to temporary file first and rename over permanent not
        # just to avoid atomicity issues (not likely an issue since if
        # interrupted the whole sdist directory is only partially
        # written) but because the upstream sdist may have made a hard
        # link, so overwriting in place will edit the source tree.
        with open(version_file + '.tmp', 'wb') as f:
            f.write('%s\n' % (pkg_version,))
        os.rename(version_file + '.tmp', version_file)

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

# XXX These should be attributes of `setup', but helpful distutils
# doesn't pass them through when it doesn't know about them a priori.
version_py = 'src/version.py'

setup(
    name='bdbcontrib',
    version=pkg_version,
    description='Hodgepodge library of extras for bayeslite',
    url='http://probcomp.csail.mit.edu/bayesdb',
    author='MIT Probabilistic Computing Project',
    author_email='bayesdb@mit.edu',
    license='Apache License, Version 2.0',
    install_requires=[
        'bayeslite==0.1.8',
        'ipython[notebook]>=3',
        'markdown2',
        'matplotlib',
        'numpy',
        'numpydoc',
        'pandas>=0.17.0',
        'seaborn>=0.7',  # seaborn 0.6 is at the root of #94.
        'scipy',
        'sklearn',
        'sklearn-pandas',
        'sphinx',
        'tornado>=4.0',
    ],
    tests_require=[
        'flaky',  # for bdb-experiments
        'mock',
        'pillow',
        'pytest',
    ],
    packages=[
        'bdbcontrib',
        'bdbcontrib.predictors',
        'bdbcontrib.metamodels',
        'bdbcontrib.experiments',
    ],
    package_dir={
        'bdbcontrib': 'src',
    },
    data_files=example_files.items(),
    scripts=[
        'scripts/bayesdb-demo',
    ],
    cmdclass={
        'build_py': local_build_py,
        'sdist': local_sdist,
        'test': cmd_pytest,
    },
)
