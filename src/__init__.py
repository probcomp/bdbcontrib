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

"""Main bdbcontrib API.

The bdbcontrib module serves a sandbox for experimental and semi-stable
features that are not yet ready for integreation to the bayeslite repository.
"""

from version import __version__
from py_utils import helpsub
from population import Population

@helpsub("__population_doc__", Population.__init__.__doc__)
def quickstart(*args, **kwargs):
  """__population_doc__"""
  return Population(*args, **kwargs)

__all__ = [
    'quickstart',
    'Population',
    '__version__',
]
