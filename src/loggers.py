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

import cgi
import sys
import traceback
import IPython.display
import IPython.utils.warn

from bayeslite.exception import BayesLiteException
from bayeslite.loggers import BqlLogger

class IpyLogger(BqlLogger):
  def info(self, msg_format, *values):
    message = msg_format % values
    html = ('<dib class="inner_cell" style="background-color:#F7F8E0">' +
            cgi.escape(message) + '</div>')
    IPython.display.display_html(html, raw=True)
  def warn(self, msg_format, *values):
    IPython.utils.warn.warn(msg_format % values)
  def exception(self, msg_format, *values):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type:
      if isinstance(exc_value, (BayesLiteException, ValueError)):
        IPython.utils.warn.error(msg_format + '\n' + str(exc_value), *values)
      else:
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        IPython.utils.warn.error('\n'.join(lines))
    else:
      self.warn("ERROR: " + msg_format, *values)

IPYTHON_LOGGER = IpyLogger()
def bayeslite_aware_render_traceback(self):
  IPYTHON_LOGGER.exception("")
  return []
BayesLiteException._render_traceback_ = bayeslite_aware_render_traceback
