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

from __future__ import print_function

import IPython.utils.warn
import logging
import matplotlib.pyplot as plt
import re
import sys
import traceback

class BqlLogger(object):
  '''A logger object for BQL.

     The idea of having a custom one is to make it easy to adapt to other
     loggers, like python's builtin one (see LoggingLogger below) or the one
     for iPython notebooks (see IpyLogger below), or for testing, or to set
     preferences (see DebugLogger, QuietLogger, SilentLogger below).

     Loggers should implement functions with the signatures of this base class.
     They are welcome to inherit from it.  Do not depend on return values from
     any of these methods.
  '''
  def info(self, msg_format, *values):
    '''For progress and other informative messages.'''
    if len(values) > 0:
      msg_format = msg_format % values
    print(msg_format)
  def warn(self, msg_format, *values):
    '''For warnings or non-fatal errors.'''
    if len(values) > 0:
      msg_format = msg_format % values
    print(msg_format, file=sys.stderr)
  def plot(self, _suggested_name, figure):
    '''For plotting.

    Name : str
      A filename fragment or window title, not intended to be part of the
      figure, not intended to be a fully qualified path, but of course if
      you know more about the particular logger handling your case, then
      use as you will.
    Figure : a matplotlib object
      on which .show or .savefig might be called.
    '''
    if (hasattr(figure, 'show')):
      figure.show()
    else:
      print(repr(figure))
  def result(self, msg_format, *values):
    '''For formatted text results. In unix, this would be stdout.'''
    if len(values) > 0:
      msg_format = msg_format % values
    print(msg_format)
  def debug(self, _msg_format, *_values):
    '''For debugging information.'''
    pass
  def exception(self, msg_format, *values):
    '''For fatal or fatal-if-uncaught errors.'''
    self.warn('ERROR: ' + msg_format, *values)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type:
      lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
      self.warn('\n'.join(lines))
  def format_escape(str):
    str = re.sub(r"%([^\(])", r"%%\1", str)
    str = re.sub(r"%$", r"%%", str)  # There was a % at the end?
    return str

class DebugLogger(BqlLogger):
  def debug(self, msg_format, *values):
    self.warn('DEBUG: ' + msg_format % values)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type:
      lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
      self.warn('\n'.join(lines))

class QuietLogger(BqlLogger):
  def info(self, _msg_format, *_values):
    pass
  def warn(self, _msg_format, *_values):
    pass

class SilentLogger(QuietLogger):
  def plot(self, _suggested_name, _figure):
    pass
  def result(self, _msg, *_values):
    pass
  def debug(self, _msg, *_values):
    pass
  def exception(self, _msg, *_values):
    pass

class LoggingLogger(BqlLogger):
  def info(self, msg_format, *values):
    logging.info(msg_format, *values)
  def warn(self, msg_format, *values):
    logging.warning(msg_format, *values, file=sys.stderr)
  def plot(self, suggested_name, figure):
    figure.savefig(suggested_name + ".png")
  def debug(self, *args, **kwargs):
    logging.debug(*args, **kwargs)
  def exception(self, *args, **kwargs):
    logging.exception(*args, **kwargs)

class IpyLogger(BqlLogger):
  def info(self, msg_format, *values):
    IPython.utils.warn.info(msg_format % values)
  def warn(self, msg_format, *values):
    IPython.utils.warn.warn(msg_format % values)
