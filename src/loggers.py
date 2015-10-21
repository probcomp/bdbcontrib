from __future__ import print_function
import matplotlib.pyplot as plt
import IPython.utils.warn
import logging
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
    print(msg_format % values)
  def warn(self, msg_format, *values):
    '''For warnings or non-fatal errors.'''
    print(msg_format % values, file=sys.stderr)
  def plot(self, _suggested_name, _figure):
    '''For plotting. Name is a string, figure a matplotlib object.'''
    plt.show()
  def result(self, msg_format, *values):
    '''For formatted text results. In unix, this would be stdout.'''
    print(msg_format % values)
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
  def plot(self, suggested_name, _figure):
    plt.savefig(suggested_name + ".png")
  def debug(self, *args, **kwargs):
    logging.debug(*args, **kwargs)
  def exception(self, *args, **kwargs):
    logging.exception(*args, **kwargs)

class IpyLogger(BqlLogger):
  def info(self, msg_format, *values):
    IPython.utils.warn.info(msg_format % values)
  def warn(self, msg_format, *values):
    IPython.utils.warn.warn(msg_format % values)
