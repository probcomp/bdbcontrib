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


###################################
# Decorators for library providers.

from collections import namedtuple
import inspect
import pydoc
import re
import types

from bayeslite.loggers import logged_query
from py_utils import helpsub
from population import Population

# This class is all about dynamically generated methods, so
# pylint: disable=no-member

PopulationTransformation = namedtuple(
    'PopulationTransformation',
    ['name', 'decorated_doc', 'method_doc', 'doc', 'transform'])

POPULATION_TRANSFORMATIONS = [
    PopulationTransformation(
        name='population',
        decorated_doc='''a bdbcontrib.population.Population''',
        method_doc=None,  # Is part of self, and should be hidden.
        doc='''Passes on the population object itself.''',
        transform=lambda pop: pop
        ),
    PopulationTransformation(
        name='interpret_bql',
        decorated_doc='''A BQL query (no %t or %g).''',
        method_doc='''A BQL query.
    You can also use %t and %g for the population-associated table name and
    the name of the population's generative-population-model-instance-generator,
    respectively.''',
        doc='''Fills in %t and %g in a BQL query.''',
        transform=lambda pop, arg: pop.interpret_bql(arg),
        ),
    PopulationTransformation(
        name='specifier_to_df',
        decorated_doc='''a pandas.DataFrame.''',
        method_doc='''a pandas.DataFrame or BQL query.''',
        doc='''Queries to create a dataframe from BQL if need be.''',
        transform=lambda pop, arg: pop.specifier_to_df(arg),
        ),
    PopulationTransformation(
        name='population_to_bdb',
        decorated_doc='''An active bayeslite.BayesDB instance.''',
        method_doc=None,
        doc='''Provides the population's BayesDB instance.''',
        transform=lambda pop: pop.bdb,
        ),
    PopulationTransformation(
        name='population_name',
        decorated_doc='''str
        the name of the relevant table in the bdb.''',
        method_doc=None,
        doc='''Provides the population's name.''',
        transform=lambda pop: pop.name,
        ),
    PopulationTransformation(
        name='generator_name',
        decorated_doc='''str
        the name of the relevant generator in the bdb.''',
        method_doc=None,
        doc='''Provides the population's generative population model's name.''',
        transform=lambda pop: pop.generator_name,
        ),
    PopulationTransformation(
        name='logger',
        decorated_doc='''A bayeslite.logger.BQLLogger instance.''',
        method_doc=None,
        doc='''Provides a logger to which one can .info, .warn, .plot, etc.''',
        transform=lambda pop: pop.logger,
        ),
    ]


ARGSPEC_TRANSFORMS = dict([(pt.name, pt.transform)
                           for pt in POPULATION_TRANSFORMATIONS])
METHOD_DOC_FILLERS = dict([('__'+pt.name+'__', pt.method_doc)
                           for pt in POPULATION_TRANSFORMATIONS])
DECORATED_DOC_FILLERS = dict([('__'+pt.name+'__', pt.decorated_doc)
                              for pt in POPULATION_TRANSFORMATIONS])
PT_DOC='\n'.join(["%s: %s\n\t%s" %
                  (pt.name, ("" if pt.method_doc is None else "[argspec]"),
                   pt.doc)
                  for pt in POPULATION_TRANSFORMATIONS])


def fill_documentation(docstr, fillers):
  if docstr is None:
    return None
  for key, explanation in fillers.iteritems():
    if explanation is None: # Remove it entirely.
      docstr = re.sub(r'\s*\w+\s*:\s*' + key, '\n', docstr, re.M)
    else:
      docstr = re.sub(key, explanation, docstr)
  return docstr

def redocument(fn, xfrms, fillers):
  title = 'wrapped as population.' + fn.__name__
  args, varargs, varkw, defaults = inspect.getargspec(fn)
  argspec = inspect.formatargspec(
      xfrms['required_names'] + xfrms['optional_names'],
      xfrms['varargs'], xfrms['varkw'], defaults,
      formatvalue=lambda x: '=' + repr(x))
  note = fill_documentation(fn.__doc__, fillers)
  first_nonempty_paragraph = []
  if note is None:
    note = ""
  else:
    first_nonempty_paragraph.append("")  # So join adds prior line break.
    lines = re.split(r'\n', note)
    for line in lines:
      line = line.strip()
      if line == "" and first_nonempty_paragraph[-1]:
        break
      if line != "":
        first_nonempty_paragraph.append(line)
    note = '\n\n' + note
  doc = title + argspec + note
  shortdoc = xfrms['name'] + argspec + "\n    ".join(first_nonempty_paragraph)
  return (doc, shortdoc)

def compile_argspec_transforms(fn, argspecs):
  '''Create a data structure to pass to apply_argspec_transforms.

  fn : the function that the argspecs should apply to.
  argspecs : argument specifiers as described in population_method.
  '''
  # There are two kinds of transforms: Ones that rely on just the population
  # object, and those that rely on both that and the data passed in for that
  # name.
  #
  # For those only in terms of the population: for optional arguments, supply
  # them if they are not present, passing through any value if they are.
  # Where an outside value is requested pass that in to the transformer as
  # specified.
  #
  # Positional arguments are similar, if they want just population, fill that
  # in and remove the associated argument. If they want a population and arg,
  # then satisfy that based on the default specified nearby.

  co_varnames, varargs, varkw, co_defaults = inspect.getargspec(fn)
  num_optional = 0 if co_defaults is None else len(co_defaults)
  num_required = len(co_varnames) - num_optional
  required_transformers = [None] * num_required
  optional_transformers = {}
  required_names = co_varnames[:num_required]
  optional_names = co_varnames[num_required:]
  for key, argspec in argspecs.iteritems():
    xform = ARGSPEC_TRANSFORMS[key]
    for argitem in (argspec if hasattr(argspec, '__iter__') else [argspec]):
      if isinstance(argitem, int):
        if argitem >= num_required or argitem < 0:
          raise IndexError(
            'Invalid positional specifier for %s: %d' % (fn.__name__, argitem))
        if required_transformers[argitem] is not None:
          raise IndexError(
            'Duplicate specifier for %s: %d' % (fn.__name__, argitem))
        required_transformers[argitem] = xform
      else:
        if argitem not in optional_names:
          raise IndexError(
            'No such optional argument for %s: %s' % (fn.__name__, argitem))
        if argitem in optional_transformers:
          raise IndexError(
            'Duplicate specifier for %s: %s' % (fn.__name__, argitem))
        population_only = (xform.__code__.co_argcount == 1)
        if population_only:
          def wrap_pop_only_opt(index, xfn): # To copy, not reference, free vars
            return lambda self, kwargs: \
              kwargs[index] if index in kwargs else xfn(self)
          newf = wrap_pop_only_opt(argitem, xform)
        else:
          def wrap_pop_arg_opt(xfn, index): # To copy, not reference, free vars
            return lambda self, kwargs: xfn(self, kwargs[index])
          newf = wrap_pop_arg_opt(xform, argitem)
        optional_transformers[argitem] = newf
  for name in optional_names:
    if name not in optional_transformers:
      def wrap_optarg_passthru(idx):
        return lambda _self, kwargs: kwargs[idx] if idx in kwargs else None
      optional_transformers[name] = wrap_optarg_passthru(name)


  wrapper_i = 0
  wrapped_transformers = []
  required_method_names = []
  for (decorated_i, xform) in enumerate(required_transformers):
    if xform is None: # No mention of this argument, so pass it through.
      def wrap_default(index): # To copy, not reference, free vars
        return lambda self, args: args[index]
      wrapped_transformers.append(wrap_default(wrapper_i))
      required_method_names.append(required_names[decorated_i])
      wrapper_i += 1
    else:
      population_only = (xform.__code__.co_argcount == 1)
      if population_only:
        def wrap_pop_only_req(xfn): # To copy, not reference, free vars.
          return lambda self, args: xfn(self)
        wrapped_transformers.append(wrap_pop_only_req(xform))
      else:
        def wrap_pop_arg_req(xfn, index): # To copy, not reference, free vars.
          return lambda self, args: xfn(self, args[index])
        wrapped_transformers.append(wrap_pop_arg_req(xform, wrapper_i))
        required_method_names.append(required_names[decorated_i])
        wrapper_i += 1
  return {'name': fn.__module__ + '.' + fn.__code__.co_name,
          'varargs': varargs,
          'varkw': varkw,
          'required_names': required_method_names,
          'required_transformers': wrapped_transformers,
          'optional_names': optional_names,
          'optional_transformers': optional_transformers,
          'co_defaults': co_defaults,
          }


def apply_argspec_transforms(pop, argspecs, args, kwargs):
  '''Apply the given transforms to the args and kwargs as specified.

     argspec: number, string, or list of numbers and strings.
         Not None. A positional argument number in args or a keyword argument
         name in kwargs.
     transforms: ([fn(pop, args)-->value], {'kwname': fn(pop, kwargs)-->value})
         The transformations to apply.
     args: tuple or list
         The args to be transformed. Note the lack of a star: just a list.
     kwargs: dict
         The kwargs to be transformed. Note the lack of stars: just a dict.

     Returns transformed (args, kwargs)
     '''
  required_names = argspecs['required_names']
  required_xforms = argspecs['required_transformers']
  optional_names = argspecs['optional_names']
  optional_xforms = argspecs['optional_transformers']
  co_defaults = argspecs['co_defaults']
  completed_args = list(args[:]) # mutable copy
  remaining_kwargs = kwargs.copy()

  # Check for duplicate args (to get the message right).
  all_names = required_names + optional_names
  for name in all_names[:len(args)]:
    if name in remaining_kwargs:
      raise TypeError("%s() got multiple values for keyword argument '%s'" %
                      (argspecs['name'], name))

  # Named required args:
  for name in required_names[len(args):]:
    if name in remaining_kwargs:
      completed_args.append(remaining_kwargs[name])
      del remaining_kwargs[name]
    else:
      raise TypeError("%s() takes at least %d arguments (%d given)" %
                      (argspecs['name'], len(required_names),
                       len(args) + len(kwargs)))

  new_varargs = []
  # Positional optional args:
  for opt_i in xrange(len(required_names), len(args)):
    value = args[opt_i]
    found = False
    for name in optional_names:
      if name not in remaining_kwargs:
        remaining_kwargs[name] = value
        found = True
        break
    if not found:
      if argspecs['varargs'] is not None:
        new_varargs.append(value)
      else:
        max_args = len(required_names) + len(optional_names)
        if max_args == 0:
          raise TypeError("%s() takes no arguments (%d given)" %
                          (argspecs['name'], len(args) + len(kwargs)))
        else:
          raise TypeError("%s() takes at most %d arguments (%d given)" %
                          (argspecs['name'], max_args, len(args) + len(kwargs)))

  new_args = []
  # Positional required args:
  for xform in required_xforms:
    new_args.append(xform(pop, completed_args))
  # Named optional args:
  for key in optional_names:
    xform = optional_xforms[key]
    new_args.append(xform(pop, remaining_kwargs))
    if key in remaining_kwargs:
      del remaining_kwargs[key]

  if argspecs['varkw'] is None:
    for key in remaining_kwargs:
      raise TypeError("%s() got an unexpected keyword argument '%s'" %
                      (argspecs['name'], key))

  # Allow the function's own defaults to override Nones:
  for i, optname in enumerate(optional_names):
      index = len(required_xforms) + i
      if new_args[index] is None:
          new_args[index] = co_defaults[i]

  return (new_args + new_varargs, remaining_kwargs)

@helpsub("__PT_DOC__", PT_DOC)
def population_method(**argspec_transforms):
  '''Create a method in the Population class to encapsulate this function.

Fill in documentation holes, but otherwise leave the function itself
untouched for library use.

For the types below, "arg" is a zero-indexed positional argument number
or a keyword argument name as a string. For keyword arguments, if the
caller provides a value, we leave it untouched. Positional arguments may
either be kept and transformed, or hidden from the argument list. If
the transform takes both a population and an arg, then it is kept and the
arg is transformed. If it takes only a population, then it is hidden (there
is no corresponding positional argument in the method, the documentation
line is removed, and other positional arguments shift to accommodate).

__PT_DOC__

In your method's docstring, use the above names with underscores to
get their appropriate documentation (if you want it replaced). For
example, use __population_name__ in place of the lines where you
would put the population name parameter's description, after its name.

For those transformations that do not have an [arg], if their documentation
line starts with the parameter name, then a colon, then the double
underscored transformation name, then the entire line is removed.
For example, a documentation line like
  table : __population_name__
would be entirely removed from the method docstring, because that argument
is filled in using the population instance, rather than being requested
from the caller.
     '''
  def decorator(fn):
    xfrms = compile_argspec_transforms(fn, argspec_transforms)
    def as_population_method(self, *args, **kwargs):
      with logged_query(query_string=fn.__code__.co_name,
                        bindings=(args, kwargs),
                        name=self.session_capture_name):
        self.check_representation()
        (dargs, dkwargs) = apply_argspec_transforms(self, xfrms, args, kwargs)
        result = None
        try:
          result = fn(*dargs, **dkwargs)
        except:
          self.logger.exception("")
          raise
        self.check_representation()
        return result

    (doc, shortdoc) = redocument(fn, xfrms, METHOD_DOC_FILLERS)
    as_population_method.__doc__ = doc
    Population.shortdocs.append(shortdoc)
    as_population_method.__name__ = xfrms['name']
    setattr(Population, fn.__code__.co_name, as_population_method)

    fn.__doc__ = fill_documentation(fn.__doc__, DECORATED_DOC_FILLERS)
    return fn

  return decorator
