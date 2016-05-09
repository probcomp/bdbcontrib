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

import pandas
import pytest
import re
from bdbcontrib import quickstart
from bdbcontrib import population_method as pm

# pylint: disable=no-member

def test_fill_documentation():
    my_fillers = {'__foo__': 'bar', '__baz__': None}
    docstr = '''Blah blah.

    Nevermind.

    Args:
    foo: __foo__
    blux: a stoats.Mink
    baz: __baz__
    fifi: optional.

    Returns loveliness.
    '''
    result = pm.fill_documentation(docstr, my_fillers)
    import re
    assert re.search(r'^\s+foo: bar$', result, re.M), result
    assert not re.search(r'baz', result), result
    assert re.search(r'Mink\n\s+fifi', result, re.M), result

def test_compile_argspec_transforms():
    def curie(fname, when, where, field1='Physics', field2='Chemistry'):
        pass
    xforms = pm.compile_argspec_transforms(curie, {})
    assert ['fname', 'when', 'where'] == xforms['required_names']
    assert 3 == len(xforms['required_transformers'])
    for i, xform in enumerate(xforms['required_transformers']):
        assert i+1 == xform(None, [1, 2, 3])
    assert 2 == len(xforms['optional_transformers'])

    xforms = pm.compile_argspec_transforms(
        curie, {'population': 0, 'generator_name': ['field1', 'field2']})
    assert ['when', 'where'] == xforms['required_names']
    dummy_self = lambda: None  # Can set attrs on this, unlike an object().
    assert 3 == len(xforms['required_transformers'])
    assert dummy_self == xforms['required_transformers'][0](dummy_self, [])
    assert 'boo' == xforms['required_transformers'][1](dummy_self, ['boo'])
    assert 'bee' == xforms['required_transformers'][2](dummy_self, [1, 'bee'])
    assert 2 == len(xforms['optional_transformers'])
    assert 'good' == xforms['optional_transformers']['field1'](
        dummy_self, {'field1': 'good'})
    dummy_self.generator_name = 'great'
    assert 'great' == xforms['optional_transformers']['field2'](
        dummy_self, {'field1': 'bad'})

    with pytest.raises(IndexError):
        xforms = pm.compile_argspec_transforms(
            curie, {'population': -1, 'generator_name': ['field1', 'field2']})

    with pytest.raises(IndexError):
        xforms = pm.compile_argspec_transforms(
            curie, {'population': 1, 'population_to_bdb': 1})

    with pytest.raises(IndexError):
        xforms = pm.compile_argspec_transforms(
            curie, {'population': 'field1', 'population_to_bdb': ['field1']})

    with pytest.raises(IndexError):
        xforms = pm.compile_argspec_transforms(
            curie, {'population': 'no_such_option'})

    with pytest.raises(IndexError):
        xforms = pm.compile_argspec_transforms(
            curie, {'population': 1, 'population_to_bdb': 1})

def test_apply_argspec_transforms():
    def curie(fname, when, where, field1='Physics', field2='Chemistry'):
        pass
    xforms = pm.compile_argspec_transforms(
        curie, {'population': 0, 'generator_name': ['field1', 'field2']})
    assert ['when', 'where'] == xforms['required_names']
    log = lambda: None
    log.generator_name = 'genny'
    (dargs, dkwargs) = pm.apply_argspec_transforms(
        log, xforms, ('one', 'two'), {'field1': 'zip!'})
    assert [log, 'one', 'two', 'zip!', 'genny'] == dargs
    assert {} == dkwargs  # Only using dkwargs for varkw at this point.

    (dargs, dkwargs) = pm.apply_argspec_transforms(
        log, xforms, ('when', 'where', 'f_one', 'f_two'), {})
    assert [log, 'when', 'where', 'f_one', 'f_two'] == dargs
    assert {} == dkwargs

@pm.population_method(specifier_to_df=[1], population_to_bdb=0,
                      generator_name='gen')
def example_population_method(bdb, df1, df2, gen=None):
    '''A silly population method.

    bdb: __population_to_bdb__
    df1: __specifier_to_df__
        Is the first one.
    df2: __specifier_to_df__
        Is the second one.
    gen: It's okay not to use the doc substitution.
    '''
    return str(len(df1)) + str(len(df2)) + gen

@pm.population_method()
def minimal_population_method():
    # It's fine to not even have a docstring, and not to use any of the
    # transformations
    return 42

def test_method_calls():
    pop = quickstart('foo', df=pandas.DataFrame({'a': [11, 22]}),
                     session_capture_name='test_population.py')
    assert 42 == pop.minimal_population_method()
    # It's ok to name or not name your positional args.
    # (But it's a syntax error to have a positional arg after a named one.)
    assert '12foo_cc', pop.example_population_method(
        df1='select * limit 1', df2=pandas.DataFrame({'b': [23, 34]}))
    assert '12foo_cc', pop.example_population_method(
        'select * limit 1', df2=pandas.DataFrame({'b': [23, 34]}))
    assert '12foo_cc', pop.example_population_method(
        'select * limit 1', pandas.DataFrame({'b': [23, 34]}))
    # It's okay to name them and get them in the wrong order too.
    assert '12foo_cc', pop.example_population_method(
        df2=pandas.DataFrame({'b': [23, 34]}), df1='select * limit 1')

    # The bdb argument should be present and explained in the function, and
    # should be absent in the method, where it's implicit.
    as_method = pop.example_population_method.__doc__
    as_function = example_population_method.__doc__
    assert not re.search(r'bdb', as_method), as_method
    assert re.search('bdb', as_function), as_function

    with pytest.raises(TypeError) as exc:
        pop.minimal_population_method(3)
    assert ('test_population_method.minimal_population_method()'
            ' takes no arguments (1 given)' == str(exc.value)), repr(exc)

    epm="test_population_method.example_population_method()"
    with pytest.raises(TypeError) as exc:
        pop.example_population_method([5])
    assert (epm + " takes at least 2 arguments (1 given)"
            == str(exc.value)), repr(exc)

    with pytest.raises(TypeError) as exc:
        pop.example_population_method([5], gen=True)
    # This message is among Python's uglier warts, but I want to be consistent.
    assert (epm + " takes at least 2 arguments (2 given)"
            == str(exc.value)), repr(exc)

    with pytest.raises(TypeError) as exc:
        pop.example_population_method([1], [2], [3], [4])
    assert (epm + " takes at most 3 arguments (4 given)"
            == str(exc.value)), repr(exc)

    with pytest.raises(TypeError) as exc:
        pop.example_population_method("SELECT * FROM %t", "SELECT * FROM %g",
                                      b="bang")
    assert (epm + " got an unexpected keyword argument 'b'"
            == str(exc.value)), repr(exc)

    with pytest.raises(TypeError) as exc:
        pop.example_population_method(
            "SELECT * FROM %t", df1="SELECT * FROM %g")
    # This is another misleading message, because the only way for it to happen
    # is with a required positional argument that happens to be named at the
    # call site. A caller might, in this case, expect keywords to be interpreted
    # first, for example, and positional arguments consumed in order thereafter.
    # Not to say that that would be a better choice. I would simply prefer "for
    # 2nd argument df1" rather than "for keyword argument df1".
    # Again, I find it more important to be consistent than to be right.
    assert (epm + " got multiple values for keyword argument 'df1'"
            == str(exc.value)), repr(exc)
    # Outright repetition, like "gen=1, gen=2", is a syntax error.


@pm.population_method(population=0)
def five_defaults(pop, a=1, b=2, c=3, d=4, e=5):
    return [a, b, c, d, e]

def test_fn_defaults():
    pop = quickstart('foo', df=pandas.DataFrame({'a': [11, 22]}),
              session_capture_name='test_population.py')
    assert [1, 2, 3, 4, 5] == pop.five_defaults()
    assert [7, 2, 3, 4, 5] == pop.five_defaults(a=7)
    assert [1, 7, 3, 4, 5] == pop.five_defaults(b=7)
    assert [1, 2, 7, 4, 5] == pop.five_defaults(c=7)
    assert [1, 2, 3, 7, 5] == pop.five_defaults(d=7)
    assert [1, 2, 3, 4, 7] == pop.five_defaults(e=7)
    assert [7, 2, 8, 4, 9] == pop.five_defaults(a=7, c=8, e=9)
    assert [1, 7, 3, 8, 5] == pop.five_defaults(b=7, d=8)
    assert [1, 7, 8, 9, 5] == pop.five_defaults(b=7, c=8, d=9)

@pm.population_method(population_name=[0])
def hasvarargs(pname, _u, pop=None, *args):
    return (pname, pop, len(args))

@pm.population_method(population_name=[1, 'pop'])
def hasvarkw(_u, pname, pop=None, **kwargs):
    return (pname, pop, len(kwargs))

@pm.population_method(population_name=1)
def hasboth_arg(_u, pname, _v, pop=None, *args, **kwargs):
    return (pname, pop, len(args), len(kwargs))
@pm.population_method(population_name=[1, 'pop'])
def hasboth_argkwarg(_u, pname, _v, pop=None, *args, **kwargs):
    return (pname, pop, len(args), len(kwargs))

@pm.population_method(population_name=[1])
def hasboth_haspop(_u, pname, _v, pop='fizz', *args, **kwargs):
    return (pname, pop, len(args), len(kwargs))

def test_variable_arglengths():
    pop = quickstart('foo', df=pandas.DataFrame({'a': [11, 22]}),
                     session_capture_name='test_population.py')
    spec = pm.compile_argspec_transforms(hasvarargs, {'population_name': 0})
    assert ['_u'] == spec['required_names']
    assert 2 == len(spec['required_transformers'])
    assert ['pop'] == spec['optional_names']
    assert 1 == len(spec['optional_transformers'])
    xfm = spec['optional_transformers']['pop']
    assert None == xfm(pop, {})
    assert 7 == xfm(pop, {'pop': 7})
    assert None == xfm(pop, {'zip': 7})
    assert 'args' == spec['varargs']
    assert None == spec['varkw']

    assert ('foo', 1, 3) == pop.hasvarargs("U", 1, 2, 3, 4)
    assert ('foo', 'pip', 3) == pop.hasvarkw("U", pop='pip', a=1, b=2, c=3)
    assert ('foo', None, 0, 0) == pop.hasboth_arg('U', 'V')
    assert ('foo', 'W', 1, 2) == pop.hasboth_arg("U", "V", "W", "X",
                                                 y="Y", z="Z")
    assert ('foo', 'foo', 0, 0) == pop.hasboth_argkwarg('U', 'V')
    assert ('foo', 'W', 1, 2) == pop.hasboth_argkwarg("U", "V", "W", "X",
                                                      y="Y", z="Z")
    assert ('foo', 'fizz', 0, 0) == pop.hasboth_haspop('U', 'V')

    with pytest.raises(TypeError) as exc:
        pop.hasvarargs("U", 1, 2, 3, 4, pop='pip')
    assert "multiple values for keyword argument 'pop'" in str(exc.value)
    with pytest.raises(TypeError):
        pop.hasvarargs("U", 1, 2, a=1, b=2)
    with pytest.raises(TypeError):
        pop.hasvarkw("U", 1, 2, 3, 4, 5, 6, 7)
    with pytest.raises(TypeError):
        pop.hasboth_arg()
    with pytest.raises(TypeError):
        pop.hasboth_argkwarg()

    # Can only fill named parameters with population-implicit values.
    with pytest.raises(IndexError):
        @pm.population_method(population_name='0')
        def cannot_fill_unnamed_kwargs(*args, **kwargs):
            pass
    with pytest.raises(IndexError):
        @pm.population_method(population_name='pop')
        def cannot_fill_unnamed_args(*args, **kwargs):
            pass
