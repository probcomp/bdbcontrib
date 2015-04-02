
import pytest
from bdbcontrib import facade
import tempfile
import pandas
from bdbcontrib import crosscat_utils


@pytest.fixture
def bdb_file(request):
    f = tempfile.NamedTemporaryFile()

    def fin():
        f.close()

    request.addfinalizer(fin)
    return f


def get_test_df():
    PANDAS_DF_DATA = [
        {
            'age': 34,
            'gender': 'M',
            'salary': 7400,
            'height': 65,
            'division': 'sales',
            'rank': 3
            },
        {
            'age': 41,
            'gender': 'M',
            'salary': 6560,
            'height': 72,
            'division': 'marketing',
            'rank': 4
            },
        {
            'age': 25,
            'gender': 'M',
            'salary': 5200,
            'height': 69,
            'division':
            'accounting',
            'rank': 5
            },
        {
            'age': 23,
            'gender': 'F',
            'salary': 8100,
            'height': 67,
            'division':
            'data science',
            'rank': 3
            },
        {
            'age': 36,
            'gender': 'F',
            'salary': 9600,
            'height': 70,
            'division': 'management',
            'rank': 2
            },
        {
            'age': 30,
            'gender': 'M',
            'salary': 7000,
            'height': 73,
            'division': 'sales',
            'rank': 4
            },
        {
            'age': 30,
            'gender': 'F',
            'salary': 8100,
            'height': 73,
            'division': 'engineering',
            'rank': 3
            },
    ]
    return pandas.DataFrame(PANDAS_DF_DATA)


def test_get_metadata(bdb_file):
    table_name = 'tmp_table'
    generator_name = 'tmp_cc'
    pandas_df = get_test_df()

    client = facade.BayesDBClient.from_pandas(bdb_file.name, table_name, pandas_df,
                                              generator_name=generator_name)
    with pytest.raises(ValueError):
        md = crosscat_utils.get_metadata(client.bdb, generator_name, 0)

    client('INITIALIZE 2 MODELS FOR {}'.format(generator_name))

    with pytest.raises(ValueError):
        crosscat_utils.get_metadata(client.bdb, 'Peter_Gabriel', 0)
    md = crosscat_utils.get_metadata(client.bdb, generator_name, 0)

    assert isinstance(md, dict)
    assert 'X_D' in md.keys()
    assert 'X_L' in md.keys()
