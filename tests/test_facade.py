import time
import pytest
from bdbcontrib import facade
import tempfile
import pandas
import os


CSV_DATA = '''age,gender,salary,height,division,rank
34,M,74000,65,sales,3
41,M,65600,72,marketing,4
25,M,52000,69,accounting,5
23,F,81000,67,data science,3
36,F,96000,70,management,2
30,M,70000,73,sales,4
30,F,81000,73,engineering,3
'''


CODEBOOK_DATA = '''column_label,short_name,description,value_map
age,age,age in people years,NaN
gender,gender,not to be confused with sex,NaN
salary,"salary, dh",yearly salary in doll hairs,NaN
division,division,division,NaN
height,"height, in",height in inches,NaN
'''


@pytest.fixture
def csv_data(request):
    test_codebook_filename = 'test_codebook.csv'
    test_csv_filename = 'test.csv'
    with open(test_csv_filename, 'w') as f:
        f.write(CSV_DATA)

    with open(test_codebook_filename, 'w') as f:
        f.write(CODEBOOK_DATA)

    def fin():
        os.remove(test_csv_filename)
        os.remove(test_codebook_filename)

    request.addfinalizer(fin)
    return test_csv_filename, test_codebook_filename


@pytest.fixture
def bdb_file(request):
    f = tempfile.NamedTemporaryFile()

    def fin():
        f.close()

    request.addfinalizer(fin)
    return f


def get_test_df():
    PANDAS_DF_DATA = [
        {'age': 34, 'gender': 'M', 'salary': 7400, 'height': 65, 'division': 'sales', 'rank': 3},
        {'age': 41, 'gender': 'M', 'salary': 6560, 'height': 72, 'division': 'marketing', 'rank': 4},
        {'age': 25, 'gender': 'M', 'salary': 5200, 'height': 69, 'division': 'accounting', 'rank': 5},
        {'age': 23, 'gender': 'F', 'salary': 8100, 'height': 67, 'division': 'data science', 'rank': 3},
        {'age': 36, 'gender': 'F', 'salary': 9600, 'height': 70, 'division': 'management', 'rank': 2},
        {'age': 30, 'gender': 'M', 'salary': 7000, 'height': 73, 'division': 'sales', 'rank': 4},
        {'age': 30, 'gender': 'F', 'salary': 8100, 'height': 73, 'division': 'engineering', 'rank': 3},
    ]
    return pandas.DataFrame(PANDAS_DF_DATA)


# ``````````````````````````````````````````````````````````````````````````````````````````````````

@pytest.mark.parametrize("no_mp", [True, False])
def test_BayesDBClient_init_crash(bdb_file, no_mp):
    facade.BayesDBClient(bdb_file.name, no_mp=no_mp)


@pytest.mark.parametrize("no_mp", [True, False])
def test_BayesDBClient_init_csv_crash(bdb_file, csv_data, no_mp):
    table_name = 'tmp_table'
    f_csv, _ = csv_data
    facade.BayesDBClient.from_csv(bdb_file.name, table_name, f_csv, no_mp=no_mp)


@pytest.mark.parametrize("no_mp", [True, False])
def test_BayesDBClient_init_csv_with_codebook_crash(bdb_file, csv_data, no_mp):
    table_name = 'tmp_table'
    f_csv, f_codebook = csv_data
    facade.BayesDBClient.from_csv(bdb_file.name, table_name, f_csv,
                                    codebook_filename=f_codebook, no_mp=no_mp)


@pytest.mark.parametrize("no_mp", [True, False])
def test_BayesDBClient_init_pandas_with_codebook_crash(bdb_file, csv_data, no_mp):
    table_name = 'tmp_table'
    pandas_df = get_test_df()
    f_csv, f_codebook = csv_data
    facade.BayesDBClient.from_pandas(bdb_file.name, table_name, pandas_df,
                                       codebook_filename=f_codebook, no_mp=no_mp)


@pytest.mark.parametrize("no_mp", [True, False])
def test_BayesDBClient_init_pandas_crash(bdb_file, no_mp):
    table_name = 'tmp_table'
    pandas_df = get_test_df()

    facade.BayesDBClient.from_pandas(bdb_file.name, table_name, pandas_df, no_mp=no_mp)


# make sure that the to_df method is handling numeric values
@pytest.mark.parametrize("no_mp", [True, False])
def test_BayesDBClient_query_numeric_output(bdb_file, no_mp):
    table_name = 'tmp_table'
    pandas_df = get_test_df()

    client = facade.BayesDBClient.from_pandas(bdb_file.name, table_name, pandas_df, no_mp=no_mp)
    res = client.query('SELECT age FROM {} WHERE division = accounting'.format(table_name))
    assert isinstance(res, facade.BQLQueryResult)
    assert res.__dict__['_df'] is None

    res_df = res.as_df()

    assert isinstance(res_df, pandas.DataFrame)
    assert res_df.shape[0] == 1
    assert res_df['age'].values[0] == 25


# make sure that to_df is handling string values
@pytest.mark.parametrize("no_mp", [True, False])
def test_BayesDBClient_query_string_output(bdb_file, no_mp):
    table_name = 'tmp_table'
    pandas_df = get_test_df()

    client = facade.BayesDBClient.from_pandas(bdb_file.name, table_name, pandas_df, no_mp=no_mp)
    res = client.query('SELECT gender FROM {} WHERE height = 73 ORDER BY RANK DESC'.format(table_name))
    assert isinstance(res, facade.BQLQueryResult)
    assert res.__dict__['_df'] is None

    res_df = res.as_df()
    assert isinstance(res_df, pandas.DataFrame)
    assert res_df.shape[0] == 2
    assert res_df['gender'].values[0] == 'M'
