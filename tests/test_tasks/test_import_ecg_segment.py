import pytest
import pandas as pd
from dags.tasks.import_annotations import sql_query,\
                                          generation_annot_list,\
                                          make_annot_df
from sqlalchemy import create_engine

patient = 'PAT_6'
ids = '2,3,4'
start_date = pd.Timestamp('2020-12-18 13:00:00')
end_date = pd.Timestamp('2020-12-18 14:30:00')
channel = 'emg6+emg6-'
record = 77
text = text = f'{channel} {record}'
sampling_frequency_hz = 256


@pytest.fixture()
def test_sql_query_init_cred(credentials_path='tests/test_credentials.csv'):
    return sql_query(credentials_path)


def test_sql_query_init(test_sql_query_init_cred):
    assert test_sql_query_init_cred.user == 'user'
    assert test_sql_query_init_cred.pw == 'password'
    assert test_sql_query_init_cred.host == 'localhost'
    assert test_sql_query_init_cred.db == 'grafana'


@pytest.fixture()
def test_sql_query(credentials_path='credentials.csv'):
    return sql_query(credentials_path)


def test_sql_query_call(test_sql_query):

    engine = create_engine(f'mysql+pymysql://'
                           f'{test_sql_query.user}:{test_sql_query.pw}'
                           f'@localhost/{test_sql_query.db}')
    assert engine is not None


@pytest.fixture()
def test_sql_query_df(credentials_path='credentials.csv',
                      start_date=start_date,
                      end_date=end_date,
                      text=text):

    query = sql_query(credentials_path)
    df_sql = query(start_date=start_date,
                   end_date=end_date,
                   text=text)
    return df_sql


def test_df_sql_type(test_sql_query_df):
    assert type(test_sql_query_df) is pd.DataFrame
    assert test_sql_query_df.shape[0] > 0


@pytest.fixture()
def test_df_users(test_sql_query_df,
                  user_id=2):
    return test_sql_query_df[test_sql_query_df['user_id'] == user_id]


def test_generation_annot_list(
        test_df_users,
        start_date=start_date,
        end_date=end_date,
        sampling_frequency_hz=sampling_frequency_hz):

    assert type(test_df_users) is pd.DataFrame
    assert test_df_users.shape[0] > 0

    list_classif = generation_annot_list(
        df_user=test_df_users,
        start_date=start_date,
        end_date=end_date,
        sampling_frequency_hz=sampling_frequency_hz)

    # "Normality" of the list
    assert type(list_classif) is list
    assert '1' in list_classif
    assert '0' in list_classif

    # Length of segment
    duration = end_date - start_date
    assert len(list_classif) == round(
        duration / pd.Timedelta(seconds=1/sampling_frequency_hz))


def test_make_annot_df(ids=ids,
                       record=record,
                       channel=channel,
                       start_date=start_date,
                       end_date=end_date,
                       sampling_frequency_hz=sampling_frequency_hz):

    df_annot = make_annot_df(ids=ids,
                             record=record,
                             channel=channel,
                             start_date=start_date,
                             end_date=end_date,
                             sampling_frequency_hz=sampling_frequency_hz)

    assert type(df_annot) is pd.DataFrame
    assert df_annot.shape[0] > 0
    assert df_annot.shape[1] == 4
