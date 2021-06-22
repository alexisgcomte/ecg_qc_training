import pytest
import pandas as pd
from dags.tasks.import_annotations import SqlQuery,\
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
def test_SqlQuery_init_cred(credentials_path='tests/test_credentials.csv'):
    return SqlQuery(credentials_path)


def test_SqlQuery_init(test_SqlQuery_init_cred):
    assert test_SqlQuery_init_cred.user == 'user'
    assert test_SqlQuery_init_cred.pw == 'password'
    assert test_SqlQuery_init_cred.host == 'localhost'
    assert test_SqlQuery_init_cred.db == 'grafana'


@pytest.fixture()
def test_SqlQuery(credentials_path='credentials.csv'):
    return SqlQuery(credentials_path)


def test_SqlQuery_call(test_SqlQuery):

    engine = create_engine(f'mysql+pymysql://'
                           f'{test_SqlQuery.user}:{test_SqlQuery.pw}'
                           f'@localhost/{test_SqlQuery.db}')
    assert engine is not None


@pytest.fixture()
def test_SqlQuery_df(credentials_path='credentials.csv',
                      start_date=start_date,
                      end_date=end_date,
                      text=text):

    query = SqlQuery(credentials_path)
    df_sql = query(start_date=start_date,
                   end_date=end_date,
                   text=text)
    return df_sql


def test_df_sql_type(test_SqlQuery_df):
    assert isinstance(test_SqlQuery_df, pd.DataFrame)
    assert test_SqlQuery_df.shape[0] > 0


@pytest.fixture()
def test_df_users(test_SqlQuery_df,
                  user_id=2):
    return test_SqlQuery_df[test_SqlQuery_df['user_id'] == user_id]


def test_generation_annot_list(
        test_df_users,
        start_date=start_date,
        end_date=end_date,
        sampling_frequency_hz=sampling_frequency_hz):

    assert isinstance(test_df_users, pd.DataFrame)
    assert test_df_users.shape[0] > 0

    list_classif = generation_annot_list(
        df_user=test_df_users,
        start_date=start_date,
        end_date=end_date,
        sampling_frequency_hz=sampling_frequency_hz)

    # "Normality" of the list
    assert isinstance(list_classif, list)
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

    assert isinstance(df_annot, pd.DataFrame)
    assert df_annot.shape[0] > 0
    assert df_annot.shape[1] == 4
