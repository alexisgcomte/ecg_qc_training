"""import_annotations script

This script connects to SQL database to request annotators noise
classifications, wrangle data into proper DataFrame format and export
it to CSV.

This file can also be imported as a module and contains the following
functions:

    * SqlQuery - makes noise annotation SQL request
    * generation_annot_list - transform noise information to a list with same
    sampling rate as ECG
    * make_annot_df - use SqlQuery and generation_annot_list to create a
    DataFrame of annotations by annotator with same sampling rate as ECG
    * main - the main function of the script

"""

import pandas as pd
from sqlalchemy import create_engine
import argparse
import re
import os


class SqlQuery:
    """
    A class to make SQL request of all noise annotations for a channel and
    a record

    """

    def __init__(self, credentials_path: str):
        """__init__

        Parameters
        ----------
        credentials_path : str
            Path to load credentials to connect to SQL database. The CSV must
            include for columns Field,Value : user, password, host

        """
        self.db_credentials = pd.read_csv(credentials_path,
                                          index_col='Field',
                                          sep=',')
        self.user = self.db_credentials.loc['user'][0]
        self.pw = self.db_credentials.loc['password'][0]
        self.host = self.db_credentials.loc['host'][0]
        self.db = self.db_credentials.loc['db'][0]

    def __call__(self,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 text: str) -> pd.DataFrame:
        """Make an SQL request to get all availiable annotations within
        start and end dates filtered by a channel and a record

        Parameters
        ----------
        start_date : pd.Timestamp
            Start of the ECG signal to filter
        end_date : pd.Timestamp
            Start of the ECG signal to filter
        text : str
            Field "text" of the SQL table to filter. Combines channel and
            record with formatting 'channel record'

        Returns
        -------
        df_sql : pd.DataFrame
            DataFrame with noise annotations by user_id with start and end
            time of noise event

        """

        engine = create_engine(
            f'mysql+pymysql://{self.user}:{self.pw}@{self.host}/{self.db}')

        # Annotation data is loaded with a difference of one month
        start_date = round(start_date.replace(month=12).value/1_000_000)
        end_date = round(end_date.replace(month=12).value/1_000_000)
        df_sql = pd.read_sql(
            f'SELECT user_id, epoch, epoch_end, text '
            f'FROM  annotation_restitution '
            f'WHERE epoch >= {start_date} '
            f'AND epoch_end < {end_date} '
            f'AND text = "{text}";',
            engine)

        return df_sql


def generation_annot_list(df_user: pd.DataFrame,
                          start_date: pd.Timestamp,
                          end_date: pd.Timestamp,
                          sampling_frequency_hz: int = 256) -> list:
    """From a DataFrame with an annotator noise input, creates a list of
    boolean noise classfication (1 = good quality, 0 = noise) with the targeted
    frequency to match ECG sampling rate

    Parameters
    ----------
    df_user : pd.DataFrame
        DataFrame with annotations of noise for an annotator, with columns
        'user_id', 'epoch' for start of noise annotation, 'epoch_end' for end
        of noise annotation
    start_date : pd.Timestamp
        Start of the ECG signal to filter
    end_date : pd.Timestamp
        Start of the ECG signal to filter
    sampling_frequency_hz : int
        The sampling frequency of the ECG signal to match

    Returns
    -------
    list_classif : list
        List with same sampling frequency as ECG with boolean indication of
        noise (1 = good quality, 0 = noise)

    """
    duration = end_date - start_date
    point_count = round(
        duration / pd.Timedelta(seconds=1/sampling_frequency_hz))
    # List to update afterward with default value  1 = good quality
    list_classif = ['1'] * point_count

    # Updating noise classifications
    for i in range(df_user.shape[0]):

        start_index = int(round((pd.to_datetime(
            df_user['epoch'].iloc[i]/1_000, unit='s')
            - start_date).value
            / (1_000_000 * 1_000 / sampling_frequency_hz), 0))
        end_index = int(round((
            pd.to_datetime(
                df_user['epoch_end'].iloc[i]/1_000, unit='s')
            - start_date).value/(1_000_000 * 1_000 / sampling_frequency_hz),
            0)) + 1

        list_classif[start_index:end_index] = ['0'] * (end_index - start_index)

    return list_classif


def make_annot_df(ids: str,
                  record: str,
                  channel: str,
                  start_date: str,
                  end_date: str,
                  sampling_frequency_hz: int = 256) -> pd.DataFrame:
    """Creates annotations DataFrame through SQL request

    Parameters
    ----------
    ids : str
        Annotators ids to request
    record : str
        ECG record to request
    channel : str
        ECG channel to request
    start_date : str
        Start of the recording to request, format YYYY-MM-DD HH:MM:SS
    end_date : str
        End of the recording to request, format YYYY-MM-DD HH:MM:SS
    sampling_frequency_hz : int
        The sampling frequency of the ECG signal

    Returns
    -------
    df_annot : pd.DataFrame
        DataFrame with annotators booleans quality classification

    """

    # Parsing to right format
    ids = [int(annotator) for annotator in re.split(',', ids)]
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    text = f'{channel} {record}'

    # SQL request
    if 'AIRFLOW_HOME' in os.environ:
        query = SqlQuery(f'{os.environ["AIRFLOW_HOME"]}/credentials.csv')
    else:
        query = SqlQuery('credentials.csv')
    df_sql = query(start_date=start_date,
                   end_date=end_date,
                   text=text)

    # Transforming user_id column in seperate user_id columns
    dfs_users = [df_sql[df_sql['user_id'] == id] for _, id in enumerate(ids)]

    list_ids = [generation_annot_list(
        df_user=df_user,
        start_date=start_date,
        end_date=end_date,
        sampling_frequency_hz=sampling_frequency_hz)
                for _, df_user in enumerate(dfs_users)]

    freq_ns = int(1_000_000_000/sampling_frequency_hz)
    df_annot = pd.DataFrame(None,
                            columns=ids,
                            index=pd.date_range(start_date,
                                                freq=f'{freq_ns}ns',
                                                periods=len(list_ids[0]))
                            )

    for i, list_id in enumerate(list_ids):
        df_annot[ids[i]] = list_id
    df_annot = df_annot.astype(int)
    df_annot.columns = [f'annot_{id}' for id in df_annot.columns]

    df_annot['record'] = df_sql['text'].iloc[0]

    return df_annot


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument('-p',
                        '--patient',
                        dest='patient',
                        help='patient to load',
                        metavar='FILE')
    parser.add_argument('-r',
                        '--record',
                        dest='record',
                        help='record to load',
                        metavar='FILE')
    parser.add_argument('-ch',
                        '--channel',
                        dest='channel',
                        help='channel to load',
                        metavar='FILE')
    parser.add_argument('-st',
                        '--start_time',
                        dest='start_time',
                        help='start time for filter',
                        metavar='FILE')
    parser.add_argument('-et',
                        '--end_time',
                        dest='end_time',
                        help='end time for filter',
                        metavar='FILE')
    parser.add_argument('-ids',
                        '--annot_ids',
                        dest='annot_ids',
                        help='ids of annotators',
                        metavar='FILE')
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        help='output_folder_for_df',
                        metavar='FILE',
                        default='./exports')
    parser.add_argument('-s', '--sampling_frequency_hz',
                        dest='sampling_frequency_hz',
                        help='sampling_frequency_hz_of_file',
                        metavar='FILE',
                        default='256')
    args = parser.parse_args()

    df_annot = make_annot_df(ids=args.annot_ids,
                             record=args.record,
                             channel=args.channel,
                             start_date=args.start_time,
                             end_date=args.end_time,
                             sampling_frequency_hz=int(
                                 args.sampling_frequency_hz))

    df_annot.to_csv(f'{args.output_folder}/'
                    f'annot_{args.patient}_{args.record}_{args.channel}.csv',
                    index=False)
