import pandas as pd
from sqlalchemy import create_engine
import argparse
import re
import os


class sql_query:

    # class to make SQL request for an annotation
    # TO DO: add annotation and segment? emg6+emg6- par ex

    def __init__(self, credentials_path):
        self.db_credentials = pd.read_csv(credentials_path, index_col="Field")

    def __call__(self,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 text: str) -> pd.DataFrame:

        engine = create_engine(
            "mysql+pymysql://{user}:{pw}@localhost/{db}".format(
                user=self.db_credentials.loc["user"][0],
                pw=self.db_credentials.loc["password"][0],
                db="grafana"))

        # TO MODIFY WITH NEW EDF FILE
        start_date = round(start_date.replace(month=12).value/1_000_000, 0)
        end_date = round(end_date.replace(month=12).value/1_000_000, 0)
        df = pd.read_sql(
            'SELECT * FROM annotation_restitution ' +
            f'WHERE epoch >= {start_date} ' +
            f'AND epoch_end < {end_date} ' +
            f'AND text = "{text}";',
            engine)

        return df


def generation_annot_list(df: pd.DataFrame,
                          start_date: pd.Timestamp,
                          end_date: pd.Timestamp) -> list:

    # Create list to compare annotations

    hz = 256
    duration = end_date - start_date
    point_count = int(round(duration / (pd.Timedelta(seconds=1/hz)), 0))

    list_classif = ['0'] * point_count

    for i in range(df.shape[0]):

        start_index = int(round((pd.to_datetime(
            df['epoch'].iloc[i]/1_000, unit='s')  # .replace(month=11)
            - start_date).value
            / (1_000_000 * 1_000 / hz), 0))
        end_index = int(round((
            pd.to_datetime(
                df['epoch_end'].iloc[i]/1_000, unit='s')  # .replace(month=11)
            - start_date).value/(1_000_000 * 1_000 / hz),
            0)) + 1

        list_classif[start_index:end_index] = ['1'] * (end_index - start_index)

    return list_classif


def make_result_df(ids: str,
                   record: str,
                   channel: str,
                   start_date: str,
                   end_date: str):

    ids = [int(annotator) for annotator in re.split(',', ids)]

    # Compute for annotators the concordance
    if "AIRFLOW_HOME" in os.environ:
        query = sql_query(f'{os.environ["AIRFLOW_HOME"]}/credentials.csv')

    else:
        query = sql_query('credentials.csv')

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    text = channel + ' ' + record
    df = query(start_date, end_date, text)

    df = df.loc[:, ['user_id', 'epoch', 'epoch_end']]

    dfs = [df[df['user_id'] == id] for _, id in enumerate(ids)]

    list_ids = [generation_annot_list(df, start_date, end_date)
                for _, df in enumerate(dfs)]

    freq_ns = int(1/256*1_000_000_000)
    results_df = pd.DataFrame(None,
                              columns=ids,
                              index=pd.date_range(start_date,
                                                  freq=f'{freq_ns}ns',
                                                  periods=len(list_ids[0]))
                              )

    for i, list_id in enumerate(list_ids):
        results_df[ids[i]] = list_id

    results_df = results_df.astype(int)

    return results_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("-p", "--patient", dest="patient",
                        help="patient to load", metavar="FILE")
    parser.add_argument("-r", "--record", dest="record",
                        help="record to load", metavar="FILE")
    parser.add_argument("-c", "--channel", dest="channel",
                        help="channel to load", metavar="FILE")
    parser.add_argument("-st", "--start_time", dest="start_time",
                        help="start time for filter", metavar="FILE")
    parser.add_argument("-et", "--end_time", dest="end_time",
                        help="end time for filter", metavar="FILE")
    parser.add_argument("-ids", "--annot_ids", dest="annot_ids",
                        help="ids of annotators", metavar="FILE")
    parser.add_argument("-o", "--output_folder", dest="output_folder",
                        help="output_folder_for_df", metavar="FILE",
                        default="./exports")

    args = parser.parse_args()

    df = make_result_df(ids=args.annot_ids,
                        record=args.record,
                        channel=args.channel,
                        start_date=args.start_time,
                        end_date=args.end_time)

    df.to_csv(args.output_folder +
              f'/annot_{args.patient}_{args.record}_{args.channel}.csv',
              index=False)
