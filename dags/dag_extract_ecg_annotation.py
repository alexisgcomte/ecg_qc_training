"""dag_extract_ecg_annotation DAG

This script is meant to be used as a dag. According to parameters loaded in
signals_to_process.txt, will combine signal data with annotator annotations.
"""

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from tasks.import_annotations import make_annot_df
from tasks.import_ecg_segment import EdfLoader
import pandas as pd
import json
import os


# Parameters
folder = os.environ['AIRFLOW_HOME']
output_folder = f'{folder}/exports'

default_args = {'owner': 'airflow',
                'depends_on_past': False,
                'retries': 0}

# Signals to reuqest
with open(f'{folder}/dags/signals_to_process.txt') as json_file:
    param_json = json.load(json_file)

parameters = [[param_json[signal]['patient'],
               param_json[signal]['record'],
               param_json[signal]['segment'],
               param_json[signal]['channel'],
               param_json[signal]['ids'],
               param_json[signal]['start_time'],
               param_json[signal]['end_time']
               ] for _, signal in enumerate(param_json)]


@dag(default_args=default_args,
     schedule_interval=None,
     start_date=days_ago(1),
     tags=['ecg_qc', 'extraction'])
def dag_extract_ecg_annotation():

    @task(depends_on_past=False)
    def extract_signal_parameters(index: str):

        # With signals to loads
        with open(folder + '/dags/signals_to_process.txt') as json_file:
            param_json = json.load(json_file)

        parameters = [[param_json[signal]['patient'],
                       param_json[signal]['record'],
                       param_json[signal]['segment'],
                       param_json[signal]['channel'],
                       param_json[signal]['ids'],
                       param_json[signal]['start_time'],
                       param_json[signal]['end_time']
                       ] for _, signal in enumerate(param_json)]

        return parameters[index]

    @task(depends_on_past=False)
    def extract_annot(patient: str,
                      record: str,
                      segment: str,
                      channel: str,
                      ids: str,
                      start_time: str,
                      end_time: str):

        df_annot = make_annot_df(ids=ids,
                                 record=record,
                                 channel=channel,
                                 start_date=start_time,
                                 end_date=end_time)

        return df_annot

    @task(depends_on_past=False)
    def extract_ecg(patient: str,
                    record: str,
                    segment: str,
                    channel: str,
                    ids: str,
                    start_time: str,
                    end_time: str):

        loader = EdfLoader(patient=patient,
                           record=record,
                           segment=segment)

        df_ecg = loader.convert_edf_to_dataframe(
                    channel_name=channel,
                    start_time=pd.Timestamp(start_time),
                    end_time=pd.Timestamp(end_time))

        return df_ecg

    @task(depends_on_past=True)
    def merge_dataframe(df_annot: pd.DataFrame, df_ecg: pd.DataFrame):

        df = pd.concat([df_ecg, df_annot], axis=1).dropna()

        return df

    @task(depends_on_past=True)
    def merge_all_df(df_list: list):

        df_consolidated = pd.concat(df_list, axis=0)
        df_consolidated.to_csv(f'{output_folder}/df_consolidated.csv')
        df_consolidated.to_pickle(f'{output_folder}/df_consolidated.pkl')

        return df_consolidated

    # Process Pipeline

    dfs_merge = [merge_dataframe(extract_ecg(*parameters[index]),
                                 extract_annot(*parameters[index]))
                 for index, _ in enumerate(parameters)]

    merge_all_df(dfs_merge)


dag_data_extraction = dag_extract_ecg_annotation()
