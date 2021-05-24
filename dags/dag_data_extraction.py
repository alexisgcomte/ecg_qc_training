from airflow.decorators import dag, task
# from airflow.operators.python import PythonOperator
# from datetime import datetime
from airflow.utils.dates import days_ago
from tasks.import_annotations import make_result_df
from tasks.import_ecg_segment import EdfLoader
import pandas as pd
import json
import os

# Parameters
# folder =  os.getcwd()
folder = '/home/aura-alexis/github/ecg_qc_training'
output_folder = folder + '/exports'

default_args = {'owner': 'airflow',
                'depends_on_past': False,
                'retries': 0}


# Signals to loads
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

# parameters = parameters[0]

@dag(default_args=default_args,
     schedule_interval=None,
     start_date=days_ago(1),
     tags=['ecg_qc', 'preprocessing', 'working'])
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
                       ] for  _, signal in enumerate(param_json)]

        return parameters[index]

    def extract_annot(patient: str,
                      record: str,
                      segment: str,
                      channel: str,
                      ids: str,
                      start_time: str,
                      end_time: str):

        df_annot = make_result_df(ids=ids,
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
        df = pd.concat(df_list, axis=0)
        df.to_csv(output_folder+"/final_df.csv")
        return df

    # Process Pipeline

    dfs_merge = [merge_dataframe(extract_ecg(*parameters[index]),
                                extract_annot(*parameters[index]))
                for index, _ in enumerate(parameters)]

    # parameters = extract_signal_parameters(0)
    # df_annot = extract_annot(*parameters)
    # df_annot = extract_annot(ids=ids,record=record,channel=channel,start_date=start_time,end_date=end_time)
    # df_ecg = extract_ecg(*parameters)
    # df_merge = merge_dataframe(df_ecg, df_annot)
   # df_merge = merge_dataframe(df_annot, df_annot)
    merge_all_df(dfs_merge)

#    parameters >> df_ecg >> df_merge

dag_data_extraction = dag_extract_ecg_annotation()
