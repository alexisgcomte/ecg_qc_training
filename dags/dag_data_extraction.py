"""create_ecg_dataset script

This script creates and exports a DataFrame for an ECG signal. It takes into
consideration several elements to load corresponding EDF file of the server.

This file can also be imported as a module and contains the following
class:

    * EdfLoader - A class used to load an edf file and export it in
    DataFrame format
"""

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from tasks.import_annotations import make_result_df
from tasks.import_ecg_segment import EdfLoader
from tasks.create_ml_dataset import compute_sqi, compute_quality, \
                                    make_consensus_and_conso
from tasks.train_model import train_model
import pandas as pd
import json
import os


# Parameters
folder = os.environ['AIRFLOW_HOME']
output_folder = f'{folder}/exports'

default_args = {'owner': 'airflow',
                'depends_on_past': False,
                'retries': 0}

# Signals to loads
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
     tags=['ecg_qc', 'preprocessing', 'extraction', 'optimized2'])
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
        # df_consolidated.to_csv(f'{output_folder}/df_consolidated.csv')

        return df

    @task(depends_on_past=True)
    def t_compute_sqi(df: pd.DataFrame,
                      window_s: int = 9,
                      consensus_treshold: float = 0.7,
                      quality_treshold: float = 0.7,
                      sampling_frequency: int = 256):

        df_sqi = compute_sqi(df_ecg=df,
                             window_s=window_s,
                             consensus_treshold=consensus_treshold,
                             quality_treshold=quality_treshold,
                             sampling_frequency=sampling_frequency)

#        df_consensus.to_csv(f'{output_folder}/df_consensus.csv')

        return df_sqi

    @task(depends_on_past=True)
    def t_compute_quality(df: pd.DataFrame,
                          window_s: int = 9,
                          consensus_treshold: float = 0.7,
                          quality_treshold: float = 0.7,
                          sampling_frequency: int = 256):

        df_annot = compute_quality(df_ecg=df,
                                   window_s=window_s,
                                   consensus_treshold=consensus_treshold,
                                   quality_treshold=quality_treshold,
                                   sampling_frequency=sampling_frequency)

#        df_consensus.to_csv(f'{output_folder}/df_consensus.csv')

        return df_annot

    @task(depends_on_past=True)
    def t_make_consensus_and_conso(df_sqi: pd.DataFrame,
                                   df_annot: pd.DataFrame,
                                   window_s: int = 9,
                                   consensus_treshold: float = 0.7,
                                   quality_treshold: float = 0.7,
                                   sampling_frequency: int = 256):

        df_conso = make_consensus_and_conso(
            df_sqi=df_sqi,
            df_annot=df_annot,
            window_s=window_s,
            consensus_treshold=consensus_treshold,
            quality_treshold=quality_treshold,
            sampling_frequency=sampling_frequency)

#        df_consensus.to_csv(f'{output_folder}/df_consensus.csv')

        return df_conso

    @task(depends_on_past=True)
    def ml_training(df: pd.DataFrame,
                    mlruns_dir: str = f'{folder}/mlruns/',
                    window_s: int = 9,
                    consensus_treshold: float = 0.7,
                    quality_treshold: float = 0.7):

        train_model(df=df,
                    mlruns_dir=mlruns_dir,
                    window_s=window_s,
                    consensus_treshold=consensus_treshold,
                    quality_treshold=quality_treshold)

    # Process Pipeline

    dfs_merge = [merge_dataframe(extract_ecg(*parameters[index]),
                                 extract_annot(*parameters[index]))
                 for index, _ in enumerate(parameters)]

    df_consolidated = merge_all_df(dfs_merge)

    # Parameter combination and comprehension list
    window_ss = [9]
    consensus_tresholds = [0.5]
    quality_tresholds = [0.5]

    for window_s in window_ss:
        df_sqi = t_compute_sqi(df=df_consolidated,
                               window_s=window_s)

        for quality_treshold in quality_tresholds:

            df_annot = t_compute_quality(df=df_consolidated,
                                         quality_treshold=quality_treshold)

            for consensus_treshold in consensus_tresholds:
                df_conso = t_make_consensus_and_conso(
                    df_sqi=df_sqi,
                    df_annot=df_annot,
                    consensus_treshold=consensus_treshold)

                ml_training(df=df_conso,
                            window_s=window_s,
                            consensus_treshold=consensus_treshold,
                            quality_treshold=quality_treshold)


dag_data_extraction = dag_extract_ecg_annotation()
