"""dag_train_models DAG

This script is meant to be used as a dag. From previously extracted
consolidated data, will make train model for use in ecg_qc according to
parameters:
    * windows_s: time window in second to split records
    * quality_treshold: by window of time, the ratio of quality required to
    determine the class (1 = good, 0 = noisy)
    * consensus_tresholds: the ratio of consensus to determine quality
"""

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from tasks.create_ml_dataset import compute_sqi, compute_quality, \
                                    make_consensus_and_conso
from tasks.train_model import train_model
from tasks.make_consolidated_consensus import make_consolidated_consensus
import pandas as pd
import os


# Parameters
folder = os.environ['AIRFLOW_HOME']
output_folder = f'{folder}/exports'

default_args = {'owner': 'airflow',
                'depends_on_past': False,
                'retries': 0}

# Combinations to make
windows_s = [9]
quality_tresholds = [0.5]
consensus_tresholds = [0.5]

@dag(default_args=default_args,
     schedule_interval=None,
     start_date=days_ago(1),
     tags=['ecg_qc', 'train', 'medium10'])
def dag_train_ecg():

    @task(depends_on_past=True)
    def t_load_df(df_path: str) -> pd.DataFrame:

        df_consolidated = pd.read_pickle(df_path)

        return df_consolidated

    @task(depends_on_past=True)
    def t_compute_sqi(df: pd.DataFrame,
                      window_s: int = 9,
                      sampling_frequency_hz: int = 256):

        df_sqi = compute_sqi(df_ecg=df,
                             window_s=window_s,
                             sampling_frequency_hz=sampling_frequency_hz)

        return df_sqi

    @task(depends_on_past=True)
    def t_compute_quality(df: pd.DataFrame,
                          window_s: int = 9,
                          quality_treshold: float = 0.7,
                          sampling_frequency_hz: int = 256):

        df_annot = compute_quality(df_ecg=df,
                                   sampling_frequency_hz=sampling_frequency_hz,
                                   window_s=window_s,
                                   quality_treshold=quality_treshold)

        return df_annot

    @task(depends_on_past=True)
    def t_make_consensus_and_conso(df_sqi: pd.DataFrame,
                                   df_annot: pd.DataFrame,
                                   consensus_treshold: float = 0.7):

        df_conso = make_consensus_and_conso(
            df_sqi=df_sqi,
            df_annot=df_annot,
            consensus_treshold=consensus_treshold)

        df_conso.to_csv(f'{output_folder}/df_consolidated_ml.csv')

        return df_conso

    @task(depends_on_past=True)
    def t_ml_training(df_ml: pd.DataFrame,
                    df_consolidated_consensus: pd.DataFrame,
                    mlruns_dir: str = f'{folder}/mlruns/',
                    window_s: int = 9,
                    consensus_treshold: float = 0.7,
                    quality_treshold: float = 0.7):

        train_model(df_ml=df_ml,
        df_consolidated_consensus=df_consolidated_consensus,
                    mlruns_dir=mlruns_dir,
                    window_s=window_s,
                    consensus_treshold=consensus_treshold,
                    quality_treshold=quality_treshold)
    
    @task(depends_on_past=True)
    def t_make_consolidated_consensus(
        df_consolidated: pd.DataFrame,
        consensus_threshold: int = 0.7) -> pd.DataFrame:

        df_consolidated_consensus = make_consolidated_consensus(df_consolidated)

        return df_consolidated_consensus

    # Parameter combination and comprehension list
    df_consolidated = t_load_df(f'{output_folder}/df_consolidated.pkl')
    df_consolidated_consensus = t_make_consolidated_consensus(df_consolidated)

    for window_s in windows_s:
        df_sqi = t_compute_sqi(df=df_consolidated,
                               window_s=window_s)

        for quality_treshold in quality_tresholds:

            df_annot = t_compute_quality(df=df_consolidated,
                                         quality_treshold=quality_treshold)

            for consensus_treshold in consensus_tresholds:
                df_ml = t_make_consensus_and_conso(
                    df_sqi=df_sqi,
                    df_annot=df_annot,
                    consensus_treshold=consensus_treshold)

                t_ml_training(df_ml=df_ml,
                                        df_consolidated_consensus=df_consolidated_consensus,
                                     window_s=window_s,
                                     consensus_treshold=consensus_treshold,
                                     quality_treshold=quality_treshold)

                


dag_train = dag_train_ecg()
