from airflow.decorators import dag, task
# from airflow.operators.python import PythonOperator
# from datetime import datetime
from airflow.utils.dates import days_ago
from tasks.import_annotations import make_result_df
from tasks.import_ecg_segment import EdfLoader
import pandas as pd


# Parameters

patient = 'PAT_6'
record = '77'
segment = 's1'
channel = 'ECG1+ECG1-'
ids = '2,3,4'
start_time = '2020-12-18 13:00:00'
end_time = '2020-12-18 14:30:00'
output_folder = '/home/aura-alexis/github/ecg_qc_training/exports'

default_args = {'owner': 'airflow',
                'depends_on_past': False,
                'retries': 0}


@dag(default_args=default_args,
     schedule_interval=None,
     start_date=days_ago(1),
     tags=['ecg_qc'])
def extract_ecg_annotation():

    @task(depends_on_past=False)
    def extract_annot():

        df_annot = make_result_df(ids=ids,
                                  record=record,
                                  channel=channel,
                                  start_date=start_time,
                                  end_date=end_time)

        return df_annot

    @task(depends_on_past=False)
    def extract_ecg():

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
        df.to_csv(output_folder+"/test_23.csv")
        return df

    # Process Pipeline
    df_annot = extract_annot()
    df_ecg = extract_ecg()
    merge_dataframe(df_ecg, df_annot)


test_dat = extract_ecg_annotation()
