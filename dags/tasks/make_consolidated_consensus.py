import pandas as pd
from dags.tasks.create_ml_dataset import consensus_creation
import os

# Parameters
folder = os.environ['AIRFLOW_HOME']
output_folder = f'{folder}/exports'


def make_consolidated_consensus(
    df_consolidated: pd.DataFrame,
    gobal_consensus_threshold: int = 0.7,
    export_name: str = 'df_consolidated_consensus') -> pd.DataFrame:

    
    annotators_columns = df_consolidated.columns.drop(['record', 'signal'])
    df_consolidated_consensus = consensus_creation(
        df_consolidated[annotators_columns],
        consensus_treshold=gobal_consensus_threshold)

    df_consolidated_consensus.to_pickle(f'{output_folder}/{export_name}.pkl')

    return df_consolidated_consensus


if __name__ == '__main__':
    df_consolidated = pd.read_csv(f'{output_folder}/ecg_annoted_PAT_6_77_emg6+emg6-.csv')
    print(df_consolidated.head())
    df_consolidated_consensus = make_consolidated_consensus(df_consolidated)
    print(df_consolidated_consensus.head())