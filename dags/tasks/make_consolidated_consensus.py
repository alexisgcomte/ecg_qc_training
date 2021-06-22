"""make_consolidated_consensus script

This script load consolidated dataset and computes consensus line by line,
that is to say at frequency precision.

This file can also be imported as a module and contains the following
fonctions:

    * make_consolidated_consensus - From a DataFrame of with ecg signal,
    create a DataFrame with SQIs computed for a time window in seconds.
"""


import pandas as pd
import argparse
import os
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent.parent))
from dags.tasks.create_ml_dataset import consensus_creation


# Parameters
folder = os.environ['AIRFLOW_HOME']
output_folder = f'{folder}/exports'


def make_consolidated_consensus(
        df_consolidated: pd.DataFrame,
        gobal_consensus_threshold: int = 0.7,
        export_name: str = 'df_consolidated_consensus') -> pd.DataFrame:

    """From consolidated DataFrame, according to a global consensus treshold,
    creates a DataFrame with global consensus.

    Parameters
    ----------
    df_consolidated : pd.DataFrame
        DataFrame with ecg signals and annotations
    gobal_consensus_threshold : int
        The consensus treshold for classification in boolean class
    export_name : str
        TName for exportation

    Returns
    -------
    df_consolidated_consensus : pd.DataFrame
        DataFrame with computed SQIs
    """

    annotators_columns = df_consolidated.columns.drop(['record', 'signal'])
    df_consolidated_consensus = consensus_creation(
        df_consolidated[annotators_columns],
        consensus_treshold=gobal_consensus_threshold)

    df_consolidated_consensus.to_pickle(f'{output_folder}/{export_name}.pkl')

    return df_consolidated_consensus


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument('-i',
                        '--input_file', dest='input_file',
                        help='dafaframe to load',
                        metavar='FILE')
    parser.add_argument('-n', '--name_export',
                        dest='name_export',
                        help='name of the exported file',
                        default='df_consolidated_consensus',
                        metavar='FILE')
    parser.add_argument('-q',
                        '--quality_treshold',
                        dest='quality_treshold',
                        help='treshold to determine quality',
                        metavar='FILE',
                        default='0.7')
    args = parser.parse_args()

    df_consolidated = pd.read_csv(f'{args.input_file}')
    df_consolidated_consensus = make_consolidated_consensus(
        df_consolidated=df_consolidated,
        gobal_consensus_threshold=args.quality_treshold,
        export_name=f'{args.name_export}')
