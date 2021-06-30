"""create_ml_dataset script

This script creates and exports a DataFrame prepared for ecg_qc training. From
a DataFrame with a signal and annotations, for a defined time window, it:
    * computes SQIs
    * computes signal quality for each annotator (treshold)
    * makes a consensus of annotators (treshold)
    * consolidates SQIs, signal quality by annotator and consensus in a
    DataFrame and exports it

This file can also be imported as a module and contains the following
fonctions:

    * compute_sqi - From a DataFrame of with ecg signal, create a DataFrame
    with SQIs computed for a time window in seconds.
    * quality_classification - Classify a segment of annotations in a
    boolean quality, thanks to a treshold.
    * compute_quality - From a DataFrame of with ecg signal, create a DataFrame
    with annotators classification for a time window in seconds.
    * consensus_creation - From a DataFrame of annotators, create the consensus
    (boolean) accord to a treshold
    * make_consensus_and_conso - From two DataFrame of same index with SQIs and
    annotations, makes a consolidated DataFrame with consensus of annotators
    computed by a treshold.
    * main - the main function of the script

"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(sys.path[0]).parent.parent))
from ecg_qc.ecg_qc import ecg_qc


def compute_sqi(df_ecg: pd.DataFrame,
                signal_col: str = 'signal',
                sampling_frequency_hz: int = 256,
                window_s: int = 9) -> pd.DataFrame:
    """From a DataFrame of with ecg signal, create a DataFrame with SQIs
    computed for a time window in seconds.

    Parameters
    ----------
    df_ecg : pd.DataFrame
        DataFrame with ecg_signal, with a constant sampling frequency
    signal_col : str
        Name of the signal column of df_ecg
    sampling_frequency_hz : int
        The sampling frequency of the ECG signal
    window_s : int
        Time window in seconds for signal split and SQIs computation

    Returns
    -------
    df_sqi : pd.DataFrame
        DataFrame with computed SQIs

    """
    ecg_qc_class = ecg_qc(sampling_frequency=sampling_frequency_hz,
                          normalized=None)
    df_sqi = pd.DataFrame(columns=['timestamp_start', 'timestamp_end',
                                   'qSQI_score', 'cSQI_score',
                                   'sSQI_score', 'kSQI_score',
                                   'pSQI_score', 'basSQI_score'])

    for i in range(
         round(df_ecg.shape[0] / (window_s * sampling_frequency_hz))):

        start_index = i * window_s * sampling_frequency_hz
        end_index = start_index + window_s * sampling_frequency_hz + 1
        sqi_scores = ecg_qc_class.compute_sqi_scores(
            ecg_signal=df_ecg[signal_col][start_index:end_index].values)

        df_sqi = df_sqi.append({'timestamp_start': df_ecg[signal_col]
                                [start_index:end_index].index[0],
                                'timestamp_end': df_ecg[signal_col]
                                [start_index:end_index].index[-1],
                                'qSQI_score': sqi_scores[0][0],
                                'cSQI_score': sqi_scores[0][1],
                                'sSQI_score': sqi_scores[0][2],
                                'kSQI_score': sqi_scores[0][3],
                                'pSQI_score': sqi_scores[0][4],
                                'basSQI_score': sqi_scores[0][5]},
                               ignore_index=True)

    return df_sqi


def quality_classification(annotations: list,
                           quality_treshold: float = 0.8) -> bool:
    """Classify a segment of annotations in a boolean quality, thanks to a
    treshold. For a list of annotations with same frequency, the treshold
    is compared with the mean for classification.

    Parameters
    ----------
    annotations : list
        List of the annotations, wich are already booleans
    quality_treshold : float
        The quality treshold for quality classification in boolean class

    Returns
    -------
    quality : float
        The classified quality of the annotations with selected treshold

    """
    if np.mean(annotations) >= quality_treshold:
        quality = 1
    else:
        quality = 0

    return quality


def compute_quality(df_ecg: pd.DataFrame,
                    signal_col: str = 'signal',
                    record_col: str = 'record',
                    sampling_frequency_hz: int = 256,
                    window_s: int = 9,
                    quality_treshold: float = 0.8) -> pd.DataFrame:
    """From a DataFrame of with ecg signal, create a DataFrame with annotators
    classification for a time window in seconds.

    Parameters
    ----------
    df_ecg : pd.DataFrame
        DataFrame with ecg_signal, with a constant sampling frequency
    signal_col : str
        Name of the signal column of df_ecg
    sampling_frequency_hz : int
        The sampling frequency of the ECG signal
    window_s : int
        Time window in seconds for signal split and SQIs computation
    quality_treshold : float
        The quality treshold for quality classification in boolean class

    Returns
    -------
    df_annot : pd.DataFrame
        DataFrame with annotators classifications

    """
    df_annot = pd.DataFrame()
    annotators = df_ecg.columns.drop([signal_col, record_col])

    for i in range(
         round(df_ecg.shape[0] / (window_s * sampling_frequency_hz))):

        start_index = i * window_s * sampling_frequency_hz
        end_index = start_index + window_s * sampling_frequency_hz + 1
        annotations = [quality_classification(
            df_ecg[annotator][start_index:end_index].values,
            quality_treshold=quality_treshold) for annotator in annotators]
        df_annot = df_annot.append([annotations], ignore_index=True)

    df_annot.reset_index()
    df_annot.columns = annotators
    df_annot['record'] = df_ecg['record']

    return df_annot


def consensus_creation(df_annot: pd.DataFrame,
                       consensus_col: str = 'consensus',
                       consensus_treshold: float = 0.5) -> pd.DataFrame:
    """From a DataFrame of annotators, create the consensus (boolean) accord to
    a treshold. The mean of annotators classification is compared with
    consensus treshold for boolean classification.

    Parameters
    ----------
    df_annot : pd.DataFrame
        DataFrame with annotations only, one annotator by column, one
        observation by line
    consensus_col : str
        Name of the column to input the consensus
    consensus_treshold : float
        The consensus treshold for classification in boolean class

    Returns
    -------
    df_annot : pd.DataFrame
        DataFrame with added boolean consensus for each observation

    """
    df_annot[consensus_col] = df_annot.mean(axis=1)
    df_annot[consensus_col] = df_annot[consensus_col].apply(
        lambda x: 1 if x >= consensus_treshold else 0)

    return df_annot


def make_consensus_and_conso(df_sqi: pd.DataFrame,
                             df_annot: pd.DataFrame,
                             consensus_treshold: float = 0.5,
                             consensus_col: str = 'consensus') -> pd.DataFrame:
    """From two DataFrame of same index with SQIs and annotations, makes a
    consolidated DataFrame with consensus of annotators computed by a treshold.

    Parameters
    ----------
    df_sqi : pd.DataFrame
        DataFrame with computed SQIs
    df_annot : pd.DataFrame
        DataFrame with annotations only, one annotator by column, one
        observation by line
    consensus_treshold : float
        The consensus treshold for classification in boolean class
    consensus_col : str
        Name of the column to input the consensus

    Returns
    -------
    df_conso : pd.DataFrame
        Consolidated DataFrame with consensus

    """
    df_annot = consensus_creation(df_annot=df_annot,
                                  consensus_col=consensus_col,
                                  consensus_treshold=consensus_treshold)
    df_conso = pd.concat([df_sqi, df_annot],  axis=1)

    return df_conso


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument('-i',
                        '--input_file', dest='input_file',
                        help='dafaframe to load',
                        metavar='FILE')
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        help='output_folder_for_ddf',
                        metavar='FILE',
                        default='./exports')
    parser.add_argument('-s', '--sampling_frequency_hz',
                        dest='sampling_frequency_hz',
                        help='sampling_frequency_hz_of_file',
                        metavar='FILE',
                        default='256')
    parser.add_argument('-w',
                        '--window_s',
                        dest='window_s',
                        help='time window_s in sec for split',
                        metavar='FILE',
                        default='9')
    parser.add_argument('-ch',
                        '--consensus_treshold',
                        dest='consensus_treshold',
                        help='percentage of agreement for consensus',
                        metavar='FILE',
                        default='0.5')
    parser.add_argument('-q',
                        '--quality_treshold',
                        dest='quality_treshold',
                        help='treshold to determine quality',
                        metavar='FILE',
                        default='0.8')
    args = parser.parse_args()

    df_ecg = pd.read_csv(args.input_file)

    df_sqi = compute_sqi(df_ecg=df_ecg,
                         sampling_frequency_hz=int(args.sampling_frequency_hz),
                         window_s=int(args.window_s))

    df_annot = compute_quality(df_ecg=df_ecg,
                               sampling_frequency_hz=int(
                                   args.sampling_frequency_hz),
                               window_s=int(args.window_s),
                               quality_treshold=float(args.quality_treshold))

    df_conso = make_consensus_and_conso(df_sqi=df_sqi,
                                        df_annot=df_annot,
                                        consensus_treshold=float(
                                            args.consensus_treshold))

    df_conso.to_csv(f'{args.output_folder}/df_ml_{int(args.window_s)}'
                    f'_{float(args.quality_treshold)}_'
                    f'{float(args.consensus_treshold)}.csv',
                    index=False)
