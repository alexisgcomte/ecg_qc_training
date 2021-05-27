import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(sys.path[0]).parent.parent))
from ecg_qc.ecg_qc import ecg_qc


window = 9
consensus_ratio = 0.7
sampling_frequency = 256


def quality_classification(annotations: list,
                           quality_treshold: float = 0.5) -> int:

    # noise treshold

    if np.mean(annotations) >= quality_treshold:
        return 1
    else:
        return 0


def consensus_creation(df_annot: pd.DataFrame,
                       consensus_ratio: float = 0.7) -> pd.DataFrame:

    df_annot['consensus'] = df_annot.mean(axis=1)
    df_annot['consensus'] = df_annot['consensus'].apply(
        lambda x: 1 if x >= consensus_ratio else 0)

    return df_annot


def compute_sqi(df_ecg: pd.DataFrame,
                window: int = 9,
                consensus_ratio: float = 0.7,
                quality_treshold: float = 0.5,
                sampling_frequency: int = sampling_frequency) -> pd.DataFrame:

    df_ml = pd.DataFrame(columns=['timestamp_start', 'timestamp_end',
                                  'qSQI_score', 'cSQI_score',
                                  'sSQI_score', 'kSQI_score',
                                  'pSQI_score', 'basSQI_score'])

    df_annot = pd.DataFrame()
    annotators = df_ecg.columns.drop('signal')

    ecg_qc_class = ecg_qc()
    print('computing SQI')

    for i in tqdm(range(int(round(
            df_ecg.shape[0] / (window * sampling_frequency),
            0)))):

        start = i * window * sampling_frequency
        end = start + window * sampling_frequency
        sqi_scores = ecg_qc_class.compute_sqi_scores(
            ecg_signal=df_ecg['signal'][start:end].values)

        df_ml = df_ml.append({'timestamp_start': df_ecg['signal']
                              [start:end].index[0],
                              'timestamp_end': df_ecg['signal']
                              [start:end].index[-1],
                              'qSQI_score': sqi_scores[0][0],
                              'cSQI_score': sqi_scores[0][1],
                              'sSQI_score': sqi_scores[0][2],
                              'kSQI_score': sqi_scores[0][3],
                              'pSQI_score': sqi_scores[0][4],
                              'basSQI_score': sqi_scores[0][5]},
                             ignore_index=True)

        # Adding annotators
        # df_annot = pd.DataFrame(columns=df_ecg.columns.drop('signal'))

        annotations = [quality_classification(
            df_ecg[annotator][start:end].values,
            quality_treshold=quality_treshold) for annotator in annotators]
        df_annot = df_annot.append([annotations], ignore_index=True)

    df_annot.reset_index()
    df_annot.columns = annotators
    df_annot = consensus_creation(df_annot, consensus_ratio=consensus_ratio)
    # Ajout du consensus

    df_ml = pd.concat([df_ml, df_annot],  axis=1)

    return df_ml


def classification_correspondance_avg(timestamp,
                                      sampling_frequency=1000,
                                      window=9):

    start = timestamp
    end = start + window * sampling_frequency - 1
    cons_value = df_ecg.loc[start:end]['cons'].values

    classif_avg = np.mean([int(cons_value)])

    return classif_avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("-w", "--window", dest="window",
                        help="time window in sec for split", metavar="FILE")
    parser.add_argument("-c", "--consensus_ratio", dest="consensus_ratio",
                        help="percentage of agreement for consensus",
                        metavar="FILE")
    parser.add_argument("-q", "--quality_treshold", dest="quality_treshold",
                        help="treshold to determine quality", metavar="FILE")
    parser.add_argument("-sf", "--sampling_frequency",
                        dest="sampling_frequency",
                        help="sampling_frequency_of_file", metavar="FILE")
    parser.add_argument("-i", "--input_file", dest="input_file",
                        help="dafaframe to load", metavar="FILE")
    parser.add_argument("-o", "--output_folder", dest="output_folder",
                        help="output_folder_for_ddf", metavar="FILE",
                        default="./exports")

    args = parser.parse_args()

    df_ecg = pd.read_csv(args.input_file,
                         index_col=0)

    df_ml = compute_sqi(df_ecg=df_ecg,
                        window=int(args.window),
                        consensus_ratio=float(args.consensus_ratio),
                        quality_treshold=float(args.quality_treshold),
                        sampling_frequency=int(args.sampling_frequency))

    df_ml.to_csv(f'{args.output_folder}/df_consolidated_consensus.csv',
                 index=False)
