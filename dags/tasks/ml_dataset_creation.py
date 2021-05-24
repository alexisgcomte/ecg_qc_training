import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(sys.path[0]).parent.parent))
from ecg_qc.ecg_qc import ecg_qc

# sys arg

window = 9
sampling_frequency = 256

# Function declaration
def quality_classification(annotations: list,
                           treshold: float = 0.7) -> int:
    
    # noise treshold

    if np.mean(annotations) >= treshold:
        return 1
    else:
        return 0

def compute_sqi(df_ecg: pd.DataFrame,
                window: str = int,
                sampling_frequency: int = sampling_frequency) -> [float]:

    df_ml = pd.DataFrame(columns=['timestamp_start', 'timestamp_end',
                                  'qSQI_score', 'cSQI_score',
                                  'sSQI_score', 'kSQI_score',
                                  'pSQI_score', 'basSQI_score'])

    df_annot = pd.DataFrame()
    
    ecg_qc_class = ecg_qc()
    print('computing SQI')

    for i in tqdm(range(int(round(
            df_ecg.shape[0] / (window * sampling_frequency),
            0)))):

        start = i * window * sampling_frequency
        end = start + window * sampling_frequency
        sqi_scores = ecg_qc_class.compute_sqi_scores(
            ecg_signal=df_ecg['signal'][start:end].values)


        df_ml = df_ml.append({'timestamp_start': df_ecg['signal'][start:end].index[0],
                              'timestamp_end': df_ecg['signal'][start:end].index[-1],
                              'qSQI_score': sqi_scores[0][0],
                              'cSQI_score': sqi_scores[0][1],
                              'sSQI_score': sqi_scores[0][2],
                              'kSQI_score': sqi_scores[0][3],
                              'pSQI_score': sqi_scores[0][4],
                              'basSQI_score': sqi_scores[0][5]},
                             ignore_index=True)

        # Adding annotators
        # df_annot = pd.DataFrame(columns=df_ecg.columns.drop('signal'))

        annotations  = [quality_classification(
            df_ecg[annotator][start:end].values,
            treshold = 0.7) for annotator in df_ecg.columns.drop('signal')]
        df_annot = df_annot.append([annotations], ignore_index=True)
        
    df_annot.reset_index()
    df_annot.columns = df_ecg.columns.drop('signal')
    # Ajout du concensus

    df_ml = pd.concat([df_ml, df_annot],  axis=1)

    # df_ml['timestamp_start'] = df_ml['timestamp_start'].astype(int)
    # df_ml['timestamp_end'] = df_ml['timestamp_end'].astype(int)

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


#     i = 1
#     while i < len(sys.argv):
#         if sys.argv[i] == '-patient' and i < len(sys.argv)-1:
#             patient = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == '-window' and i < len(sys.argv)-1:
#             window = int(sys.argv[i+1])
#             i += 2
#         elif sys.argv[i] == '-sampling_frequency' and i < len(sys.argv)-1:
#             sampling_frequency = int(sys.argv[i+1])
#             i += 2
#         elif sys.argv[i] == '-input_data_folder' and i < len(sys.argv)-1:
#             input_data_folder = sys.argv[i+1]
#             i += 2
#         elif sys.argv[i] == '-output_folder' and i < len(sys.argv)-1:
#             output_folder = sys.argv[i+1]
#             i += 2
#         else:
#             print('Unknown argument' + str(sys.argv[i]))
#             break


    df_ecg = pd.read_csv('/home/aura-alexis/github/ecg_qc_training/exports/final_df.csv', 
                         index_col=0)
    df_ecg[df_ecg.columns.drop('signal')] = df_ecg[df_ecg.columns.drop('signal')].astype(int)

    df_ecg = df_ecg.iloc[:100_000]

    df_ml = compute_sqi(df_ecg=df_ecg,
                        window=window,
                        sampling_frequency=sampling_frequency)

    print(df_ml.head())


   #  df_ml['classif_avg'] = df_ml['timestamp_start'].apply(
   #      lambda x: classification_correspondance_avg(x, window=9))
# 
   #  df_ml.to_csv('{}/df_ml_{}.csv'.format(output_folder, patient),
   #               index=False)
# 
   #  print('done!')
