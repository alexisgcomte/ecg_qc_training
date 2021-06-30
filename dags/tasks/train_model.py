"""train_model script

This script imports ml_dataset (with specific time window, quality and
consensus, train as model and uploads artificts and metrics to MLFlow.

This file can also be imported as a module and contains the following
fonctions:

    * compute_metrics - for a model, X and y, computes and uploads metrics and
    graphs to analyse the model to MLFlow
    * train_model - From a DataFrame, trains a Random Forest Classifier with
    grid search and exports it with metrics in MLFlow
    * main - the main function of the script

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mlflow
import os
import itertools

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score,\
                            roc_auc_score, precision_score,\
                            confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


def compute_metrics(prefix: str,
                    y_pred: np.array,
                    y_true: np.array,
                    total_seconds=None,
                    mlruns_dir: str = f'{os.getcwd()}/mlruns'):

    """From a model, features X, targets y_true, computes several metrics and
    upload them to ML Flow

    Parameters
    ----------
    model :
        Sklearn model to evaluate
    X : np.array
        Explicative features
    y_pred : np.array
        Target data
    mlruns_dir : str
        Directory where to export MLFlows runs

    """
    mlflow.set_tracking_uri(f'file:///{mlruns_dir}')

    mlflow.log_metric(f'{prefix}_Accuracy', accuracy_score(y_true, y_pred))
    mlflow.log_metric(f'{prefix}_f1-score', f1_score(y_true, y_pred))
    mlflow.log_metric(f'{prefix}_Recall', recall_score(y_true, y_pred))
    mlflow.log_metric(f'{prefix}_precision', precision_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    try:
        tn, fp, fn, tp = cm.ravel()
        mlflow.log_metric(f'{prefix}_tp', tn)
        mlflow.log_metric(f'{prefix}_fp', fp)
        mlflow.log_metric(f'{prefix}_fn', fn)
        mlflow.log_metric(f'{prefix}_tp', tp)
        mlflow.log_metric(f'{prefix}_tp_rate', tn/np.sum(cm))
        mlflow.log_metric(f'{prefix}_fp_rate', fp/np.sum(cm))
        mlflow.log_metric(f'{prefix}_fn_rate', fn/np.sum(cm))
        mlflow.log_metric(f'{prefix}_tp_rate', tp/np.sum(cm))

    except ValueError:
        print('cannot compute metrics')

    try:
        mlflow.log_metric(f'{prefix}_ROC_AUC_score',
                          roc_auc_score(y_true, y_pred))

    except ValueError:
        print('cannot compute ROC_AUC_score')

    try:
        titles_options = [(f'{prefix} - Confusion Matrix', None),
                          (f'{prefix} - Normalized Confusion Matrix', 'true')]
        for title, normalize in titles_options:

            if normalize is None:
                cm_disp = np.round(cm, 0)
            else:
                cm_disp = np.round(cm/np.sum(cm.ravel()), 2)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm_disp,
                                          display_labels=[0, 1])
            disp = disp.plot(cmap=plt.cm.Blues)
            disp.ax_.set_title(title)
            temp_name = f'{mlruns_dir}/{title}.png'
            plt.savefig(temp_name)
            mlflow.log_artifact(temp_name, "confusion-matrix-plots")

        if total_seconds is not None:
            titles_options = [
                (f'{prefix} - Confusion Matrix Minutes', None, 'minutes'),
                (f'{prefix} - Confusion Matrix Seconds', None, 'seconds')]

            for title, normalize, time_unit in titles_options:

                if time_unit == 'minutes':
                    cm_disp = np.round(
                        cm*total_seconds/(60*np.sum(cm.ravel())), 2)
                else:
                    cm_disp = np.round(
                        cm*total_seconds/(np.sum(cm.ravel())), 2)

                disp = ConfusionMatrixDisplay(confusion_matrix=cm_disp,
                                              display_labels=[0, 1])
                disp = disp.plot(cmap=plt.cm.Blues)
                disp.ax_.set_title(title)
                temp_name = f'{mlruns_dir}/{title}.png'
                plt.savefig(temp_name)
                mlflow.log_artifact(temp_name, "confusion-matrix-plots")

    except ValueError:
        print('cannot generate confusion matrices')


def train_model(df_ml: pd.DataFrame,
                df_consolidated_consensus: pd.DataFrame,
                window_s: int,
                consensus_treshold: float,
                quality_treshold: float,
                global_consensus_treshold: float = 0.5,
                sampling_frequency_hz: int = 256,
                mlruns_dir: str = f'{os.getcwd()}/mlruns') -> str:

    mlflow.set_tracking_uri(f'file:///{mlruns_dir}')

    with mlflow.start_run():

        # Declaration of target and features_list

        target_variable = 'consensus'
        features_list = ['qSQI_score', 'cSQI_score', 'sSQI_score',
                         'kSQI_score', 'pSQI_score', 'basSQI_score']

        X = df_ml.loc[:, features_list]
        y = df_ml.loc[:, target_variable]

        # TO MODIFY
        X = X.fillna(0)

        try:
            quality_ratio = round(
                y.value_counts().loc[1] / y.value_counts().sum(),
                4)
        except ValueError:
            quality_ratio = 0

        # Making train and test variables

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print('Repartition of values:', df_ml[target_variable].value_counts())

        # Convertion of pandas DataFrames to numpy arrays
        # before using scikit-learn

        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

        # algo = RandomForestClassifier()
        algo = RandomForestClassifier()
        params = {'min_samples_leaf': np.arange(1, 5, 1),
                  'max_depth': np.arange(11, 16, 1),
                  'max_features': ['auto'],
                  'n_estimators': np.arange(15, 20, 1),
                  }
        grid_search = GridSearchCV(estimator=algo,
                                   param_grid=params,
                                   scoring='f1',
                                   cv=5,
                                   verbose=5,
                                   n_jobs=-1)

        grid_search.fit(X_train, y_train)

        # Performance logging
        mlflow.sklearn.log_model(grid_search, 'model')

        mlflow.log_param('window', window_s)
        mlflow.log_param('consensus_treshold', consensus_treshold)
        mlflow.log_param('global_consensus_treshold',
                         global_consensus_treshold)
        mlflow.log_param('quality_treshold', quality_treshold)
        mlflow.log_param('quality_ratio', quality_ratio)
        mlflow.log_param('best_param', grid_search.best_params_)
        mlflow.log_param('algorith', 'rfc')

        # Train logging

        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)

        compute_metrics('train',
                        y_pred=y_train_pred,
                        y_true=y_train,
                        mlruns_dir=mlruns_dir)

        compute_metrics('test',
                        y_pred=y_test_pred,
                        y_true=y_test,
                        mlruns_dir=mlruns_dir)

        y_pred = grid_search.predict(X)

        predictions = [(window_s * sampling_frequency_hz) * [
            y_pred[i]] for i, _ in enumerate(y_pred)]
        predictions = list(itertools.chain(*predictions))  # flattening
        predictions = predictions[:df_consolidated_consensus.shape[0]]
        df_consolidated_consensus['predictions'] = predictions

        total_seconds = round(df_consolidated_consensus.shape[0] /
                              sampling_frequency_hz, 0)

        compute_metrics(
                'global',
                y_pred=df_consolidated_consensus['predictions'].values,
                y_true=df_consolidated_consensus['consensus'].values,
                mlruns_dir=mlruns_dir,
                total_seconds=total_seconds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument('-i',
                        '--input_file', dest='input_file',
                        help='dafaframe to load',
                        metavar='FILE')
    parser.add_argument('-w',
                        '--window_s',
                        dest='window_s',
                        help='time window_s in sec for split',
                        metavar='FILE',
                        default='9')
    parser.add_argument('-c',
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
                        default='0.5')
    args = parser.parse_args()

    df_ml = pd.read_csv(args.input_file,
                        index_col=0)

    df_consolidated_consensus = pd.read_pickle(
        'exports/df_consolidated_consensus.pkl')

    train_model(df_ml=df_ml,
                window_s=int(args.window_s),
                df_consolidated_consensus=df_consolidated_consensus,
                consensus_treshold=float(args.consensus_treshold),
                quality_treshold=float(args.quality_treshold))
