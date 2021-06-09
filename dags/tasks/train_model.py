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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score,\
                            roc_auc_score, precision_score,\
                            plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def compute_metrics(prefix: str,
                    model,
                    X: np.array,
                    y_true: np.array,
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
    y_pred = model.predict(X)

    mlflow.log_metric(f'{prefix}_Accuracy', accuracy_score(y_true, y_pred))
    mlflow.log_metric(f'{prefix}_f1-score', f1_score(y_true, y_pred))
    mlflow.log_metric(f'{prefix}Recall', recall_score(y_true, y_pred))
    mlflow.log_metric(f'{prefix}precision', precision_score(y_true, y_pred))
    mlflow.log_metric(f'{prefix}_ROC_AUC_score', roc_auc_score(y_true, y_pred))
    titles_options = [(f'{prefix} - Confusion matrix', None),
                      (f'{prefix} - Normalized confusion matrix', 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(estimator=model,
                                     X=X,
                                     y_true=y_true,
                                     display_labels=[0, 1],
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        temp_name = f'{mlruns_dir}/{title}.png'
        plt.savefig(temp_name)
        mlflow.log_artifact(temp_name, "confusion-matrix-plots")


def train_model(df_ml: pd.DataFrame,
                window_s: int,
                consensus_treshold: float,
                quality_treshold: float,
                mlruns_dir: str = f'{os.getcwd()}/mlruns'):

    mlflow.set_tracking_uri(f'file:///{mlruns_dir}')

    with mlflow.start_run():

        # Setting Nan Value at 0 (neutral)
        df_ml = df_ml.fillna(0)

        # Declaration of target and features_list

        target_variable = 'consensus'
        features_list = ['qSQI_score', 'cSQI_score', 'sSQI_score',
                         'kSQI_score', 'pSQI_score', 'basSQI_score']

        X = df_ml.loc[:, features_list]
        y = df_ml.loc[:, target_variable]

        try:
            quality_ratio = round(
                y.value_counts().loc[1] / y.value_counts().sum(),
                4)
        except Exception:
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

        # Declaration, fit and application of numeric transfomer

        std = StandardScaler()
        std.fit(X_train)
        X_train = std.transform(X_train)
        X_test = std.transform(X_test)

        # Declaration of algorithm and parameters for gridsearch

        algo = RandomForestClassifier()
        params = {'min_samples_leaf': np.arange(6, 12, 3),
                  'max_depth': np.arange(6, 12, 3),
                  # 'max_features' : np.arange(6,12,2),
                  'n_estimators': np.arange(6, 12, 3),
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
        mlflow.log_param('quality_treshold', quality_treshold)
        mlflow.log_param('quality_ratio', quality_ratio)

        # Train logging
        compute_metrics('train',
                        model=grid_search,
                        X=X_train,
                        y_true=y_train,
                        mlruns_dir=mlruns_dir)
        compute_metrics('test',
                        model=grid_search,
                        X=X_test,
                        y_true=y_test,
                        mlruns_dir=mlruns_dir)


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

    train_model(df_ml=df_ml,
                window_s=int(args.window_s),
                consensus_treshold=float(args.consensus_treshold),
                quality_treshold=float(args.quality_treshold))
