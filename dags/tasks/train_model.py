import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mlflow
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score,\
                            roc_auc_score, precision_score,\
                            plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def train_model(df: pd.DataFrame,
                mlruns_dir: str = f'{os.getcwd()}/mlruns',
                window: int = 9,
                consensus_ratio: float = 0.7,
                quality_treshold: float = 0.7):

    mlflow.set_tracking_uri(f'file:///{mlruns_dir}')

    with mlflow.start_run():
        df = df.fillna(0)  # To change?
        print(df.head())
        print(df['consensus'].value_counts())

        # Declaration of target and features_list

        target_variable = 'consensus'
        features_list = ['qSQI_score', 'cSQI_score', 'sSQI_score',
                         'kSQI_score', 'pSQI_score', 'basSQI_score']

        X = df.loc[:, features_list]
        y = df.loc[:, target_variable]

        # Distinction of categorical features

        categorical_features_str = (X.select_dtypes(
            include=['object']).columns)
        categorical_features = [X.columns.get_loc(i)
                                for i in categorical_features_str]

        # Distinction of numeric features

        numeric_features_str = X.columns.drop(categorical_features_str)
        numeric_features = [X.columns.get_loc(i) for i in numeric_features_str]

        # Convertion of pandas DataFrames to numpy arrays
        # before using scikit-learn

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Convertion of pandas DataFrames to numpy arrays
        # before using scikit-learn

        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

        # Declaration of the categorical and numeric transfomers

        categorical_transformer = OneHotEncoder(drop='first')
        numeric_transformer = StandardScaler()

        # Declaration of the feature encoder

        feature_encoder = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        feature_encoder.fit(X_train)
        X_train = feature_encoder.transform(X_train)
        X_test = feature_encoder.transform(X_test)

        # Declaration of algorithm and parameters for gridsearch

        algo = RandomForestClassifier()
        params = {'min_samples_leaf': np.arange(6, 12, 2),
                  'max_depth': np.arange(6, 12, 2),
                  # 'max_features' : np.arange(6,12,2),
                  'n_estimators': np.arange(6, 12, 2),
                  }
        grid_search = GridSearchCV(estimator=algo,
                                   param_grid=params,
                                   scoring='f1',
                                   cv=5,
                                   verbose=5,
                                   n_jobs=-1)

#         grid_search = LogisticRegression()

        grid_search.fit(X_train, y_train)

        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)

        # Performance logging

        mlflow.sklearn.log_model(grid_search, 'model')

        mlflow.log_param('window', window)
        mlflow.log_param('consensus_ratio', consensus_ratio)
        mlflow.log_param('quality_treshold', quality_treshold)

        # Train logging

        mlflow.log_metric('train_Accuracy', accuracy_score(y_train,
                                                           y_train_pred))
        mlflow.log_metric('train_f1-score', f1_score(y_train,
                                                     y_train_pred))
        mlflow.log_metric('train_Recall', recall_score(y_train,
                                                       y_train_pred))
        mlflow.log_metric('train_precision', precision_score(y_train,
                                                             y_train_pred))
        mlflow.log_metric('train_ROC_AUC_score', roc_auc_score(y_train,
                                                               y_train_pred))

        titles_options = [('Train - Confusion matrix', None),
                          ('Train - Normalized confusion matrix', 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(grid_search, X_train, y_train,
                                         display_labels=[0, 1],
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)

            temp_name = f'{mlruns_dir}/{title}.png'
            plt.savefig(temp_name)
            mlflow.log_artifact(temp_name, "confusion-matrix-plots")

        # Test logging

        mlflow.log_metric('test_Accuracy', accuracy_score(y_test,
                                                          y_test_pred))
        mlflow.log_metric('test_f1-score', f1_score(y_test,
                                                    y_test_pred))
        mlflow.log_metric('test_Recall', recall_score(y_test,
                                                      y_test_pred))
        mlflow.log_metric('test_precision', precision_score(y_test,
                                                            y_test_pred))
        mlflow.log_metric('test_ROC_AUC_score', roc_auc_score(y_test,
                                                              y_test_pred))

        titles_options = [('Test - Confusion matrix', None),
                          ('Test - Normalized confusion matrix', 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(grid_search, X_test, y_test,
                                         display_labels=[0, 1],
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)

            temp_name = f'{mlruns_dir}/{title}.png'
            plt.savefig(temp_name)
            mlflow.log_artifact(temp_name, 'confusion-matrix-plots')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("-w", "--window", dest="window",
                        help="time window in sec for split", metavar="FILE")
    parser.add_argument("-c", "--consensus_ratio", dest="consensus_ratio",
                        help="percentage of agreement for consensus",
                        metavar="FILE")
    parser.add_argument("-q", "--quality_treshold", dest="quality_treshold",
                        help="treshold to determine quality", metavar="FILE")
    parser.add_argument("-i", "--input_file", dest="input_file",
                        help="dafaframe to load", metavar="FILE")

    args = parser.parse_args()

    df_ecg = pd.read_csv(args.input_file,
                         index_col=0)

    train_model(df=df_ecg,
                window=int(args.window),
                consensus_ratio=float(args.consensus_ratio),
                quality_treshold=float(args.quality_treshold))
