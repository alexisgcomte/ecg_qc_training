# ecg_qc_training


## I) Reminder of annotated segments

patient: PAT_6 / record:77 / channel: emg6+emg6- / time: 2020-12-18 13:00:00 à 2020-12-18 14:30:00
patient: PAT_4 / record: 11/ channel: ECG1+ECG1- / time: 2020-12-15 21:50:00 à 2020-12-15 22:50:00
patient: PAT_2 / record: _3 / channel:  EMG1+EMG1- / time : 2020-12-14 21:55:00 à 2020-12-14 23:05:00


## II) Individual scripts

### 1) import_ecg_segment (optional)

Allow to import from an edf file a certain window of time for a particular channel.

Exemple of use:
```python
python3 dags/tasks/import_ecg_segment.py -p PAT_6 -r 77 -sg s1 -ch emg6+emg6- -st '2020-12-18 13:00:00' -et '2020-12-18 14:30:00'
```

### 2) import_annotations (optional)

From the SQL database storing annotations, import them for a certain windows time and other patient parameters.

Exemple of use:
```python
python3 dags/tasks/import_annotations.py -p PAT_6 -r 77 -ch emg6+emg6- -ids 2,3,4 -st '2020-12-18 13:00:00' -et '2020-12-18 14:30:00' -s 256
```

### 3) create_annoted_dataset

Combines import_ecg_segment and import_annotations.

Exemple of use:
```python
python3 dags/tasks/create_annoted_dataset.py -p PAT_6 -r 77 -sg s1 -ch emg6+emg6- -ids 2,3,4 -st '2020-12-18 13:00:00' -et '2020-12-18 14:30:00' -s 256
```

### 4) create_ml_dataset

Compute SQL for merge dataset and concensus.

Exemple of use:
```python
python3 dags/tasks/create_ml_dataset.py -w 9 -c 0.5 -q 0.3 -s 256 -i ./exports/ecg_annoted_PAT_6_77_emg6+emg6-.csv -o ./exports
```


### TO DO

Coder un bout manquant ici (transformation du dataframe consolidé en df avec consensus par metrics)
(modifier nom des colonnes censensus global, non global, etc)
Améliorer les scripts de génération de matrice de confusion
Améliorer lisibilté des datasets successifs
Modifier Python en bash
Variabiliser le consensus par point (2 ou 3 paramètres maximums: tous, majorité, au moins 1)
import de from dags.tasks.create_ml_dataset import consensus_creation

```python
python3 dags/tasks/make_consolidated_consensus.py 
```
### 5) train_model

From previous csv, train a model and logs it in MLFlows.

Exemple of use:
```python
python3 dags/tasks/train_model.py -w 9 -c 0.5 -q 0.3 -i ./exports/df_ml_9_0.3_0.5.csv
```

## III) Using Airflow and MLFlow

Start PostgreSQL server:
```bash
source env/bin/activate
source env.sh
docker-compose up -d
```

In Python venv:

The first time:
```bash
make init_airflow
```

Starting Airflow:
```bash
make start_airflow
```

Starting MLFlow:
```bash
mlflow ui -p <port>
```
