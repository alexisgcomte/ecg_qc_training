# ecg_qc_training


## I) Reminder of annotated segments

patient: PAT_6 / record:77 / channel: emg6+emg6- / time: 2020-12-18 13:00:00 à 2020-12-18 14:30:00
patient: PAT_4 / record: 11/ channel: ECG1+ECG1- / time: 2020-12-15 21:50:00 à 2020-12-15 22:50:00
patient: PAT_2 / record: _3 / channel:  EMG1+EMG1- / time : 2020-12-14 21:55:00 à 2020-12-14 23:05:00


## II) Individual scripts

### 1) import_ecg_segment

Allow to import from an edf file a certain window of time for a particular channel.

Exemple of use:
```python
python3 dags/tasks/import_ecg_segment.py -p PAT_6 -r 77 -s s1 -c ECG1+ECG1- -st '2020-12-18 13:00:00' -et '2020-12-18 14:30:00'
```

### 2) import_annotations

From the SQL database storing annotations, import them for a certain windows time and other patient parameters.

Exemple of use:
```python
python3 dags/tasks/import_annotations.py -p PAT_6 -r 77 -c ECG1+ECG1- -ids 2,3,4 -st '2020-12-18 13:00:00' -et '2020-12-18 14:30:00'
```

### 3) create_ecg_dataset

Combines import_ecg_segment and import_annotations.

Exemple of use:
```python
python3 dags/tasks/create_ecg_dataset.py -p PAT_6 -r 77 -s s1 -c ECG1+ECG1- -ids 2,3,4 -st '2020-12-18 13:00:00' -et '2020-12-18 14:30:00'
```
