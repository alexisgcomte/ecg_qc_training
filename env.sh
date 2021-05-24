export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH;\
export AIRFLOW_HOME=$(pwd);\
export AIRFLOW_CONFIG=$AIRFLOW_HOME/airflow.cfg;\
export WEBSERVER_CONFIG=$AIRFLOW_HOME/webserver_config.py;\