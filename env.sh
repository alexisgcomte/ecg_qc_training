export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH;\
export AIRFLOW_HOME=$(pwd);\
export AIRFLOW_CONFIG=$AIRFLOW_HOME/airflow.cfg;\
export WEBSERVER_CONFIG=$AIRFLOW_HOME/webserver_config.py;\


# PostgreSQL env vars
export POSTGRES_HOST_URL=localhost
export POSTGRES_DATABASE=postgres
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres

# Airflow env vars
export POSTGRES_DATABASE_AIRFLOW=airflow_db

# Test variables
export TEST_PATH=$AIRFLOW_HOME/tests