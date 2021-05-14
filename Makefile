FOLDER_PATH= .

init_airflow:
	export LD_LIBRARY_PATH=/usr/local/lib:LD_LIBRARY_PATH;\
	export AIRFLOW_HOME=pwd;\
	ecgi AIRFLOW_HOME;\
	export AIRFLOW_CONFIG=AIRFLOW_HOME/airflow.cfg;\
	airflow db init;\
	airflow users create \
		--username admin \
		--firstname Firstname \
		--lastname Lastname \
		--role Admin \
		--email admin@admin.org;\
	airflow webserver