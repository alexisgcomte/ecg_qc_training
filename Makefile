FOLDER_PATH= .

init_airflow:
	source $(FOLDER_PATH)/source.sh;\
	airflow db init;\
	airflow users create \
		--username admin \
		--firstname Firstname \
		--lastname Lastname \
		--role Admin \
		--email admin@admin.org;\

start_airflow:
	source $(FOLDER_PATH)/source.sh;\
	airflow scheduler -D;\
	airflow webserver;\
