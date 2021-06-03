FOLDER_PATH= .
SRC_PATH=./dags
TEST_PATH=./tests

init_airflow:
	source $(FOLDER_PATH)/env.sh;\
	airflow db init;\
	airflow users create \
		--username admin \
		--firstname Firstname \
		--lastname Lastname \
		--role Admin \
		--email admin@admin.org;\

start_airflow:
	source $(FOLDER_PATH)/env.sh;\
	airflow scheduler & airflow webserver

test:
	pytest -s -vvv $(TEST_PATH)

coverage:
	pytest --cov=$(SRC_PATH) --cov-report html $(TEST_PATH) 