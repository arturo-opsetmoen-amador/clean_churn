#!/bin/bash
pytest /home/arturo_docker/clean_churn/tests/churn_script_logging_and_tests.py
pylint /home/arturo_docker/clean_churn/tests/churn_script_logging_and_tests.py > /home/arturo_docker/clean_churn/logs/pylint.log
python /home/arturo_docker/clean_churn/churn_library.py