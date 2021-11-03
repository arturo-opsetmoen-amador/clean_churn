#!/bin/bash
pytest /clean_churn/tests/churn_script_logging_and_tests.py
python /clean_churn/churn_library.py
pylint /clean_churn/tests/churn_script_logging_and_tests.py > /clean_churn/logs/pylint_tests.log
pylint /clean_churn/churn_library.py > /clean_churn/logs/pylint_main_library.log
