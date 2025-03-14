TEST_DIR = /Users/sam/porfolio_workspace/Logistic-Regression-from-Scratch/tests

install: 
	pip install --upgrade pip &&\
		pip install -r requirements.txt

pipeline_file = $(TEST_DIR)/test_pipeline.py
pipeline_edits_file = $(TEST_DIR)/test_addsteps_pipeline.py
classifier_file = $(TEST_DIR)/test_classifier.py

lint:
	PYTHONPATH=$(TEST_DIR)/.. pylint --disable=R,C $(pipeline_file)
	PYTHONPATH=$(TEST_DIR)/.. pylint --disable=R,C $(pipeline_edits_file)
	PYTHONPATH=$(TEST_DIR)/.. pylint --disable=R,C $(classifier_file)

test:
	PYTHONPATH=$(TEST_DIR) coverage run -m pytest -vv $(pipeline_file)
	PYTHONPATH=$(TEST_DIR) coverage run -m pytest -vv $(pipeline_edits_file)
	PYTHONPATH=$(TEST_DIR) coverage run -m pytest -vv $(classifier_file)


format:
	black $(TEST_DIR)/*.py

coverage: 
	coverage report -m

all: install format lint test coverage