ROOT_DIR = /Users/sam/porfolio_workspace/Logistic-Regression-from-Scratch
TEST_DIR = $(ROOT_DIR)/tests
TOOLS_DIR = $(ROOT_DIR)/tools

install: 
	/bin/bash tools/install.sh

pipeline_file = $(TEST_DIR)/test_pipeline.py
pipeline_edits_file = $(TEST_DIR)/test_addsteps_pipeline.py
classifier_file = $(TEST_DIR)/test_classifier.py

lint:
	PYTHONPATH=$(ROOT_DIR) pylint --disable=R,C $(pipeline_file)
	PYTHONPATH=$(ROOT_DIR) pylint --disable=R,C $(pipeline_edits_file)
	PYTHONPATH=$(ROOT_DIR) pylint --disable=R,C $(classifier_file)

test:
	PYTHONPATH=$(ROOT_DIR) coverage run -m pytest -vv $(pipeline_file)
	PYTHONPATH=$(ROOT_DIR) coverage run -m pytest -vv $(pipeline_edits_file)
	PYTHONPATH=$(ROOT_DIR) coverage run -m pytest -vv $(classifier_file)


format:
	PYTHONPATH=$(ROOT_DIR) black $(TEST_DIR)/*.py

coverage: 
	PYTHONPATH=$(ROOT_DIR) coverage report -m


installing: install
formatting: install format
linting: install format lint
testing: install format lint test
full_coverage: install format lint test coverage