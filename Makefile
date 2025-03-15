ROOT_DIR = /Users/sam/porfolio_workspace/Logistic-Regression-from-Scratch
TEST_DIR = $(ROOT_DIR)/tests
TOOLS_DIR = $(ROOT_DIR)/tools

install: 
	/bin/bash $(TOOLS_DIR)/install.sh

pipeline_file = $(TEST_DIR)/test_pipeline.py
pipeline_edits_file = $(TEST_DIR)/test_addsteps_pipeline.py
classifier_file = $(TEST_DIR)/test_classifier.py


lint:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && pylint --disable=R,C $(pipeline_file)"
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && pylint --disable=R,C $(pipeline_edits_file)"
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && pylint --disable=R,C $(classifier_file)"

test:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && coverage run -m pytest -vv $(pipeline_file)"
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && coverage run -m pytest -vv $(pipeline_edits_file)"
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && coverage run -m pytest -vv $(classifier_file)"


format:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && black $(TEST_DIR)/*.py"

coverage: 
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && coverage report -m"


installing: install
formatting: install activate format
linting: install activate format lint
testing: install activate format lint test
full_coverage: install format lint test coverage