ROOT_DIR = /Users/sam/porfolio_workspace/Logistic-Regression-from-Scratch
TEST_DIR = $(ROOT_DIR)/tests
TOOLS_DIR = $(ROOT_DIR)/tools

install: 
	/bin/bash $(TOOLS_DIR)/install.sh

pipeline_file = $(TEST_DIR)/test_pipeline.py
pipeline_edits_file = $(TEST_DIR)/test_addsteps_pipeline.py
classifier_file = $(TEST_DIR)/test_classifier.py
optimizer_file = $(TEST_DIR)/test_optimizerandloss.py
utils_file = $(TEST_DIR)/test_utils.py


lint:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && pylint --disable=R,C $(TEST_DIR)/*.py"

test:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && coverage run -m pytest -vv $(TEST_DIR)/*.py"

format:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && black $(TEST_DIR)/*.py"

coverage: 
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "source .venv/bin/activate && coverage report -m"


formatting: install format
linting: install format lint
testing: install format lint test
full_coverage: install format lint test coverage