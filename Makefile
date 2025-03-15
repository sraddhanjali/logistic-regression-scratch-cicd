ROOT_DIR := $(CURDIR)
TEST_DIR := $(ROOT_DIR)/tests
TOOLS_DIR := $(ROOT_DIR)/tools

install: 
	/bin/bash $(TOOLS_DIR)/install.sh

pipeline_file = $(TEST_DIR)/test_pipeline.py
pipeline_edits_file = $(TEST_DIR)/test_addsteps_pipeline.py
classifier_file = $(TEST_DIR)/test_classifier.py
optimizer_file = $(TEST_DIR)/test_optimizerandloss.py
utils_file = $(TEST_DIR)/test_utils.py

VENV_DIR := $(ROOT_DIR)/.venv
ACTIVATE := source $(VENV_DIR)/bin/activate

lint:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "$(ACTIVATE) && pylint --disable=R,C $(TEST_DIR)/*.py"

test:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "$(ACTIVATE) && coverage run -m pytest -vv $(TEST_DIR)/*.py"

format:
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "$(ACTIVATE) && black $(TEST_DIR)/*.py"

coverage: 
	PYTHONPATH=$(ROOT_DIR) /bin/bash -c "$(ACTIVATE) && coverage report -m"


formatting: install format
linting: install format lint
testing: install format lint test
full_coverage: install format lint test coverage