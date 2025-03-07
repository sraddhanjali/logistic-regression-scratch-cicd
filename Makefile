TEST_DIR = /Users/sam/porfolio_workspace/Logistic-Regression-from-Scratch/tests
install: 
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv $(TEST_DIR)/test_classifier.py
	python -m pytest -vv $(TEST_DIR)/test_pipeline.py
	python -m pytest -vv $(TEST_DIR)/test_addsteps_pipeline.py


format:
	black *.py

lint:
	pylint --disable=R,C hello.py

all: install lint test