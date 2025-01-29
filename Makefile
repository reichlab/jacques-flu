#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = jacques_flu
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements/requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 jacques_flu
	isort --check --diff --profile black jacques_flu
	black --check --config pyproject.toml jacques_flu

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml jacques_flu



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) jacques_flu/dataset.py

.PHONY: hhs
hhs:
	$(PYTHON_INTERPRETER) jacques_flu/hhs_data.py


## Featurize Data
.PHONY: features
features:
	$(PYTHON_INTERPRETER) jacques_flu/features.py


## Splits data up into training and testing data
.PHONY: traintest
traintest:
	$(PYTHON_INTERPRETER) jacques_flu/train_test_split.py


## Splits data up into training and testing data
.PHONY: train
train:
	$(PYTHON_INTERPRETER) jacques_flu/modeling/train.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
