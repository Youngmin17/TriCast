# One gate for local runs, CI and coding agents: the CPU and lint part of AGENTS.md §6.
#   make check                     lint + CPU tests
#   make test PYTEST_ARGS="-x -k mma"
PYTHON ?= python3

.PHONY: install-cpu lint test test-app gpu check build ci

# CPU-only environment at the versions the cluster suite passed with. Editable install:
# tests read the git SHA and config/ relative to the source tree.
install-cpu:
	$(PYTHON) -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[dev,eval]" transformers==4.55.2 lm_eval==0.4.13 datasets==4.8.5 ruff==0.16.9 \
		build twine

lint:
	$(PYTHON) -m ruff check .

# CPU tests never need the network (HF loaders are monkeypatched); offline makes that explicit.
test:
	HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MKL_CBWR=COMPATIBLE $(PYTHON) -m pytest -q $(PYTEST_ARGS)

# TriCast Studio tests (needs the app extra: pip install -e ".[app]"); separate from the CPU suite.
test-app:
	HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MKL_CBWR=COMPATIBLE $(PYTHON) -m pytest app/tests -q -p no:cacheprovider $(PYTEST_ARGS)

# Triton parity tests. Without CUDA + triton every test is skipped, so a green run proves nothing there.
gpu:
	$(PYTHON) -m pytest tests/gpu -q $(PYTEST_ARGS)

check: lint test

build:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

# The three CI jobs (lint, test, build) run locally, after `make install-cpu`.
ci: lint test build
