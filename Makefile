PYTHON ?= python3

.PHONY: smoke test verify distribution

smoke:
	$(PYTHON) scripts/iteration_gate.py smoke

test verify:
	$(PYTHON) scripts/verify.py

distribution:
	$(PYTHON) scripts/check_distribution.py
