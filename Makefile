.PHONY: install train eval run clean login check-venv

PYTHON := .venv/bin/python
UV := .venv/bin/uv

check-venv:
	@test -x $(PYTHON) || ( \
		echo "🔧 Creating .venv and installing uv..." && \
		python3 -m venv .venv && \
		.venv/bin/pip install -U pip && \
		.venv/bin/pip install uv \
	)

install: check-venv
	$(UV) pip install --editable .

login:
	$(PYTHON) scripts/hf_login.py

train: check-venv
	$(PYTHON) scripts/train_adapter.py

eval: check-venv
	$(PYTHON) scripts/eval_adapter.py

run: check-venv
	$(PYTHON) scripts/inference_example.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + || true
	rm -rf ./adapter
