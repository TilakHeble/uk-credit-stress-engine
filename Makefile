.PHONY: install panel test lint notebooks dashboard all

install:
	pip install -e ".[dev,dashboard]"

panel:
	python -m ukstress.ingest.build_panel

test:
	pytest -q

lint:
	ruff check src tests dashboards

notebooks:
	for nb in notebooks/*.ipynb; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $$nb; done

dashboard:
	streamlit run dashboards/app.py

all: panel test notebooks
