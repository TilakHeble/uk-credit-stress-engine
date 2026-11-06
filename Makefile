.PHONY: install panel test lint notebooks dashboard all

install:
	pip install -e ".[dev,dashboard]"

panel:
	python -m ukstress.ingest.build_panel

test:
	pytest -q

lint:
	ruff check src tests

notebooks:
	for nb in notebooks/*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $$nb; done

dashboard:
	streamlit run dashboards/app.py

all: panel test notebooks
