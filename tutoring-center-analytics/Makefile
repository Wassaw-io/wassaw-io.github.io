.PHONY: all data analysis survival risk uplift forecast monitor pooling retrieval model test clean

all: data analysis survival risk uplift forecast monitor pooling retrieval model test

data:
	python src/generate_data.py
	python src/generate_students.py

analysis:
	python src/analyze.py

survival:
	python src/survival.py

risk:
	python src/risk_model.py

uplift:
	python src/uplift.py

forecast:
	python src/forecast.py

monitor:
	python src/monitoring.py

pooling:
	python src/hierarchical.py

retrieval:
	python src/retrieval.py

model:
	python src/score.py train
	python src/score.py score
	python src/score.py check

test:
	python -m pytest tests/ -q

clean:
	rm -rf charts/*.png results*.json results.md data/*.csv data/*.json \
	       models/ src/__pycache__ tests/__pycache__ .pytest_cache
