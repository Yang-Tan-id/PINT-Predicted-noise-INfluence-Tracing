.PHONY: install test validate

install:
	python -m pip install -e .

test:
	python -m unittest discover -s tests -v

validate:
	python -m pint.cli validate
