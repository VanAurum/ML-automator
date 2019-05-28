.PHONY: docs
init:
	pip install pipenv --upgrade
	pipenv install --dev

test:
	python -m unittest discover

coverage:
	coverage run --source=./test -m unittest discover

publish:
	pip install 'twine>=1.5.0'
	python setup.py sdist bdist_wheel
	twine upload dist/*
	rm -fr build dist .egg requests.egg-info
