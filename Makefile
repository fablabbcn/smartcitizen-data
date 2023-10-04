.PHONY: package release

all: package

test:
	pytest

package:
	rm -rf dist/*
	rm -rf build/*
	python setup.py sdist bdist_wheel

release: package
	git tag -a v$(python setup.py --version)
	twine upload dist/*

clean:
	rm dist/*
	rm -rf build/*