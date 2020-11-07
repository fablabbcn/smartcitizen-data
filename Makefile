.PHONY: package release

all: package

package:
# 	rm dist/*
	python setup.py sdist bdist_wheel

release: package
	twine upload dist/*

clean:
	rm dist/*
	rm -rf build/*