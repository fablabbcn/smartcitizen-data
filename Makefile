.PHONY: package release

all: package

test:
	pytest

clean:
	rm -rf dist/*
	rm -rf build/*

package: clean
	git add -p setup.py scdata/__init__.py
	RELEASE=$(python setup.py --version) && git commit -m "Version $RELEASE" && git tag -a v$RELEASE -m "Version $RELEASE"
	python setup.py sdist bdist_wheel

release:
	# Still testing
	git push
	git push --tags
