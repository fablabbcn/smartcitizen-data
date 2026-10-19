# make package builds the source and wheel distributions in dist/ (requires the build package).
# make release publishes the version in setup.py as a GitHub release, which the release workflow
# uploads to PyPI. It must be run from the default branch once the version bump is merged, and
# requires the GitHub CLI (gh).
.PHONY: all package release clean

VERSION = $(shell sed -n -E "s/^ *version *= *['\"]([^'\"]+)['\"].*/\1/p" setup.py)
# Tag format of the previous releases (1.5.8 or v1.6.5)
TAG = $(shell git describe --tags --abbrev=0 2>/dev/null | grep -o '^v')$(VERSION)
BRANCH = $(shell git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|^origin/||')

all: package

package: clean
	python -m build

release:
	@test -n "$(BRANCH)" || { echo "No default branch: git remote set-head origin --auto"; exit 1; }
	@git fetch -q --tags origin
	@test "$$(git branch --show-current)" = "$(BRANCH)" || { echo "Releases are made from $(BRANCH)"; exit 1; }
	@test "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/$(BRANCH))" || { echo "$(BRANCH) is not origin/$(BRANCH)"; exit 1; }
	@git diff --quiet HEAD || { echo "Uncommitted changes"; exit 1; }
	@! git rev-parse -q --verify "refs/tags/$(TAG)" >/dev/null || { echo "$(TAG) is already released"; exit 1; }
# The release targets the verified commit, even if the branch has advanced since
	gh release create $(TAG) --target $$(git rev-parse HEAD) --title $(TAG) --generate-notes

clean:
	rm -rf dist build *.egg-info
