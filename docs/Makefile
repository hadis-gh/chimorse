SPHINXOPTS ?= -W
SPHINXBUILD ?= python -m sphinx
SOURCEDIR = .
BUILDDIR = _build

.PHONY: html clean

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
