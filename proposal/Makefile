# Build the proposal. XeLaTeX is required: the author names contain Chinese
# characters and are typeset with xeCJK + Noto CJK TC.
#
#   make        build main.pdf AND the e-Campus submission copy
#   make clean  remove build artefacts (keeps the PDFs)
#   make purge  also remove main.pdf and the submission copy

LATEXMK ?= latexmk
MAIN    := main
# e-Campus filename rule: Proposal_<Group_id>.pdf . We are group 6.
GROUP   ?= 6
SUBMIT  := Proposal_$(GROUP).pdf

all: $(MAIN).pdf $(SUBMIT)

$(MAIN).pdf: $(MAIN).tex refs.bib figures/arch.tex
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

# The upload must be named Proposal_<Group_id>.pdf, so every build produces it.
$(SUBMIT): $(MAIN).pdf
	@cp -f $(MAIN).pdf $@
	@echo "ready to upload: $@"

submit: $(SUBMIT)

clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml

purge: clean
	rm -f $(MAIN).pdf $(SUBMIT)

.PHONY: all submit clean purge
