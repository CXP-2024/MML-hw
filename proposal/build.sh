#!/usr/bin/env bash
set -euo pipefail

pdflatex StreamAvatar_proposal.tex
bibtex StreamAvatar_proposal
pdflatex StreamAvatar_proposal.tex
pdflatex StreamAvatar_proposal.tex
