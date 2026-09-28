# LaTeX report

`main.tex` is the paper source and `main.pdf` is the compiled artifact. Figures are
referenced directly from `report/figures/` and `RLVR/figures/`, so there are no copied
binary assets to keep in sync.

Build from this directory:

```bash
tectonic main.tex
```

Before submitting, regenerate the figures using the commands in `report/README.md`,
compile twice if citations or float placement changed, and inspect every rendered page.
The independent AI-judge result for the curriculum RLVR checkpoint is still pending;
the paper deliberately treats the latest checkpoint as provisional until it is added.
