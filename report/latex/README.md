# LaTeX report

`main.tex` is the paper source and `main.pdf` is the compiled artifact. Figures are
referenced directly from `report/figures/` and `RLVR/figures/`, so there are no copied
binary assets to keep in sync.

**Submission status: complete.** All scheduled benchmark, verifier, stylometric, and
persona-judge evaluations are incorporated. No result is marked pending.

Build from this directory:

```bash
tectonic main.tex
cp main.pdf ../REPORT.pdf
```

The copy step keeps the easy-to-find submission PDF at `report/REPORT.pdf` identical to
the compiled artifact.

Before submitting, regenerate the figures using the commands in `report/README.md`,
compile twice if citations or float placement changed, and inspect every rendered page.
The final PDF has been visually inspected page by page.
