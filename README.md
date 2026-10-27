# Folie à Deux as a Cusp Catastrophe in Coupled Active Inference

Code and figures for:

Beasley, D. (2026). Folie à deux as a cusp catastrophe in coupled active inference: a dynamical systems account of shared psychotic disorder. *Frontiers in Psychiatry*. https://doi.org/10.3389/fpsyt.2026.1886942

## Correction (October 2026)

After publication the author found errors in the derivation in Section 5 and in the simulations in Section 6, and has asked the journal to publish a correction. The corrected code is in `correction-2026-10/`:

- `corrected_model.py`: the explicit two-person model, its equilibria (computed numerically and checked by residual), the exact threshold and fold, and the checks reported in the corrected Section 6. Its output is saved in `corrected_model_output.txt`.
- `corrected_figures.py`: regenerates the corrected Figure 1 and the new Figure 3 into `correction-2026-10/figures/`.

The files in the top folder are the code and figures from the published version. They are kept for the record and are superseded by `correction-2026-10/`. In particular, the parameter-space percentage produced by `folie_simulations.py` is withdrawn in the correction.

## Contents

- `correction-2026-10/`: corrected code, output and figures (current)
- `folie_simulations.py`: original simulations (superseded)
- `fig1_phase_diagram.png`, `fig2_trajectories.png`, `fig3_sensitivity.png`: original figures. Figure 2 is unchanged in the correction; Figures 1 and 3 are superseded.

## Requirements

Python 3, numpy, matplotlib. The original `folie_simulations.py` also uses scipy.

## Usage

```
python correction-2026-10/corrected_model.py
python correction-2026-10/corrected_figures.py
```

## License

MIT, see LICENSE file.

## Author

Dustin Beasley, Golden Gate University
dustybeas@gmail.com
