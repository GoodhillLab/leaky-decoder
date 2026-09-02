# Environment

- install miniforge
- `conda env create -f environment.yml` (blas can be optionally changed from `mkl` to `openblas` if you experience performance issues with an AMD CPU, see more information [here](https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation))
- Use `run_all_figs.sh` (either with bash, powershell or Command Prompt) to run all figures.

# Paper figures

## Figure 1

Illustration showing data leakage in a decoding pipeline.

## Supervised feature selection

Run [`fig2.py`](fig2.py) and [`fish_decoding_2f.py`](fish_decoding_2f.py) to reproduce panels in Fig. 2, Fig. S1{a,b} and Fig. S2.

## Supervised dimensionality reduction

- Run [`fig3.py`](fig3.py) to reproduce Fig. 3{a,b,c,d} and Fig. S3.
- RUn [`figS4.py`](figS4.py) to reproduce Fig. S4.
- Run [`fish_decoding_3e.py`](fish_decoding_3e.py) to reproduce Fig. 3e.
- Run [`cebra_decoding.py`](cebra_decoding.py) with `python cebra_decoding.py --condition supervised` to reproduce Fig. 3f. Additional dataset required:
    + [Public rat's hippocampus dataset](https://doi.org/10.6080/K0862DC5) from Grosmark and Buzsáki (2016). We use the preprocessed data version from [`pi-vae`](https://github.com/zhd96/pi-vae?tab=readme-ov-file#data). Please place the file `Achilles_data.mat` under the [`data/`](data/) folder.
- Run [`pls_mse.py`](pls_mse.py) to reproduce Fig. 3{g,h}. Requires long runtime.

## Unsupervised feature selection

Run [`run_fig4.sh`](run_fig4.sh) to reproduce Figure 4 and Figure S5.
Requires long runtime.

## Unsupervised dimensionality reduction

### Bias-variance decomposition of PCR

- `python pcr_mse_n_samples.py` to reproduce Fig. 5 {b,c,d}
- `python bias_variance_decomp.py --compute` then `python bias_variance_decomp.py --plot` to reproduce Fig. 5e.
- `python pcr_mse.py` to reproduce Fig. 5{f,g,h}.
- `python bias_variance_decomp_n_features.py --compute` then `python bias_variance_decomp_n_features.py --plot` to reproduce Fig. 5{i,j}.

### GPFA

- Run [`gpfa_run_sim.py`](gpfa_run_sim.py), it generates the data [`gpfa_simulation_results.h5`](gpfa_simulation_results.h5) for Fig. 6{a,b,c}.
- Run [`gpfa.ipynb`](gpfa.ipynb) to reproduce Fig. 6{a,b,c}.

### Mouse visual cortex data

Here we compare the decoding results on calcium imaging dataset from mouse visual cortex.
For convinience, preprocessed data file is provided for plotting.
They can be reproduced by running different branches of [stringer-et-al-2019](https://github.com/rob-the-bot/stringer-et-al-2019).

- Run first and second code blocks in [`v1_comp.py`](v1_comp.py) to generate Fig. 6d, which reproduces Fig. S2I in [String et al., 2021](https://doi.org/10.1016/j.cell.2021.03.042).
- Run the rest of the code in [`v1_comp.py`](v1_comp.py) to generate Fig. 6e and Fig. S7.

### Rat hippocampal data

Run `python cebra_decoding.py --condition unsupervised` to reproduce Fig. 6f.

## leakage through temporal autocorrelation

- Run [`calcium_data_decoding.py`](calcium_data_decoding.py) to reproduce Fig. 7 panel a,d.
Requires long runtime.
- Run [`cv_plot.py`](cv_plot.py) to reproduce Fig. 7 panel b,c
- Run [`temporal.py`](temporal.py) to reproduce Fig. 7 panel e&ndash;h.
