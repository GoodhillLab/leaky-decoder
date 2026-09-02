conda activate leaky

# Figure 2 and Figure S2
python fig2.py
python fish_decoding_2f.py

# Figure 3, S3, S4
python fig3.py
python figS3.py
python fish_decoding_3e.py
python cebra_decoding.py --condition supervised
python pls_mse.py # long runtime!

# Figure 4
bash run_fig4.sh

# Figure 5, this section takes a long time to run!

# panel b-d
python pcr_mse_n_samples.py

# panel e (and Fig. S6). One h5 store per covariance structure; --compute skips
# a store that already exists, so pass --force only when you mean to replace it.
python bias_variance_decomp.py --compute --sigma-types random_power_law
python bias_variance_decomp.py --plot --sigma-types random_power_law spiked

# panel f-h
python pcr_mse.py

# panel i, j
python bias_variance_decomp_n_features.py --compute
python bias_variance_decomp_n_features.py --plot

# panel k-m
python gpfa_run_sim.py
# then run the gpfa.ipynb notebook

# panel n, o, decoding result for Stringer et al. 2021 and Fig. S7
python v1_comp.py

# panel p, decoding result for hippocampus data
python cebra_decoding.py --condition unsupervised

# leakage through temporal autocorrelation (Figure 6)
python calcium_data_decoding.py
python cv_plot.py
python temporal.py
