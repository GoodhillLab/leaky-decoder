# Leaky decoder

![workflow](https://github.com/rob-the-bot/leaky-decoder-code/actions/workflows/tests.yml/badge.svg)

This repository contains the code accompanying the manuscript **Preventing Data Leakage in Neural Decoding**.
The `demo.ipynb` file includes a Jupyter Notebook that:

- Demonstrates a recommended method to prevent data leakage in decoding pipelines.
- Provides examples of decoding pipelines with and without data leakage.
- Shows the impact of data leakage on decoding performance.
- Illustrates how to use permutation tests (shuffle control) to detect data leakage in a decoding pipeline.

## Requirements

The code was written in Python 3.10, and has been tested on Python versions 3.10 to 3.14.
The required packages are listed in [`requirements.txt`](requirements.txt) and can be installed using pip:

```bash
pip install -r requirements.txt
```

## Figures

To reproduce all figures in the manuscript, use the code files in the [`figs`](figs/README.md) directory.
A more detailed README is available in that directory, specifying which figure panels each script generates.
