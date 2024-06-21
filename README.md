This repository hosts the code to reproduce the results in the paper 

Fabian Falck*, Ziyu Wang* and Chris Holmes, *Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective*, in ICML 2024.

For the moment we are providing the LM samples and plotting scripts. The remaining scripts are 
being cleaned up and and will be released shortly.

## Result plots

The plots in the paper can be reproduced using `agg_plot.py` as exemplified below.  In all 
commands below, you need to update `--root_path` to point to the root directory of the LM samples, 
which can be downloaded from [this link](https://drive.google.com/file/d/1N-hOMuIYgsZpmLtXYnRVP-lAZbvZ7yLj/view?usp=sharing).

```sh
# Bernoulli MP check
python agg_plot.py --plot bernoulli-mp-check --mpc_models gpt-3d gpt-4 llama mistral --mpc_n 20 50 --bern_N_ratio 0.5 2  # Fig. 3 and Fig. 7 (a,b,e,f)
python agg_plot.py --plot bernoulli-mp-check --mpc_models gpt-3b gpt-3d gpt-3.5 --mpc_n 20 50 100 --bern_N_ratio 0.5 2   # Fig. 8
python agg_plot.py --plot bernoulli-mp-check --mpc_models gpt-3b gpt-3b-ft --mpc_n 50 --bern_N_ratio 2  # Fig. 13 (c,d)
# Gaussian MP check
python agg_plot.py --plot gaussian-mp-check  --mpc_n 20 50  --mpc_N_ratio 0.5 2   # Fig.9 and 10 (a-d) in a different layout
# Synthetic NLP experiment (Fig. 5 and Fig. 13 (a,b))
python agg_plot.py --plot nlp-mp-check 
# Scaling of epistemic uncertainty (Fig. 6 and Fig. 12)
python agg_plot.py --plot bernoulli-eu
```

There may be a very small variation across runs due to the randomness in bootstrapping.
