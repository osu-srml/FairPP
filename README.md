This is the code for our paper `Addressing Polarization And Unfairness In Performative Prediction`.

Run the notebooks below to reproduce each experiment:

- `Examples.ipynb`: reproduces examples in main paper.
- `gaussian_clf_exp.ipynb`: reproduces every synthetic Gaussian classification experiment, from trajectory generation to fairness/utility figures.
- `credit_exp.ipynb`: loads the credit dataset, runs the retention/fairness baselines, and exports the reported plots.
- `income_exp.ipynb`: runs the ACS Income pipeline end-to-end, including sampling, training, and visualization of the main metrics.
- `mnist_exp.ipynb`: evaluates the MNIST group-split setting with strategic behavior toggles.
- `Strat_exp.ipynb`: studies strategic manipulation in credit data setting.
- `Strat_k_delayed_exp.ipynb`: extends the strategic experiments to the K-delayed feedback model.
- `regression_exp.ipynb`: reproduces the regression version of the Gaussian experiments.
- `regression_e5.ipynb`: focuses on Example 5 from the paper, illustrating the variance-regularized objective.
- `regression_exp_multi.ipynb`: covers the multi-group regression simulations and associated fairness metrics.

The paper is mainly theoretical. If you take interest, check the paper [here](https://arxiv.org/pdf/2406.16756)
