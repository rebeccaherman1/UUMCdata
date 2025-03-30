# UUMC SCM and Data Generation
Unitless, Unrestricted, Markov-Consistent Random Static (or Time Series, beta) SCM and Data Generation [[paper]](https://doi.org/10.48550/arXiv.2405.13100).

Our work focuses on generation of linear additive Gaussian Structural Causal Models (SCM) given a causal graph. In addition to our proposed approach, our package supports other approaches examined in the UUMC paper for comparison:
| Method | Description |
| :-- | :-- |
| [UUMC](https://doi.org/10.48550/arXiv.2503.17037) | Procudes unitless, unrestricted, Markov-consistent SCMs. introduced here, recommended. |
| [unit-variance-noise](https://doi.org/10.48550/arXiv.1803.01422) | Draws coefficients uniformly from [-HIGH, -LOW] U [LOW, HIGH], and sets all noise variances to 1. Defaults LOW=.5, HIGH=2. |
| [iSCM](https://arxiv.org/abs/2406.11601) | Begins with UVN SCM generation. During data generation, the coefficients (and data) for each variable are standardized by the sample standard deviation of the generated data before moving on to the next variable in the topological order. |
| [IPA](http://jmlr.org/papers/v21/17-123.html) | Each variable is scaled down by the variance it would have had if its parents were independent. |
| [50-50](https://proceedings.mlr.press/v177/squires22a.html) | Begins with UVN SCM generation. The SCM is not complete until calling GEN_DATA. During data generation, data for each variable is generated first without noise, then the coefficients and data are scaled down to have a variance of 1/2, and noise with variance 1/2 is added before moving on to the next variable in the topological order. |
| [DaO](https://doi.org/10.48550/arXiv.2405.13100) | DAG Adaptation of the Onion Method. |
                  
To generate random data ([example](example.ipynb)):
1. Initialize a graph in a `CausalModel` or `tsCausalModel` object ([CausalModel.py](CausalModel.py)). This can be done:
   * randomly using Erdös-Rényi sampling
   * from a user-provided array where $a_{ji}=1 \Leftrightarrow X_j \rightarrow X_i$. For time series, $a_{ji\tau}=1 \Leftrightarrow X_j(t-\tau)\rightarrow X_i(t)$.
2. Call `gen_coefficients()` on the CausalModel using the options from the table above. This sets the coefficient matrix `A` and the noise vector `s`. (Time series UUMC SCM generation is available, but under development.)
3. Call `gen_data()` on the CausalModel, providing the number of samples. This returns a `Data` or `TimeSeires` object ([Data.py](Data.py)) which is also stored in the `data` attribute of the CausalModel.

[Var-](https://doi.org/10.48550/arXiv.2102.13647) and [R2-](https://proceedings.neurips.cc/paper_files/paper/2023/file/027e86facfe7c1ea52ca1fca7bc1402b-Paper-Conference.pdf)sortability can be examined by calling `sortability()` on the CausalModel. Large datasets over multiple SCMs can be generated using `CausalModel.gen_dataset()`, and [AnalysisPlotting.py](AnalysisPlotting.py) and [UUMC.ipynb](UUMC.ipynb) contain code that can be used to re-create figures from the UUMC paper.