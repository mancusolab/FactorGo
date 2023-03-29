.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/factorgo.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/factorgo
    .. image:: https://readthedocs.org/projects/factorgo/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://factorgo.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/factorgo/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/factorgo
    .. image:: https://img.shields.io/pypi/v/factorgo.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/factorgo/
    .. image:: https://img.shields.io/conda/vn/conda-forge/factorgo.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/factorgo
    .. image:: https://pepy.tech/badge/factorgo/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/factorgo
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/factorgo

|Maintenance yes| |Star| |Data| |MIT license| |PyScaffold|

.. |PyScaffold| image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

.. |Maintenance yes| image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://github.com/mancusolab/FactorGo/pulse

.. |Star| image:: https://img.shields.io/github/stars/mancusolab/factorgo?style=social
        :alt: Github
        :target: https://github.com/mancusolab/factorgo

.. |MIT license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://lbesson.mit-license.org/

.. |Data| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7765048.svg
   :target: https://doi.org/10.5281/zenodo.7765048

=======
FactorGo
=======
FactorGo is a scalable variational factor analysis model that learns pleiotropic factors using GWAS summary statistics!

We present **Factor** analysis model in **G**\enetic ass\ **O**\ciation (FactorGo) to learn latent
pleiotropic factors using GWAS summary statistics. Our model is implemented using `Just-in-time` (JIT)
via `JAX <https://github.com/google/jax>`_ in python, which generates and compiles heavily optimized
C++ code in real time and operates seamlessly on CPU, GPU or TPU. FactorGo is a command line tool and
please see example below and full documentation.

For preprint, please see: 

    | https://www.medrxiv.org/content/10.1101/2023.03.27.23287801v1

|Model|_ | |Installation|_ | |Example|_ | |Notes|_ | |References|_ | |Support|_ | |Other Softwares|_

=================

.. _Model:
.. |Model| replace:: **Model**
FactorGo model
=================
FactorGo assumes the true genetic effect can be decomposed into latent pleiotropic factors.
Briefly, we model test statistics at $p$ independent variants from the ith GWAS $Z_i \\approx \\sqrt{N}_i \\hat{\\beta}_i$  as a
linear combination of $k$ shared latent variant loadings $L \\in R^{p \\times k}$  with trait-specific factor scores $f_i \\in R^{k \\times 1}$ as

$$Z_i = \\sqrt{N}_i \\beta_i + \\epsilon_i = \\sqrt{N}_i (L f_i + \\mu) + \\epsilon_i $$

where $N_i$ is the sample size for the $i^{th}$ GWAS , $\\mu$  is the intercept and $\\epsilon_i \\sim N(0, \\tau^{-1}I_p)$ reflects residual
heterogeneity in statistical power across studies with precision scalar .
Given $Z = \\{Z_i\\}^n_{i=1}$, and model parameters  $L$, $F$, $\\mu$, $\\tau$, we can compute the likelihood as

$$\\mathcal{L}(L, F, \\mu, \\tau |Z) = \\prod_i \\mathcal{N}_p ( \\sqrt{N_i} (L f_i + \\mu), \\tau^{-1} I_p)$$

To model our uncertainty in $L$, $F$, $\\mu$, we take a full Bayesian approach in the lower dimension latent space
similar to a Bayesian PCA model [1]_ as,

$$\Pr(F) = \\prod_{i=1}^{n} \\mathcal{N}_k (f_i | 0, I_k)$$

$$\Pr(L | \\alpha) = \\prod_{j=1}^{p} \\mathcal{N}_k (l^j | 0, diag(\\alpha^{-1}))$$

$$\Pr(\\mu) = \\mathcal{N}_p (\\mu | 0, \\phi^{-1} I_p)$$

where $\\alpha \\in R^{k \\times 1}_{>0} (\\phi > 0)$ controls the prior precision for variant loadings (intercept). To avoid overfitting,
and “shut off” uninformative factors when $k$ is misspecified, we use automatic relevance determination (ARD) [1]_
and place a prior over $\\alpha$ as

$$\Pr(\\alpha | \\alpha_a, \\alpha_b) = \\prod_{q=1}^{k} G(\\alpha_q | \\alpha_a, \\alpha_b)$$

$$\Pr(\\tau | \\tau_a, \\tau_b) = G(\\tau | \\tau_a, \\tau_b)$$

Lastly, we place a prior over the shared residual variance across GWAS studies as $\\tau \\sim G(a , b)$.
We impose broad priors by setting hyperparameters $\\phi = a_k = b_k= a_{\\tau} = b_{\\tau} = 10^{-5}$.


.. _Installation:
.. |Installation| replace:: **Installation**

Install ``factorgo``
=================
We recommend first create a conda environment and have `pip` installed.

.. code-block:: bash

   # download use http address
   git clone https://github.com/mancusolab/FactorGo.git
   # or use ssh agent
   git clone git@github.com:mancusolab/FactorGo.git

   cd factorgo
   pip install .


.. _Example:
.. |Example| replace:: **Example**

Example
=================
For iilustration, we use example data stored in `/example/data`,
including Z score summary statistics file and sample size file.

To run ``factorgo`` command line tool, we specify the following input files and flags:

* GWAS Zscore file: n20_p1k.Zscore.tsv.gz
* Sample size file: n20_p1k.SampleN.tsv
* -k 5: estimate 5 latent factors
* --scale: the snp columns of Zscore matrix is center and standardized
* -o: output directory and prefix

For all available flags, please use ``factorgo -h``.

.. code-block:: bash

   factorgo \
        ./example/data/n20_p1k.Zscore.tsv.gz \
        ./example/data/n20_p1k.SampleN.tsv \
        -k 5 \
        --scale \
        -o ./example/result/demo_test

The output contains five result files:

1. demo_test.Wm.tsv.gz: posterior mean of loading matrix W (pxk)
2. demo_test.Zm.tsv.gz:  posterior mean of factor score Z (nxk)
3. demo_test.Wvar.tsv.gz:  posterior variance of loading matrix W (kx1)
4. demo_test.Zvar.tsv.gz:  posterior variance of factor score Z (nxk)
5. demo_test.factor.tsv.gz:  contains the following three columns

   | a) factor index (ordered by R2),
   | b) posterior mean of ARD precision parameters,
   | c) variance explained by each factor (R2)


.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====
The default computation device for ``factorgo`` is CPU. To switch to GPU device, you can specify the platform (cpu/gpu/tpu) using the flag `-p gpu` 
for example:

.. code-block:: bash

   factorgo \
        ./example/data/n20_p1k.Zscore.tsv.gz \
        ./example/data/n20_p1k.SampleN.tsv \
        -k 5 \
        --scale \
        -p gpu \ # use gpu device
        -o ./example/result/demo_test
        
``factorgo`` uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compilation to achieve high-speed computation. However, there are some `issues <https://github.com/google/jax/issues/5501>`_ for JAX with Mac M1 chip. To solve this, users need to initiate conda using `miniforge <https://github.com/conda-forge/miniforge>`_, and then install ``factorgo`` using ``pip`` in the desired environment.

.. _References:
.. |References| replace:: **References**

References
==========
.. [1] Bishop, C.M. (1999). Variational principal components. 509–514.


.. _Support:
.. |Support| replace:: **Support**

Support
=======
Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/FactorGo/issues>`_.
If you have any questions or comments please contact zzhang39@usc.edu and/or nmancuso@usc.edu.


.. _OtherSoftwares:
.. |Other Softwares| replace:: **Other Softwares**

Other Softwares
==============

Please check out other software developed by `Mancuso Lab <https://www.mancusolab.com/>`_ and more to come:

* `SuShiE <https://github.com/mancusolab/sushie>`_: a Python software to fine-map causal SNPs, compute prediction weights, and infer effect size correlation across multiple ancestries.

* `MA-FOCUS <https://github.com/mancusolab/ma-focus>`_: a Bayesian fine-mapping framework using `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics across multiple ancestries to identify the causal genes for complex traits.

* `SuSiE-PCA <https://github.com/mancusolab/sushie>`_: a scalable Bayesian variable selection technique for sparse principal component analysis

* `twas_sim <https://github.com/mancusolab/twas_sim>`_: a Python software to simulate `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
