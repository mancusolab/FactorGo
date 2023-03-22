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

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=======
FactorGo
=======


    FactorGo is a scalable variational factor analysis model that learns pleiotropic factors using GWAS summary statistics !


Here we present **Factor** analysis model in **G**\enetic ass\ **O**\ciation (FactorGo) to learn latent
pleiotropic factors using GWAS summary statistics. Our model is implemented using `Just-in-time` (JIT)
via `JAX <https://github.com/google/jax>`_ in python, which generates and compiles heavily optimized
C++ code in real time and operates seamlessly on CPU, GPU or TPU. FactorGo is a command line tool and
please see example below and full documentation.

For preprint, please see: ...

|Model|_ | |Installation|_ | |Example|_ | |Notes|_ | |Support|_

.. _Model:
.. |Model| replace:: **Model**

FactorGo model
=================
FactorGo assumes the true genetic effect can be decomposed into latent pleiotropic factors.
Briefly, we model test statistics at p independent variants from the ith GWAS Zi Ni i  as a
linear combination of k shared latent variant loadings L Rpk  with trait-specific factor scores fiRk1 as

.. math::
   something

where Ni is the sample size for the ith GWAS ,  is the intercept and i N(0, -1Ip) reflects residual
heterogeneity in statistical power across studies with precision scalar .
Given Z ={Zi }n i=1, and model parameters  L, F, , , we can compute the likelihood as

$$\\mathcal{L}(L, F, \\mu, \\tau |Z) = \\prod_i \\mathcal{N}_p ( \\sqrt{N_i} (L f_i + \\mu), \\tau^{-1} I_p)$$

To model our uncertainty in L, F,  , we take a full Bayesian approach in the lower dimension latent space
similar to a Bayesian PCA model16 as,

.. math::
    \Pr(F) &= \prod_{i=1}^{n} \mathcal{N}_k (f_i \ | \ 0, I_k)\\
    \Pr(\mu) &= \mathcal{N}_p (\mu \ | \ 0, \phi^{-1} I_p)\\
    \Pr(L \ | \ \alpha) &= \prod_{j=1}^{p} \mathcal{N}_k (l^j \ | \ 0, diag(\alpha^{-1}))\\

where Rk1>0 (> 0) controls the prior precision for variant loadings (intercept). To avoid overfitting,
and “shut off” uninformative factors when k is misspecified, we use automatic relevance determination (ARD)16
and place a prior over  as

.. math::
    \Pr(\alpha \ | \ \alpha_a, \alpha_b) &= \prod_{q=1}^{k} \Gamma(\alpha_q \ | \ \alpha_a, \alpha_b) \\
    \Pr(\tau \ | \ \tau_a, \tau_b) &= \Gamma(\tau \ | \ \tau_a, \tau_b)

Lastly, we place a prior over the shared residual variance across GWAS studies as G(a , b).
We impose broad priors by setting hyperparameters = ak = bk= a =b = 10-5.

.. _Installation:
.. |Installation| replace:: **Installation**

Install factorgo
=================
something goes here

.. code-block::

   # download use http address
   git clone https://github.com/mancusolab/factorgo.git
   # or use ssh agent
   git clone https://github.com/mancusolab/FactorGo.git

   cd factorgo
   pip install -e .


.. _Example:
.. |Example| replace:: **Example**

Example
=================
For iilustration, we use example data stored in `/example`,
including Z score summary statistics file and sample size file.

To run factorgo command line tool:

* Zscore file: n20_p1k.Zscore.gz
* Sample size file: n20_p1k.SampleN.tsv
* --k 5: estimate 5 latent factors
* --scale: the snp columns of Zscore matrix is center and standardized
* -o: output directory and prefix

.. code-block::

   factorgo \
        ./example/n20_p1k.Zscore.gz \
        ./example/n20_p1k.SampleN.tsv \
        -k 5 \
        --scale \
        -o ./example/demo_test

The output contains five files:

1. demo_test.Wm.tsv.gz: posterior mean of loading matrix W (pxk)

2. demo_test.Zm.tsv.gz:  posterior mean of factor score Z (nxk)

3. demo_test.W_var.tsv.gz:  posterior variance of loading matrix W (kx1)

4. demo_test.Z_var.tsv.gz:  posterior variance of factor score Z (nxk)

5. demo_test.factor.tsv.gz:  contains three columns,

| a) factor index (ordered by :math:`$R^2$`),
| b) posterior mean of ARD precision parameters,
| c) variance explained by each factor (:math:`$R^2$`)


.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====
something about change precision 64bits and platform

.. _Support:
.. |Support| replace:: **Support**

Support
=======
Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/FactorGo/issues>`_.
If you have any questions or comments please contact zzhang39@usc.edu and/or nmancuso@usc.edu.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
