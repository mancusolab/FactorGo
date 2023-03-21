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


    Add a short description here!


A longer description of your project goes here...

|Installation|_ | |Example|_ | |Notes|_ | |References|_ | |Support|_

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
As an example, we use example data stored in `/example`,
including one Z score summary statistics and sample size.
To run factorgo as a command line tool:

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

| a) factor index (ordered by R2),
| b) posterior mean of ARD precision parameters,
| c) variance explained by each factor ($R^2$)


.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====


.. _References:
.. |References| replace:: **References**

References
==========


.. _Support:
.. |Support| replace:: **Support**

Support
=======


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
