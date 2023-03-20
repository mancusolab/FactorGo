import numpy.testing as nptest
import pandas as pd


def assert_array_eq(target_path, queue_path, rtol=1e-5):
    target = pd.read_csv(target_path, delimiter="\t", header=None).to_numpy()
    queue = pd.read_csv(queue_path, delimiter="\t", header=None).to_numpy()

    nptest.assert_allclose(target, queue, rtol=rtol)
