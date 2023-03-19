import logging

from src.factorgo import io
from src.factorgo.cli import get_logger

# import numpy.testing as nptest


Zscore_path = "./example/demo_n20_p1k.Zscore.gz"
N_path = "./example/demo_n20_p1k.SampleN.tsv"
output = "./example/demo"

log = get_logger(__name__, output)

log.setLevel(logging.INFO)

df_z, sampleN, sampleN_sqrt = io.read_data(Zscore_path, N_path, log, False)
