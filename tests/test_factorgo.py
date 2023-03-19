import logging

# import numpy.testing as nptest
import jax.random as rdm

from src.factorgo import infer, io, util
from src.factorgo.cli import get_logger


class args:
    k = 5
    verbose = False
    platform = "cpu"
    seed = 123456789
    Zscore_path = "../example/demo_n20_p1k.Zscore.gz"
    N_path = "../example/demo_n20_p1k.SampleN.tsv"
    output = "../example/demo_test"
    scale = True
    elbo_tol = 1e-3
    max_iter = 100000
    hyper = None
    init_factor = "random"
    rate = 20


log = get_logger(__name__, args.output)

log.setLevel(logging.INFO)

log = get_logger(__name__, args.output)
if args.verbose:
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

# setup to use either CPU (default) or GPU
util.set_platform(args.platform)

# ensure 64bit precision (default use 32bit)
util.update_x64(True)

# init key (for jax), split into 2 chunk
key = rdm.PRNGKey(args.seed)
key, key_init = rdm.split(key, 2)

log.info("Loading GWAS effect size and standard error.")
B, sampleN, sampleN_sqrt = io.read_data(args.Zscore_path, args.N_path, log, args.scale)
log.info("Finished loading GWAS summary statistics and sample size.")

n_studies, p_snps = B.shape
log.info(f"Found N = {n_studies} studies, P = {p_snps} SNPs")

# number of factors
k = args.k
log.info(f"User set K = {k} latent factors.")

# set options for stopping rule
options = infer.Options(args.elbo_tol, args.max_iter)

# set 5 hyper-parameters: otherwise use default 1e-5
if args.hyper is not None:
    util.set_hyper(args.hyper)

log.info(
    f"""set parameters
      halpha_a: {infer.HyperParams.halpha_a},
      halpha_b: {infer.HyperParams.halpha_b},
      htau_a: {infer.HyperParams.htau_a},
      htau_b: {infer.HyperParams.htau_b},
      hbeta: {infer.HyperParams.hbeta}
    """
)

W_m, W_var, Z_m, Z_var, f_info, f_order, ordered_W_m, ordered_Z_m = infer.fit(
    B, args, k, key_init, log, options, sampleN, sampleN_sqrt
)

log.info("Writing results.")
io.write_results(args.output, f_info, ordered_Z_m, ordered_W_m, W_var, Z_var, f_order)
log.info("Finished. Goodbye.")
