from logging import Logger

import numpy as np
import pandas as pd

from jax import numpy as jnp


def read_data(z_path: str, N_path: str, log: Logger, scale: bool = True):
    """
    input z score summary stats:
    headers are ["snp", "trait1", "trait2", ..., "traitn"]

    input sample size file:
    one column of sample size (with header) which has the same order as above
    """

    # Read dataset (rows are SNPs)
    df_z = pd.read_csv(z_path, delimiter="\t", header=0)
    snp_col = df_z.columns[0]

    # drop the first column (axis = 1) and convert to nxp
    df_z.drop(labels=[snp_col], axis=1, inplace=True)
    df_z = df_z.astype("float").T

    if scale:
        df_z = df_z.subtract(df_z.mean())
        df_z = df_z.divide(df_z.std())
        log.info("Scale SNPs to mean zero and sd 1")

    # convert to numpy/jax device-array (n,p)
    df_z = jnp.array(df_z)

    # read sample size file and convert str into numerics, convert to nxp matrix
    df_N = pd.read_csv(N_path, delimiter="\t", header=0)
    df_N = df_N.astype("float")

    # convert sampleN (a file with one column and header)to arrays
    N_col = df_N.columns[0]
    sampleN = df_N[N_col].values
    sampleN_sqrt = jnp.sqrt(sampleN)

    return df_z, sampleN, sampleN_sqrt


def write_results(output, f_info, ordered_Z_m, ordered_W_m, W_var, Z_var, f_order):
    n, k = ordered_Z_m.shape

    ordered_W_var = jnp.diagonal(W_var)[f_order]

    Z_var_diag = np.zeros((n, k))
    for i in range(n):
        Z_var_diag[i] = jnp.diagonal(Z_var[i])
    ordered_Z_var_diag = Z_var_diag[:, f_order]

    np.savetxt(f"{output}.factor.tsv.gz", f_info, fmt="%s", delimiter="\t")
    np.savetxt(f"{output}.Zm.tsv.gz", ordered_Z_m.real, fmt="%s", delimiter="\t")
    np.savetxt(f"{output}.Wm.tsv.gz", ordered_W_m.real, fmt="%s", delimiter="\t")
    np.savetxt(
        f"{output}.Zvar.tsv.gz", ordered_Z_var_diag.real, fmt="%s", delimiter="\t"
    )
    np.savetxt(f"{output}.Wvar.tsv.gz", ordered_W_var.real, fmt="%s", delimiter="\t")
    return
