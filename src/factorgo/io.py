import numpy as np
import pandas as pd

from jax import numpy as jnp


def read_data(z_path, N_path, log, scale=True):
    """
    input z score summary stats:
    headers are ["snp", "trait1", "trait2", ..., "traitn"]

    input sample size file:
    one column of sample size (with header) which has the same order as above
    """

    # Read dataset (read as example frame)
    df_z = pd.read_csv(z_path, delimiter="\t", header=0)
    snp_col = df_z.columns[0]

    # drop the first column (axis = 1) and convert to nxp
    # snp_col = df_z.columns[0]
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


def write_results(output, EW2, EZ2, W_var, f_order):
    ordered_EW2 = EW2[:, f_order]
    ordered_EZ2 = EZ2[:, f_order]
    ordered_W_var = jnp.diagonal(W_var)[f_order]
    np.savetxt(f"{output}.EW2.tsv.gz", ordered_EW2.real, fmt="%s", delimiter="\t")
    np.savetxt(f"{output}.EZ2.tsv.gz", ordered_EZ2.real, fmt="%s", delimiter="\t")
    np.savetxt(f"{output}.W_var.tsv.gz", ordered_W_var.real, fmt="%s", delimiter="\t")
    return
