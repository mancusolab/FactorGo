import os
import subprocess

from utils import assert_array_eq


def test_cli_tool_writes_file():
    target_file = "../example/result/demo_old.factor.tsv.gz"
    output_file = "../example/result/demo_test.factor.tsv.gz"

    subprocess.run(
        [
            "factorgo",
            "./example/data/n20_p1k.Zscore.tsv.gz",
            "./example/data/n20_p1k.SampleN.tsv",
            "-k",
            "5",
            "--scale",
            "-o",
            "./example/result/demo_test",
        ],
        capture_output=False,
        text=False,
    )

    assert os.path.isfile(output_file)
    # TODO: not sure why raise error, not see difference
    assert assert_array_eq(target_file, output_file)
