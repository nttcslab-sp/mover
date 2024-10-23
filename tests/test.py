import pytest

from mover.__main__ import cli


@pytest.mark.parametrize("strategy", ["fullset", "subset"])
def test(strategy):
    cli(
        [
            "--infile",
            "./example_files/*.json",
            "--outfile",
            f"./test_products/{strategy}.json",
            "--strategy",
            strategy,
        ],

    )
