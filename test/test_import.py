import re

import cosmobench


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", cosmobench.__version__)
