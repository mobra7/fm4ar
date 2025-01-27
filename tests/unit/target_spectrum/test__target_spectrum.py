"""
Tests for `fm4py.importance_sampling.target_spectrum`.
"""

from pathlib import Path

import numpy as np
import pytest

from fm4ar.target_spectrum import load_target_spectrum
from fm4ar.utils.hdf import save_to_hdf


@pytest.fixture
def path_to_target_spectrum(tmp_path: Path) -> Path:
    """
    Fixture for the path to the target spectrum.
    """

    file_path = tmp_path / "target_spectrum.hdf"
    save_to_hdf(
        file_path=file_path,
        wlen=np.array([1, 2, 3]),
        flux=np.array([[4, 5, 6], [7, 8, 9]]),
        error_bars=np.array([[1, 1, 1], [2, 2, 2]]),
        theta=np.array([[0, 0, 0], [1, 1, 1]])
    )
    return file_path


def test__load_target_spectrum(path_to_target_spectrum: Path) -> None:
    """
    Test `load_target_spectrum()`.
    """

    target = load_target_spectrum(file_path=path_to_target_spectrum, index=0)
    assert np.allclose(target["wlen"], [1, 2, 3])
    assert np.allclose(target["flux"], [4, 5, 6])
    assert np.allclose(target["error_bars"], [1, 1, 1])
    assert np.allclose(target["theta"], [0, 0, 0])

    target = load_target_spectrum(file_path=path_to_target_spectrum, index=1)
    assert np.allclose(target["wlen"], [1, 2, 3])
    assert np.allclose(target["flux"], [7, 8, 9])
    assert np.allclose(target["error_bars"], [2, 2, 2])
    assert np.allclose(target["theta"], [1, 1, 1])
