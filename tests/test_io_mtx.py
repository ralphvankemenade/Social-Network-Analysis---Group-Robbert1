"""Unit tests for Matrix Market I/O utilities."""

import io
import pytest
from dss.utils.io_mtx import load_mtx


def test_load_mtx_square():
    mtx = """%%MatrixMarket matrix coordinate real general\n2 2 2\n1 2 1\n2 1 1\n"""
    A = load_mtx(io.StringIO(mtx))
    assert A.shape == (2, 2)
    assert A.nnz == 2


def test_load_mtx_non_square():
    mtx = """%%MatrixMarket matrix coordinate real general\n2 3 1\n1 2 1\n"""
    with pytest.raises(ValueError):
        load_mtx(io.StringIO(mtx))