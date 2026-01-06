"""Input/output helpers for Matrix Market files.

This module provides functions to load a `.mtx` file into a sparse matrix
representation.  The `scipy.io.mmread` function is used under the hood.
"""

from typing import Union, IO
from pathlib import Path
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, isspmatrix


def load_mtx(source: Union[str, Path, IO]) -> csr_matrix:
    """Load a Matrix Market file into a CSR sparse matrix.

    Parameters
    ----------
    source: str or Path or file‑like object
        Path to the `.mtx` file or an open file handle.  When a file
        handle is provided, its contents are read directly.

    Returns
    -------
    scipy.sparse.csr_matrix
        The adjacency matrix of the network as a CSR matrix of floats.

    Raises
    ------
    ValueError
        If the loaded matrix is not square.
    """

    # Use mmread directly on the source; it accepts filenames and file handles
    matrix = mmread(source)
    if not isspmatrix(matrix):
        # Convert dense to sparse
        matrix = csr_matrix(matrix)
    # Ensure square adjacency
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.  Check your .mtx file.")
    return matrix.tocsr()


if __name__ == "__main__":
    # Self‑test: load a simple 2×2 matrix from a string
    import io
    mtx_data = """%%MatrixMarket matrix coordinate real general\n2 2 2\n1 2 1\n2 1 1\n"""
    f = io.StringIO(mtx_data)
    A = load_mtx(f)
    print(A.todense())