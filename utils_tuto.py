import numpy as np
from qat.core.circuit_builder.matrix_util import get_predef_generator
def make_matrix(hamiltonian):
    mat = np.zeros((2**hamiltonian.nbqbits, 2**hamiltonian.nbqbits), np.complex_)
    for term in hamiltonian.terms:
        op_list = ["I"]*hamiltonian.nbqbits
        for op, qb in zip(term.op, term.qbits):
            op_list[qb] = op
        def mat_func(name): return np.identity(2) if name == "I" else get_predef_generator()[name]
        term_mat = mat_func(op_list[0])
        for op in op_list[1:]:
            term_mat = np.kron(term_mat, mat_func(op))
        mat += term.coeff * term_mat
    return mat
