import numpy as np
import scipy.special
from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered

from sslab_txz.fp_theory.modes import ModeBasis, ScalarLGMode, ScalarModeBasis


class ScalarModeOperator(BosonOperator):
    def toarray(self, basis: ModeBasis):

        d = len(basis)
        matrix = np.zeros((d, d), dtype='complex')

        for term, coeff in normal_ordered(self).terms.items():
            modes, actions = np.array(term).reshape(-1, 2).T
            quanta_change = 2 * actions - 1

            plus_mode_inds = (modes == 1)
            np_shift = quanta_change[plus_mode_inds].sum()
            np_raise = actions[plus_mode_inds].sum()
            np_lower = np_raise - np_shift

            minus_mode_inds = (modes == 2)
            nm_shift = quanta_change[minus_mode_inds].sum()
            nm_raise = actions[minus_mode_inds].sum()
            nm_lower = nm_raise - nm_shift

            def matrix_elt_func(nplus, nminus):
                return coeff * np.sqrt(np.prod([
                    scipy.special.poch(nplus - np_lower + 1, np_lower),
                    scipy.special.poch(nplus - np_lower + 1, np_raise),
                    scipy.special.poch(nminus - nm_lower + 1, nm_lower),
                    scipy.special.poch(nminus - nm_lower + 1, nm_raise),
                ]))

            for j, mode in enumerate(basis):
                coupled_mode = ScalarLGMode(mode.nplus + np_shift, mode.nminus + nm_shift)
                try:
                    i = basis.ind(coupled_mode)
                    matrix[i, j] += matrix_elt_func(mode.nplus, mode.nminus)
                except KeyError:
                    pass

        return matrix

    def vectorize(self):
        return VectorModeOperator([[self, 0], [0, self]])


ScalarModeOperator.zero = ScalarModeOperator('', coefficient=0)

a_plus = ScalarModeOperator('1')
ScalarModeOperator.a_plus = a_plus

a_plus_dag = ScalarModeOperator('1^')
ScalarModeOperator.a_plus_dag = a_plus_dag

a_minus = ScalarModeOperator('2')
ScalarModeOperator.a_minus = a_minus

a_minus_dag = ScalarModeOperator('2^')
ScalarModeOperator.a_minus_dag = a_minus_dag


class VectorModeOperator:
    def __init__(self, components):
        z = ScalarModeOperator.zero
        self.components = np.array(components) + z

    def __add__(self, other):
        if isinstance(other, type(self)):
            return type(self)(self.components + other.components)

        return type(self)(self.components + other)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return type(self)(self.components * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, divisor):
        return type(self)(self.components / divisor)

    def __neg__(self):
        return type(self)(-self.components)

    def toarray(self, basis: ScalarModeBasis):
        d = len(basis)
        matrix = np.zeros((2*d, 2*d), dtype='complex')

        matrix[:d, :d] = self.components[0, 0].toarray(basis)
        matrix[:d, d:] = self.components[0, 1].toarray(basis)
        matrix[d:, :d] = self.components[1, 0].toarray(basis)
        matrix[d:, d:] = self.components[1, 1].toarray(basis)

        return matrix, basis.vectorize()


_sqrt2 = np.sqrt(2)

a_xi = (a_plus + a_minus) / _sqrt2
a_eta = 1j * (a_plus - a_minus) / _sqrt2

a_xi_dag = (a_plus_dag + a_minus_dag) / _sqrt2
a_eta_dag = -1j * (a_plus_dag - a_minus_dag) / _sqrt2

xi = (a_xi + a_xi_dag) / _sqrt2
eta = (a_eta + a_eta_dag) / _sqrt2

nplus = ScalarModeOperator('1^ 1')
nminus = ScalarModeOperator('2^ 2')
n_op = nplus + nminus
n = n_op

k_op = ScalarModeOperator('1 2')
k_dag_op = ScalarModeOperator('1^ 2^')

rho_sq_op = (n_op + 1) + k_op + k_dag_op

# dimensionless momentum^2 (-\Delta_\perp)
pi_sq_op = (n_op + 1) - k_op - k_dag_op

rho_4 = rho_sq_op ** 2
pi_4_op = pi_sq_op ** 2

m_op = rho_sq_op * (n_op + 1) + (n_op + 1) * rho_sq_op
