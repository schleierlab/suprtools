import itertools
from dataclasses import dataclass

import numpy as np
from scipy.constants import c, pi

import sslab_txz.fp_theory.operators as ops
from sslab_txz.fp_theory.coupling import (CouplingConfig,
                                          NearConfocalCouplingMatrix)
from sslab_txz.fp_theory.geometry._base import CavityGeometry
from sslab_txz.fp_theory.modes import ScalarModeBasis
from sslab_txz.fp_theory.operators import VectorModeOperator


@dataclass
class SymmetricCavityGeometry(CavityGeometry):
    length: float
    mirror_curv_rad: float
    eta_astig: float = 0
    asphere_p: float = 0  # \tilde{p}, as defined in van Exter et al. (2022), eq. 28.

    @property
    def fsr(self) -> float:
        return c / (2 * self.length)

    @property
    def g(self) -> float:
        return 1 - self.length / self.mirror_curv_rad

    @property
    def z0(self) -> float:
        return np.sqrt(self.z1 * (self.mirror_curv_rad - self.z1))

    @property
    def z1(self) -> float:
        return self.length / 2

    @property
    def alpha(self) -> float:
        # == w1 / w0
        return np.sqrt(self.mirror_curv_rad / (self.mirror_curv_rad - self.z1))

    @staticmethod
    def _even_modes_list(n_max):
        return list(itertools.chain(*[
            itertools.chain(*[
                [
                    (p, n_total - 2*p, s)
                    for s in ([+1, -1] if n_total != 2*p else [+1])
                ]
                for p in range(0, n_total // 2 + 1)
            ])
            for n_total in range(0, n_max+1, 2)
        ]))

    def luk_near_confocal_modes(self, q_base, max_order=6):
        even_modes = self._even_modes_list(max_order)
        paraxial_freq_00 = self.fsr * (q_base + np.arccos(self.g) / pi)
        k = 2 * pi * paraxial_freq_00 / c

        return np.array([
            self.fsr * (
                q_base
                + 0.5
                + (2*p + abs(l) + 1) * (np.arccos(self.g)/pi - 0.5)
                + 1 / (4 * pi * k * self.mirror_curv_rad) * (
                    2*p**2 + 2*p*l - l**2 + 2*p + l + 2
                    + 4*l*s - 4
                )
            )
            for (p, l, s) in even_modes
            # if not (l == 0 and s == -1)
        ])

    def near_confocal_coupling_matrix(
            self,
            q,
            config: CouplingConfig,
            max_order=6) -> NearConfocalCouplingMatrix:
        paraxial_freq_00 = self.fsr * (q + np.arccos(self.g) / pi)
        k = 2 * pi * paraxial_freq_00 / c
        # w0 = np.sqrt(2 * self.z0 / k)

        pi_k_rm = pi * k * self.mirror_curv_rad
        h_prop_cyc = self.alpha**2 / (8 * pi_k_rm) * ops.pi_4_op
        h_wave_cyc = ((3 - self.alpha**2) * ops.rho_4 - 2 * ops.m_op) / (8 * pi_k_rm)

        h_asphere_cyc = -self.asphere_p * ops.rho_4 / (8 * pi_k_rm) \
            * self.z1 / (self.mirror_curv_rad - self.z1)

        h_vec_dimless = VectorModeOperator([
            [1 + ops.nplus - ops.nminus, 0],
            [0, 1 - ops.nplus + ops.nminus],
        ])
        h_vec_cyc = - h_vec_dimless / (2 * pi_k_rm)

        # [(operator, activation_bool, cross_coupling_bool)*]
        cyc_ops_configs = [
            (h_prop_cyc.vectorize(), config.prop, config.prop_xcoupling),
            (h_wave_cyc.vectorize(), config.wave, config.wave_xcoupling),
            (h_vec_cyc, config.vec, config.vec_xcoupling),
            (h_asphere_cyc.vectorize(), config.asphere, config.asphere_xcoupling),
        ]

        h_parax_cyc = (ops.n_op + 1) * (np.arccos(self.g)/pi - 0.5)
        scalar_basis = ScalarModeBasis.make_transverse_mode_basis(
            range(max_order % 2, max_order + 1, 2)
        )
        matrix, vector_basis = h_parax_cyc.vectorize().toarray(scalar_basis)
        for op, include_bool, xcouple_bool in cyc_ops_configs:
            if not include_bool:
                continue

            contrib = 2 * op.toarray(scalar_basis)[0]
            if xcouple_bool:
                matrix += contrib
            else:
                matrix += np.diag(np.diagonal(contrib))

        # H_astig
        if config.astig:
            if config.astig_xcoupling:
                h_astig_dimless = ops.xi**2 - ops.eta**2
                dimless_operator = h_astig_dimless
            else:
                # this drops a LOT of off-resonant terms!
                h_astig_nocrosscouple_dimless = \
                    ops.a_plus_dag * ops.a_minus + ops.a_minus_dag * ops.a_plus
                dimless_operator = h_astig_nocrosscouple_dimless

            h_astig_cyc = self.eta_astig / (2 * pi) * np.sqrt(self.alpha**2 - 1) * dimless_operator
            matrix += 2 * h_astig_cyc.vectorize().toarray(scalar_basis)[0]

        if config.v_plus_a:

            h_v_plus_a_dimless = VectorModeOperator([
                [0, 1 - ops.nplus + ops.nminus],
                [1 + ops.nplus - ops.nminus, 0],
            ])

            h_v_plus_a = - self.eta_astig / (2 * pi_k_rm) * h_v_plus_a_dimless
            matrix += 2 * h_v_plus_a.toarray(scalar_basis)[0]

        return NearConfocalCouplingMatrix(
            cavity_geo=self,
            matrix=matrix,
            basis=vector_basis,
            longi_ind_base=q,
            parity=(max_order % 2),
        )
