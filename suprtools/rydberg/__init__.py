from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import arc
import lmfit
import numpy as np
from lmfit import Parameter
from suprtools.typing import PlotKwargs

QuantumNumberSpec = int | float | str
LjmjSpec = tuple[QuantumNumberSpec, QuantumNumberSpec, QuantumNumberSpec]


@dataclass
class PowerLawInterpolator:
    power: Real | Parameter
    log_prefactor: Real | Parameter

    def __call__(self, x):
        return np.exp(self.log_prefactor + np.log(x) * self.power)


class RydbergTransitionSeries(ABC):
    n_range: Sequence[int]
    plot_kw: PlotKwargs
    plot_color: str

    @property
    def label(self):
        return fr'${self.label1} \to {self.label2}$'

    @property
    @abstractmethod
    def label1(self):
        ...

    @property
    @abstractmethod
    def label2(self):
        ...

    @abstractmethod
    def transition_frequencies(self):
        ...

    @property
    @abstractmethod
    def polarization(self):
        ...

    @abstractmethod
    def dipole_matrix_elements(self):
        ...

    @abstractmethod
    def init_state_lifetimes(self):
        ...

    @abstractmethod
    def final_state_lifetimes(self):
        ...

    @staticmethod
    def fit_power_law(x, y, power=None):
        model = lmfit.models.LinearModel()
        params = model.make_params(
            slope=dict(value=(0 if power is None else power), vary=(power is None)),
            intercept=0,
        )
        fit = model.fit(np.log(y), x=np.log(x), params=params)

        return PowerLawInterpolator(fit.params['slope'], fit.params['intercept'])

    def plot_vs_freq(self, ax, vals, **kwargs):
        freqs = np.abs(self.transition_frequencies())
        ax.plot(
            freqs,
            vals,
            **(dict(marker='.') | kwargs),
        )
        for i in [0, -1]:
            ax.annotate(
                f'$n = {int(self.n_range[i]):d}$', (freqs[i], vals[i]),
                horizontalalignment='center',
                fontsize='xx-small',
            )

    def power_law_params(self):
        freqs = np.abs(self.transition_frequencies())

        return {
            'd': self.fit_power_law(freqs, np.abs(self.dipole_matrix_elements())),
            'lifetime': self.fit_power_law(freqs, self.init_state_lifetimes()),
        }

    def plot_numbers(self, ax_d, ax_lifetime, **kwargs):
        common_kw = self.plot_kw | kwargs

        self.plot_vs_freq(
            ax_d,
            np.abs(self.dipole_matrix_elements()),
            **(dict(label=self.label) | common_kw),
        )

        self.plot_vs_freq(
            ax_lifetime,
            self.init_state_lifetimes(),
            marker='^',
            **(dict(label=f'${self.label1}$') | common_kw),
        )
        self.plot_vs_freq(
            ax_lifetime,
            self.final_state_lifetimes(),
            marker='v',
            **(dict(label=f'${self.label2}$') | common_kw),
        )


class LowLTransitionSeries(RydbergTransitionSeries):
    atom: arc.AlkaliAtom

    def __init__(self, atom: arc.AlkaliAtom, n_delta, ljmj1: LjmjSpec, ljmj2: LjmjSpec, n_range: Sequence[int], plot_kw: PlotKwargs):
        self.atom = atom
        self.n_delta = n_delta
        self.ljmj1 = ljmj1
        self.ljmj2 = ljmj2
        self.n_range = n_range
        self.plot_kw = plot_kw

    def transition_frequencies(self):
        return np.array([
            self.atom.getTransitionFrequency(
                n, *self.ljmj1[:2],
                n + self.n_delta, *self.ljmj2[:2],
            )
            for n in self.n_range
        ])

    @property
    def polarization(self):
        mj1 = self.ljmj1[2]
        mj2 = self.ljmj2[2]
        return mj2 - mj1

    @staticmethod
    def state_label(n_delta, ljmj):
        ell, j, mj = ljmj
        l_sym = 'SPDF'[ell]
        n_str = 'n' if n_delta == 0 else f'(n{n_delta:+d})'

        return fr'{n_str}{l_sym}_{{{int(2*j)}/2}}(m_j = \frac{{{int(2*mj)}}}{{2}})'

    @property
    def label1(self):
        return self.state_label(0, self.ljmj1)

    @property
    def label2(self):
        return self.state_label(self.n_delta, self.ljmj2)

    def dipole_matrix_elements(self):
        return np.array([
            self.atom.getDipoleMatrixElement(n, *self.ljmj1, n + self.n_delta, *self.ljmj2, self.polarization)
            for n in self.n_range
        ])

    def init_state_lifetimes(self):
        return np.array([
            self.atom.getStateLifetime(int(n), *self.ljmj1[:2])
            for n in self.n_range
        ])

    def final_state_lifetimes(self):
        return np.array([
            self.atom.getStateLifetime(int(n + self.n_delta), *self.ljmj2[:2])
            for n in self.n_range
        ])


class CircularTransitionSeries(RydbergTransitionSeries):
    atom: arc.AlkaliAtom

    def __init__(self, atom: arc.AlkaliAtom, n_range: Sequence[int], plot_kw: PlotKwargs):
        self.atom = atom
        self.plot_kw = plot_kw
        self.n_range = n_range

    def transition_frequencies(self):
        return np.array([
            self.atom.getTransitionFrequency(
                n, n-1.5, n-0.5,
                n-1, n-2, n-1.5,
            )
            for n in self.n_range
        ])

    @property
    def polarization(self):
        return -1

    @property
    def label1(self):
        return 'nC'

    @property
    def label2(self):
        return '(n-1)C'

    def dipole_matrix_elements(self):
        return np.array([
            self.atom.getDipoleMatrixElement(
                n, n-1, n-0.5, n-0.5,
                n-1, n-2, n-1.5, n-1.5,
                self.polarization,
            )
            for n in self.n_range
        ])

    def init_state_lifetimes(self):
        return np.array([
            self.atom.getStateLifetime(int(n), n-1, n-0.5)
            for n in self.n_range
        ])

    def final_state_lifetimes(self):
        return np.array([
            self.atom.getStateLifetime(int(n)-1, n-2, n-1.5)
            for n in self.n_range
        ])


def get_common_series(atom: arc.AlkaliAtom) -> list[RydbergTransitionSeries]:
    return [
        CircularTransitionSeries(atom, n_range=range(38, 50), plot_kw=dict(color='C2')),
        LowLTransitionSeries(atom, 0, (1, 1.5, 1.5), (0, 0.5, 0.5), n_range=range(35, 43), plot_kw=dict(color='C3')),
        LowLTransitionSeries(atom, 0, (2, 2.5, 2.5), (1, 1.5, 1.5), n_range=range(44, 53), plot_kw=dict(color='C4')),
        LowLTransitionSeries(atom, -1, (0, 0.5, 0.5), (1, 1.5, 1.5), n_range=range(35, 43), plot_kw=dict(color='C5')),
        LowLTransitionSeries(atom, -1, (0, 0.5, 0.5), (1, 1.5, 0.5), n_range=range(35, 43), plot_kw=dict(color='C6')),
        LowLTransitionSeries(atom, -1, (0, 0.5, 0.5), (1, 1.5, -0.5), n_range=range(35, 43), plot_kw=dict(color='C7')),
        # LowLTransitionSeries(atom, -1, (0, 0.5, 0.5), (1, 1.5, -1.5), n_range=range(35, 43), plot_kw=dict(color='C8')),
    ]
