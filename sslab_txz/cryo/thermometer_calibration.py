from __future__ import annotations

import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import ClassVar, Optional, assert_never, cast

import aenum
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special
from jinja2 import Environment, PackageLoader, select_autoescape
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy import odr
from scipy.optimize import RootResults

from sslab_txz._typing import PathSpec
from sslab_txz.plotting import sslab_style

_jinja_env = Environment(
    loader=PackageLoader('sslab_txz.cryo', package_path='templates'),
    autoescape=select_autoescape(),
)

_latex_jinja_env = Environment(
    block_start_string="((*",
    block_end_string="*))",
    variable_start_string="(((",
    variable_end_string=")))",
    comment_start_string="((#",
    comment_end_string="#))",
    loader=PackageLoader('sslab_txz.cryo', package_path='templates'),
    autoescape=select_autoescape(),
)


class Lakeshore340ParseError(Exception):
    pass


class DataFormat(aenum.Enum):
    _init_ = 'value string'

    # linear interpolation data formats
    LINEAR_MV_KELVIN = 1, 'Millivolts/Kelvin'
    LINEAR_VOLTS_KELVIN = 2, 'Volts/Kelvin'
    LINEAR_OHMS_KELVIN = 3, 'Ohms/Kelvin'
    LINEAR_LOGOHMS_KELVIN = 4, 'Log Ohms/Kelvin'

    # for cubic spline data formats
    CUBIC_VOLTS_KELVIN = 6, 'Volts/Kelvin'
    CUBIC_OHMS_KELVIN = 7, 'Ohms/Kelvin'

    def unit_to_si_base(self, measurement):
        measurement = np.asarray(measurement)
        match self:
            case DataFormat.LINEAR_MV_KELVIN:
                return measurement * 1e-3
            case (DataFormat.LINEAR_VOLTS_KELVIN
                    | DataFormat.LINEAR_OHMS_KELVIN
                    | DataFormat.CUBIC_VOLTS_KELVIN
                    | DataFormat.CUBIC_OHMS_KELVIN):
                return measurement
            case DataFormat.LINEAR_LOGOHMS_KELVIN:
                return 10 ** measurement
            case _:
                assert_never(self)

    def si_base_to_unit(self, base_value):
        base_value = np.asarray(base_value)
        match self:
            case DataFormat.LINEAR_MV_KELVIN:
                return base_value * 1e+3
            case (DataFormat.LINEAR_VOLTS_KELVIN
                    | DataFormat.LINEAR_OHMS_KELVIN
                    | DataFormat.CUBIC_VOLTS_KELVIN
                    | DataFormat.CUBIC_OHMS_KELVIN):
                return base_value
            case DataFormat.LINEAR_LOGOHMS_KELVIN:
                return np.log10(base_value)
            case _:
                assert_never(self)


class TemperatureCoefficient(aenum.Enum):
    _init_ = 'value string'

    NEGATIVE = 1, 'Negative'
    POSITIVE = 2, 'Positive'


@dataclass
class ThermometerSpec:
    name: str
    serial_number: str
    notes: str = ''


class ThermometerCalibration(ThermometerSpec, ABC):
    calibration_date: Optional[str] = None
    fit_temp_range: tuple[float, float]
    temp_range: tuple[float, float]
    fiducial: Optional[ThermometerSpec]
    serial_number: str
    name: str

    interp_temps: ClassVar[NDArray[np.float_]] = np.concatenate((
        # `np.arange(0.05, 0.20, 0.01)` would include 0.20, see np.arange docs
        np.linspace(0.05, 0.20, num=15, endpoint=False),
        np.arange(0.20, 0.50, 0.02),
        np.arange(0.50, 1.20, 0.05),
        np.arange(1.20, 4.00, 0.10),
        np.arange(4.00, 6.00, 0.20),
        np.arange(6.00, 20.0, 0.50),
        np.arange(20.0, 40.0, 1.00),
        np.arange(40.0, 60.0, 2.00),
        np.arange(60.0, 200., 5.00),
        np.arange(200., 290., 10.0),
        np.arange(290., 300., 2.00),
    ))

    @abstractmethod
    def temp_to_resistance(self, temp):
        raise NotImplementedError

    @abstractmethod
    def resistance_to_temp(self, resistance):
        raise NotImplementedError

    @abstractmethod
    def sensitivity(self, temp):
        raise NotImplementedError

    @abstractmethod
    def dimless_sensitivity(self, temp):
        '''
        d(log R) / d(log T)
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_340_file(cls, fname):
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> tuple[Figure, Sequence[Axes]]:
        raise NotImplementedError

    def as_latex(self) -> str:
        raise NotImplementedError

    def write_pdf(
            self,
            fname: PathSpec,
            sensor_model: str,
            temp_low: Optional[float] = None,
            temp_high: Optional[float] = None,
            *,
            notes: Optional[str] = None,
            force: bool = False,
    ) -> None:
        path = Path(fname)
        if path.exists() and not force:
            raise FileExistsError
        if path.suffix != '.tex':
            raise ValueError
        path.parent.mkdir(parents=True, exist_ok=True)

        t_low: float = self.temp_range[0] if temp_low is None else temp_low
        t_high: float = self.temp_range[1] if temp_high is None else temp_high
        if not self.temp_range[0] <= t_low < t_high <= self.temp_range[1]:
            raise ValueError

        temps_mask = (t_low <= self.interp_temps) & (t_high >= self.interp_temps)
        temps = self.interp_temps[temps_mask]

        rs = self.temp_to_resistance(temps)
        dlogr_dlogt = self.dimless_sensitivity(temps)
        sensitivity = self.sensitivity(temps)

        # interp_table_arr = np.rec.fromarrays(
        #     [
        #         temps,
        #         rs,
        #         sensitivity,
        #         dlogr_dlogt,
        #     ],
        #     names=['Temperature (K)', 'Resistance (Ohm)', 'dR/dT (Ohm/K)', 'dlogR/dlogT']
        # )
        # np.savetxt(
        #     fname,
        #     interp_table_arr,
        #     delimiter=',',
        #     fmt=['%#.4g', '%#.6g', '%#.5g', '%#.5g'],
        #     header=','.join(interp_table_arr.dtype.names),
        # )

        template = _latex_jinja_env.get_template('calibration_report.tex.jinja2')
        records = list(zip(
            temps,
            rs,
            sensitivity,
            dlogr_dlogt,
        ))
        rendered_file = template.render(
            sensor_model=sensor_model,
            serial_number=self.serial_number,
            calibration_date=self.calibration_date,
            calibration_range=self.temp_range,
            fiducial_sensor=self.fiducial,
            fit_details=self.as_latex(),
            notes=notes,
            records=records,
        )

        fig, _ = self.plot()
        fig.savefig(path.parent / 'caldata.pdf')

        with open(path, 'w') as fp:
            fp.write(rendered_file)

        subprocess.run(
            ['latexmk', f'-outdir={path.parent}', '-pdf', str(path)],
            check=True,
            text=True,
        )

    def write_340_file(
            self,
            fname: PathSpec,
            temps: Sequence[float],
            data_format: DataFormat,
            sensor_model: str,
            setpoint_limit: Optional[float] = None,
    ) -> None:
        if len(temps) <= 1:
            raise ValueError

        path = Path(fname)
        if path.exists():
            raise FileExistsError

        min_temp, max_temp = min(temps), max(temps)
        min_temp_r, max_temp_r = \
            self.temp_to_resistance(min_temp), self.temp_to_resistance(max_temp)
        temp_coeff = (
            TemperatureCoefficient.POSITIVE
            if max_temp_r > min_temp_r
            else TemperatureCoefficient.NEGATIVE
        )
        temps_sorted = sorted(
            temps,
            reverse=(temp_coeff == TemperatureCoefficient.NEGATIVE),
        )

        template = _jinja_env.get_template('lakeshore340.340.jinja2')
        records = list(zip(
            data_format.si_base_to_unit(self.temp_to_resistance(temps_sorted)),
            temps_sorted,
        ))
        rendered_file = template.render(
            name=sensor_model,
            serial_number=self.serial_number,
            data_format=data_format,
            setpoint_limit=(self.temp_range[1] if setpoint_limit is None else setpoint_limit),
            temp_coeff=temp_coeff,
            records=records,
        )

        with open(path, 'w') as fp:
            fp.write(rendered_file)

    def export(
            self,
            dir: PathSpec,
            interp_temps: Sequence[float],
            data_format: DataFormat = DataFormat(4),  # LINEAR_LOGOHMS_KELVIN
            setpoint_limit: Optional[float] = None,
            notes: Optional[str] = None,
    ):
        path = Path(dir)
        # if path.exists():
        #     raise FileExistsError
        path.mkdir(parents=True, exist_ok=False)

        self.write_340_file(
            path / f'{self.serial_number}.340',
            interp_temps,
            data_format,
            sensor_model=self.name,
            setpoint_limit=setpoint_limit,
        )

        self.write_pdf(
            path / f'{self.serial_number}.tex',
            sensor_model=self.name,
            notes=notes,
        )


class LinearInterpolator(ThermometerCalibration):
    def __init__(self, calibration_data, serial_number: str, format: DataFormat, name: str):
        '''
        calibration_data: structured array, fields ('temp', 'units')
            Calibration data, sorted by increasing resistance.
        '''
        self.calibration_data = calibration_data
        self.format = format
        self.serial_number = serial_number
        self.name = name

    def temp_to_resistance(self, temp):
        # assumes a negative temperature coefficient
        cal_data = self.calibration_data[::-1]
        interp_val = np.interp(temp, cal_data['temp'], cal_data['units'])
        return self.format.unit_to_si_base(interp_val)

    def resistance_to_temp(self, resistance):
        return np.interp(
            self.format.si_base_to_unit(resistance),
            self.calibration_data['units'],
            self.calibration_data['temp'],
        )

    def sensitivity(self, temp):
        '''dR/dT'''
        # assuming NTC
        cal_data = self.calibration_data[::-1]
        cal_temps = cal_data['temp']
        inds = np.searchsorted(cal_temps, temp)

        resistances = self.format.unit_to_si_base(cal_data['units'])
        dRs = resistances[inds + 1] - resistances[inds]
        dTs = cal_temps[inds + 1] - cal_temps[inds]

        return dRs / dTs

    def dimless_sensitivity(self, temp):
        return self.sensitivity(temp) * temp / self.temp_to_resistance(temp)

    @staticmethod
    def _match_regex_group(regex: str, matchstr: str, group: int = 1) -> str:
        re_match = re.fullmatch(regex, matchstr)
        if re_match is None:
            raise Lakeshore340ParseError
        return re_match.group(group)

    @classmethod
    def from_340_file(cls, fname) -> LinearInterpolator:
        with open(fname) as fp:
            for i, line in enumerate(fp):
                if i == 0:
                    name = cls._match_regex_group(R'Sensor Model:\s*(.+)\n', line)
                elif i == 1:
                    serial_number = cls._match_regex_group(R'Serial Number:\s*(.+)\n', line)
                elif i == 2:
                    data_format_value: int = int(
                        cls._match_regex_group(R'Data Format:\s*(\d+)\s+\(.+\)\n', line, group=1),
                    )
                    break

        fmt = DataFormat(data_format_value)
        data = np.genfromtxt(fname, names=['n', 'units', 'temp'], skip_header=9)

        # to be parsimonious, we could instead use
        # np.rec.fromarrays(
        #     [data['units'], data['temp']],
        #     formats=None,  # type stub doesn't account for default value of `formats`
        #     names=['resistance', 'temp'],
        # )
        return cls(data, serial_number, format=fmt, name=name)

    def plot(self):
        fig, ax = plt.subplots()
        ax = cast(Axes, ax)

        ax.plot(
            self.calibration_data['temp'],
            self.format.unit_to_si_base(self.calibration_data['units']),
            marker='.',
        )
        if self.format == DataFormat.LINEAR_LOGOHMS_KELVIN:
            ax.set_yscale('log')


class ChebyshevFit(ThermometerCalibration):
    def __init__(
            self,
            calibration_data,
            temp_range,
            resistance_range,
            fit_order: int,
            serial_number: str,
            name: str,
            fiducial: Optional[ThermometerSpec] = None,
    ):
        if fit_order < 0:
            raise ValueError

        self.calibration_data = calibration_data
        self.fit_temp_range = temp_range
        self.resistance_range = resistance_range
        self.serial_number = serial_number
        self.name = name
        self.fiducial = fiducial

        calibration_temps = calibration_data['temp']
        fit_data_mask = (temp_range[0] <= calibration_temps) & (calibration_temps <= temp_range[1])

        self.fit_data = self.calibration_data[fit_data_mask]

        model = odr.Model(self.chebyshev_fit_func)
        odr_data = odr.Data(
            self.resistance_to_k(self.fit_data['resistance'], resistance_range),
            self.fit_data['temp'],
        )

        self.fit_order = fit_order
        self.odr = odr.ODR(odr_data, model, beta0=np.zeros(1 + fit_order))

    def run(self):
        self.odr_result = self.odr.run()

    def sensitivity(self, temp):
        '''dR/dT'''
        raise NotImplementedError

    def dimless_sensitivity(self, temp):
        '''
        d log R / d log T = sum_{k=1} a_k k (log T)^{k-1}
        '''
        raise NotImplementedError

    @staticmethod
    def resistance_to_k(resistances, range_endpoints):
        zl, zu = np.log10(range_endpoints)
        z = np.log10(resistances)
        return ((z - zl) - (zu - z)) / (zu - zl)

    @staticmethod
    def chebyshev_fit_func(coeffs, log_r_param):
        return np.sum(
            [coeffs[i] * scipy.special.chebyt(i)(log_r_param) for i in range(len(coeffs))],
            axis=0,
        )

    def __call__(self, resistance):
        return self.resistance_to_temp(resistance)

    def from_340_file(self, fname):
        raise NotImplementedError

    def resistance_to_temp(self, resistance):
        return self.chebyshev_fit_func(
            self.odr_result.beta,
            self.resistance_to_k(resistance, self.resistance_range),
        )

    def temp_to_resistance(self, temp):
        # scipy.optimize.root_scalar(self.resistance_to_temp, )
        raise NotImplementedError

    def plot(self):
        fig, axs = plt.subplots(
            ncols=2,
            figsize=(8, 4),
            layout='constrained',
            sharey=True,
            gridspec_kw=dict(width_ratios=[2, 1]),
        )
        ax, ax_resid = axs
        ax.scatter(self.fit_data['temp'], self.fit_data['resistance'], s=1)

        r_range = np.linspace(*self.resistance_range)
        ax.plot(
            self.resistance_to_temp(r_range),
            r_range,
            color='red',
            linewidth=1,
        )

        ax_resid.scatter(
            self.fit_data['temp'] - self.resistance_to_temp(self.fit_data['resistance']),
            self.fit_data['resistance'],
            s=1
        )

        ax.set_xlabel('Fiducial temperature [K]')
        ax_resid.set_xlabel('Fit residual [K]')
        axs[0].set_ylabel(r'Resistance [$\Omega$]')

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_xlim(0.5, 10)
        # ax_resid.set_xlim(-0.2, 0.2)
        # ax.set_ylim(2650, 2750)

        sslab_style(ax)
        sslab_style(ax_resid)


class LogLogPolyFit(ThermometerCalibration):
    def __init__(
            self,
            calibration_data,
            fit_temp_range: tuple[float, float],
            resistance_range: tuple[float, float],
            fit_order: int,
            serial_number: str,
            name: str,
            fiducial: Optional[ThermometerSpec] = None,
            calibration_date: Optional[str] = None,
            temp_range: Optional[tuple[float, float]] = None,
    ):
        if fit_order < 0:
            raise ValueError

        self.calibration_data = calibration_data
        self.calibration_date = calibration_date
        self.serial_number = serial_number
        self.name = name
        self.fiducial = fiducial
        self.fit_temp_range = fit_temp_range
        self.temp_range = self.fit_temp_range if temp_range is None else temp_range
        if not (self.fit_temp_range[0]
                <= self.temp_range[0]
                < self.temp_range[1]
                <= self.fit_temp_range[1]):
            raise ValueError

        self.resistance_range = resistance_range

        calibration_temps = calibration_data['temp']
        fit_data_mask = (self.fit_temp_range[0] <= calibration_temps) \
            & (calibration_temps <= self.fit_temp_range[1])
        self.fit_data = self.calibration_data[fit_data_mask]

        model = odr.polynomial(fit_order)
        self.fit_order = fit_order

        odr_data = odr.Data(
            np.log10(self.fit_data['temp']),
            np.log10(self.fit_data['resistance']),
        )
        self.odr = odr.ODR(odr_data, model, beta0=np.zeros(1 + fit_order))

    def as_latex(self) -> str:
        template = _latex_jinja_env.get_template('loglogpoly.tex.jinja2')
        return template.render(coeffs=self.odr_result.beta)

    def run(self):
        self.odr_result = self.odr.run()

    def __call__(self, temp):
        return self.temp_to_resistance(temp)

    def logtemp_to_logresist(self, logtemp):
        return np.poly1d(self.odr_result.beta[::-1])(logtemp)

    def temp_to_resistance(self, temp):
        return 10 ** self.logtemp_to_logresist(np.log10(temp))

    def logresist_to_logtemp(self, logresist: Real) -> float:
        sol: RootResults = scipy.optimize.root_scalar(
            lambda log_t: self.logtemp_to_logresist(log_t) - logresist,
            fprime=(lambda log_t: self.dimless_sensitivity(10 ** log_t)),
            fprime2=self._second_derivative,
            x0=np.log10(self.temp_range[0]),
            x1=np.log10(self.temp_range[1]),
            xtol=1e-8,
        )
        if not sol.converged:
            raise RuntimeError
        return sol.root

    def resistance_to_temp(self, resistance: ArrayLike):
        logtemps = np.vectorize(self.logresist_to_logtemp)(np.log10(resistance))
        return 10 ** logtemps

    def dimless_sensitivity(self, temp:  ArrayLike) -> ArrayLike:
        '''
        d log(10) R / d log(10) T = sum_{k=1} a_k k (log(10) T)^{k-1} = dR/dT * T/R
        '''
        temp = np.asarray(temp)
        if np.any(temp < self.fit_temp_range[0]) | np.any(temp > self.fit_temp_range[1]):
            raise ValueError

        # k = 1 to polynomial deg
        poly_coeffs = self.odr_result.beta[1:] * np.arange(1, self.fit_order + 1)

        return np.poly1d(poly_coeffs[::-1])(np.log10(temp))

    def sensitivity(self, temp: ArrayLike):
        '''
        dR / dT
        '''
        return self.dimless_sensitivity(temp) * self.temp_to_resistance(temp) / temp

    def _second_derivative(self, logtemp):
        '''
        d^2 (log10 R) / (d log10 T)^2 = sum_{k=2} a_k k(k-1) (log10 T)^{k-2}
        '''
        k = np.arange(2, self.fit_order + 1)
        poly_coeffs = self.odr_result.beta[2:] * k * (k-1)
        return np.poly1d(poly_coeffs[::-1])(logtemp)

    def from_340_file(self, fname):
        raise NotImplementedError

    def plot(self):
        fig, axs = plt.subplots(
            nrows=3,
            figsize=(6, 7),
            layout='constrained',
            sharex=True,
            gridspec_kw=dict(height_ratios=[2, 1, 1]),
        )
        axs = cast(tuple[Axes, ...], axs)
        ax, ax_resid, ax_uncert = axs
        ax.scatter(self.fit_data['temp'], self.fit_data['resistance'], s=1)

        # temp_range = np.linspace(*self.temp_range, 5001)
        fit_temp_linspace = np.geomspace(*self.fit_temp_range, 1001)

        ax.plot(
            fit_temp_linspace,
            self.temp_to_resistance(fit_temp_linspace),
            color='red',
            linewidth=1,
        )

        residuals = self.fit_data['resistance'] - self.temp_to_resistance(self.fit_data['temp'])
        ax_resid.scatter(
            self.fit_data['temp'],
            residuals,
            s=1
        )

        ax_uncert.scatter(
            self.fit_data['temp'],
            residuals / self.sensitivity(self.fit_data['temp']),
            s=1,
        )

        axs[-1].set_xlabel('Fiducial temperature [K]')
        ax_resid.set_ylabel(r'Fit residual [$\Omega$]')
        ax.set_ylabel(r'Resistance [$\Omega$]')
        ax_uncert.set_ylabel(r'Temperature deviation [K]')

        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax_uncert.set_yscale('symlog', linthresh=1e-3)
        # ax.set_xlim(0.5, 10)
        # ax_resid.set_ylim(-0.2, 0.2)
        # ax.set_ylim(2650, 2750)

        for ax in axs:
            sslab_style(ax)

        return fig, axs
