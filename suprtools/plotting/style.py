from collections.abc import Callable
from numbers import Real
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib.typing import ColorType, MarkerType
from scipy.constants import pi
from suprtools.typing import PlotKwargs
from uncertainties import unumpy as unp

textwidth_pts = 510
columnwidth_pts = 246
columnwidth_mpl_in = 5  # dimension for for plt.subplots()
textwidth_mpl_in = columnwidth_mpl_in * textwidth_pts / columnwidth_pts


descending_vlabel_kw = dict(
    labelpad=10,
    rotation=-90,
)

annotation_arrowprops_default = dict(
    color='0.3',
    shrinkA=0,
    shrinkB=0,
)


polarization_markers = ('.', 'x')


def mode_palette(n_colors):
    return sns.cubehelix_palette(
        n_colors=n_colors,
        start=0,
        rot=6,
        reverse=True,
        gamma=0.8,
        dark=0.2,
        light=0.7,
    )


def kwarg_func_factory(
        label: bool | str,
        q_range: tuple[int, int],
        markers: tuple[MarkerType, MarkerType] = polarization_markers,
        palette_func: Callable[[int], list[ColorType]] = mode_palette,
        **kwargs,
) -> Callable[[Any, int, int], PlotKwargs]:
    '''
    Parameters
    ----------
    label: bool or str
    q_range: tuple[int, int]
        The range of longitudinal indices q (min, max, inclusive) for
        which we are plotting.
    markers: tuple[MarkerType, MarkerType], optional
        Markers for x and y polarization.
    palette_func: (int) -> list[ColorType]
        A function that takes in a number of colors and returns a color
        palette with that many colors.
    kwargs:
        Additional kwargs that will be included in the output of
        kwarg_func for any input.

    Returns
    -------
    kwarg_func: (Any, int, int) -> PlotKwargs
        Function that takes in mode_data, q (longitudinal index), and
        polarization (+/-1) and gives a set of keyword-args for ax.plot
    '''
    def kwarg_func(mode_data, q, pol):
        polstr = 'x' if pol == +1 else 'y'
        palette = palette_func(q_range[1] - q_range[0] + 1)

        base_label = fr'$\mathrm{{TEM}}_{{{q},0,0,{polstr}}}$'
        if mode_data is not None:
            if isinstance(mode_data, Real):
                meanfreq = mode_data
            else:
                freqs = unp.nominal_values(mode_data['pole_i']) / (2 * pi)
                meanfreq = np.nanmean(freqs)

            base_label = fr'$\mathrm{{TEM}}_{{{q},0,0,{polstr}}}$ @ {meanfreq/1e+9:.4f} GHz'

        match label:
            case bool(b):
                label_str = base_label if b else ''
            case str(suffix):
                label_str = ' '.join([base_label, suffix])
            case _:
                raise TypeError

        final_kwargs = dict(
            alpha=0.5,
            color=palette[q - q_range[0]],
            marker=(markers[0] if pol == +1 else markers[1]),
            label=label_str,
        ) | kwargs
        return final_kwargs
    return kwarg_func
