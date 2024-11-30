import re
from collections.abc import Sequence
from typing import assert_never, cast

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.ticker import AutoMinorLocator

from ._angleannotation import AngleAnnotation as AngleAnnotation


def minor_ticks_on(ax, which='both'):
    if ax.get_xscale() != 'log' and which in ['both', 'x']:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() != 'log' and which in ['both', 'y']:
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def make_grids(ax):
    ax.grid(which='major', color='0.7')
    ax.grid(which='minor', color='0.9')


def set_reci_ax(ax):
    def reci(x):
        return 1/(1e-15 + x)
    ax.set_xscale('function', functions=(reci, reci))


def sslab_style(ax):
    minor_ticks_on(ax)
    make_grids(ax)


def _digit_to_roman(digit: int, place: int) -> str:
    if not 1 <= digit <= 9:
        raise ValueError

    ROMAN = [('I', 'V'), ('X', 'L'), ('C', 'D'), ('M', 'MMMM')]
    one, five = ROMAN[place]
    ten, _ = ROMAN[place + 1]

    if digit == 4:
        return one + five
    elif digit == 9:
        return one + ten

    n_fives, n_ones = divmod(digit, 5)
    return five * n_fives + one * n_ones


def to_roman(n: int) -> str:
    digits_reversed = map(int, reversed(str(n)))
    return ''.join(reversed([_digit_to_roman(d, place) for place, d in enumerate(digits_reversed)]))


def expand_range(values, factor=1.1):
    '''
    Given a set of real values with extrema val_min, val_max,
    give a pair of numbers lo, hi such that the interval [lo, hi]
    is centered on [val_min, val_max] and is longer by `factor`.
    '''
    halfspan = (max(values) - min(values)) / 2
    midpt = (max(values) + min(values)) / 2
    return midpt - halfspan * factor, midpt + halfspan * factor


def frexp10(x):
    exp = int(np.floor(np.log10(abs(x))))
    return x / 10**exp, exp


def latex_frexp10(x):
    significand, exp = frexp10(x)

    if significand == 1:
        return fr'10^{{{exp}}}'
    return fr'{significand} \times 10^{{{exp}}}'


def label_subplots(
        fig: Figure,
        artists: Sequence[Axes] | Sequence[SubFigure],
        label_fmt='(alph)',
        adjust=(5, -5),
        colors=None,
):
    if colors is None:
        colors = ['black'] * len(artists)

    for i, artist in enumerate(artists):
        # label physical distance in and down:
        adjust_x, adjust_y = adjust
        trans = matplotlib.transforms.ScaledTranslation(adjust_x/72, adjust_y/72, fig.dpi_scale_trans)
        # label = f'({chr(97 + i)})'

        def numsubber(m: re.Match[str]) -> str:
            match m.group(0):
                case 'alph':
                    return chr(97 + i)
                case 'Alph':
                    return chr(65 + i)
                case 'arabic':
                    return str(i + 1)
                case 'roman':
                    return to_roman(i + 1).lower()
                case 'Roman':
                    return to_roman(i + 1)
                case str():
                    raise ValueError
                case _:
                    raise TypeError

        label = re.sub(
            'alph|Alph|roman|Roman|arabic',
            numsubber,
            label_fmt,
        )

        artist = cast(Axes | SubFigure, artist)
        match artist:
            case Axes() as ax:
                transform = ax.transAxes
                text = ax.text
            case SubFigure() as subfig:
                transform = subfig.transSubfigure
                text = subfig.text
            case _:
                assert_never()

        text(
            0.0, 1.0,
            label,
            transform=(transform + trans),
            fontsize='large',
            verticalalignment='top',
            color=colors[i],
            # fontfamily='serif',
            # bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0),
        )


def watermark(ax: Axes, text: str, **kwargs):
    default_kw = dict(
        color='red',
        alpha=0.5,
        fontsize=30,
        ha='center',
        va='center',
        rotation=30,
    )
    ax.text(
        0.5, 0.5,
        text,
        transform=ax.transAxes,
        fontdict=None,  # redundant line for mypy
        **(default_kw | kwargs),
    )


def annotate_length(
        ax,
        text,
        left_endpt,
        right_endpt,
        offset_points=8,
        horizontalalignment='center',
        verticalalignment='center',
        reverse=False,
        arrowprops=dict(),
):
    left_endpt = np.asarray(left_endpt)
    right_endpt = np.asarray(right_endpt)

    midpt = (left_endpt + right_endpt) / 2
    arrowprops_default = dict(
        arrowstyle='<|-|>',
        # width=0.5,
        # headwidth=4,
        # headlength=8,
        color='0.3',
        shrinkA=1,
        shrinkB=1,
    )
    ax.annotate(
        '',
        xy=left_endpt,
        xycoords=ax.transData,
        xytext=right_endpt,
        textcoords=ax.transData,
        arrowprops=(arrowprops_default | arrowprops),
    )

    linevec_x, linevec_y = right_endpt - left_endpt
    normalvec = np.array([linevec_y, -linevec_x])
    normalvec /= np.linalg.norm(normalvec)
    if normalvec[1] < 0:
        normalvec = -normalvec
    elif normalvec[1] == 0 and normalvec[0] < 0:
        normalvec = -normalvec

    ax.annotate(
        text,
        xy=midpt,
        xycoords=ax.transData,
        xytext=normalvec * offset_points * (-1 if reverse else +1),
        textcoords='offset points',
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
    )
