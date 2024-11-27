import re
from collections.abc import Sequence

import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


def label_subplots(fig: Figure, axs: Sequence[Axes], label_fmt='(alph)'):
    for i, ax in enumerate(axs):
        # label physical distance in and down:
        trans = matplotlib.transforms.ScaledTranslation(5/72, -5/72, fig.dpi_scale_trans)
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

        ax.text(
            0.0, 1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize='large',
            verticalalignment='top',
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
