from matplotlib.ticker import AutoMinorLocator


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
