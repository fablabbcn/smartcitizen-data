# from .ts_iplot import ts_iplot
from scdata._config import config

from .ts_plot import ts_plot

if config._ipython_avail:
    from .ts_uplot import ts_uplot
    from .ts_dispersion_uplot import ts_dispersion_uplot

# from .heatmap_iplot import heatmap_iplot
from .box_plot import box_plot
from .heatmap_plot import heatmap_plot
from .maps import device_metric_map, path_plot
from .scatter_dispersion_grid import scatter_dispersion_grid
from .scatter_plot import scatter_plot
from .ts_dendrogram import ts_dendrogram
from .ts_dispersion_grid import ts_dispersion_grid
from .ts_dispersion_plot import ts_dispersion_plot
# from .scatter_iplot import scatter_iplot
from .ts_scatter import ts_scatter

# from .tools import (target_diagram, scatter_diagram)
