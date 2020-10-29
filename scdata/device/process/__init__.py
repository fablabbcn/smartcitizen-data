''' Implementation of different processes to be done in each device '''

from scdata.utils import LazyCallable
from .formulae import absolute_humidity, exp_f, fit_exp_f
from .geoseries import is_within_circle
from .timeseries import clean_ts, merge_ts, rolling_avg, poly_ts, geo_located
from .baseline import find_min_max, baseline_calc, get_delta_baseline, get_als_baseline
from .alphasense import alphasense_803_04, basic_4electrode_alg, baseline_4electrode_alg, deconvolution
from .regression import apply_regressor