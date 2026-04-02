''' Implementation of different processes to be done in each device '''

# TODO UPDATE all these functions to make them comply with StatusCode types
from scdata.tools.lazy import LazyCallable
from .formulae import absolute_humidity, exp_f, fit_exp_f
from .geoseries import is_within_circle
from .timeseries import clean_ts, merge_ts, rolling_avg, poly_ts, within, time_derivative, delta_index_ts, baseline_als
from .alphasense import alphasense_803_04, alphasense_als, alphasense_pt1000, channel_names, basic_4electrode_alg, baseline_4electrode_alg, deconvolution, ec_sensor_temp
from .regression import apply_regressor