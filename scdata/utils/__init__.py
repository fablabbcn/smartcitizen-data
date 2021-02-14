from .out import std_out
from .date import localise_date, find_dates
from .units import get_units_convf
from .dictmerge import dict_fmerge
from .lazy import LazyCallable
from .logs import get_tests_log
from .meta import get_current_blueprints, load_blueprints, get_json_from_url
from .report import include_footer
from .stats import spearman, get_metrics
from .cleaning import clean
from .location import get_elevation
from .url_check import url_checker
# from .other.manage_post_info import create_post_info
# from .zenodo import zenodo_upload