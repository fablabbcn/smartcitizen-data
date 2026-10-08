import sys
from .custom_logger import logger

class LazyCallable(object):
    '''
        Adapted from Alex Martelli's answer on this post on stackoverflow:
        https://stackoverflow.com/questions/3349157/python-passing-a-function-name-as-an-argument-in-a-function
    '''
    def __init__(self, name):
        self.n = name
        self.f = None
    def __call__(self, *a, **k):
        if self.f is None:
            logger.info(f"Loading {self.n.rsplit('.', 1)[1]} from {self.n.rsplit('.', 1)[0]}")
            modn, funcn = self.n.rsplit('.', 1)
            if modn not in sys.modules:
                __import__(modn)
            self.f = getattr(sys.modules[modn], funcn)
        return self.f(*a, **k)


PLOT_EXTRA = 'pip install "scdata[plot]"'


def plot_method(name):
    '''
    Method that calls scdata.plot.<name> with the object (Device or Test) as first argument.
    Plotting is an optional extra: its libraries are only imported when a plot is made
    '''
    def method(self, *args, **kwargs):
        from importlib import import_module
        try:
            module = import_module('scdata.plot')
        except ImportError as error:
            raise ImportError(f'{name} needs the plotting libraries: {PLOT_EXTRA}') from error
        function = getattr(module, name, None)
        if function is None:
            # Some plots need optional libraries (folium for maps) or IPython (uplot)
            raise ImportError(f'{name} is not available here: it needs IPython or {PLOT_EXTRA}')
        return function(self, *args, **kwargs)

    method.__name__ = name
    method.__doc__ = f'Plot with scdata.plot.{name} (needs {PLOT_EXTRA})'
    return method
