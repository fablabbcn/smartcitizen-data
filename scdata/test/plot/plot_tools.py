# Numeric
from math import sqrt, ceil, isnan
# from sklearn.metrics import mean_squared_error
from numpy import power, array, inf
from pandas import cut, date_range
from scipy import stats

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import gridspec
plt.ioff()

# Seaborn
# Plotly
import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.io as pio
pio.renderers.default = "jupyterlab"

# Other
from traceback import print_exc

'''
Available styles
['_classic_test', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 
'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 
'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 
'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 
'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
'''

markers = ["o" ,"v" ,"^" ,"<" ,">" ,"1" ,"2" ,"3" ,"4" ,"8" ,"s" ,"p" ,"P" ,"*" ,"h" ,"H" ,"+" ,"x" ,"X" ,"D" ,"d" ,"|", "_"]
colors = ['black',    'silver',    'red',        'sienna',     'gold',
          'orange',   'salmon',    'chartreuse', 'green',      'mediumspringgreen', 'lightseagreen',
          'darkcyan', 'royalblue', 'blue',       'blueviolet', 'purple',            'fuchsia',
          'pink',     'tan',       'olivedrab',  'tomato',     'yellow',            'turquoise']