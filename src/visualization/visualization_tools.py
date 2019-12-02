from math import sqrt
import numpy as np
import matplotlib

import pandas as pd

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import gridspec
plt.ioff()

import seaborn as sns
import math

from scipy import stats

'''
Available styles
['_classic_test', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10']
'''

# Plotly
import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.io as pio
pio.renderers.default = "jupyterlab"
import traceback

from sklearn.metrics import mean_squared_error
from math import sqrt

markers = ["o" ,"v" ,"^" ,"<" ,">" ,"1" ,"2" ,"3" ,"4" ,"8" ,"s" ,"p" ,"P" ,"*" ,"h" ,"H" ,"+" ,"x" ,"X" ,"D" ,"d" ,"|", "_"]
colors = ['black',    'silver',    'red',        'sienna',     'gold',
		  'orange',   'salmon',    'chartreuse', 'green',      'mediumspringgreen', 'lightseagreen',
		  'darkcyan', 'royalblue', 'blue',       'blueviolet', 'purple',            'fuchsia',
		  'pink',     'tan',       'olivedrab',  'tomato',     'yellow',            'turquoise']

def targetDiagram(models, plot_train, style_to_use = 'seaborn-paper'):
	style.use(style_to_use)
	
	def minRtarget(targetR):
		return sqrt(1+ np.power(targetR,2)-2*np.power(targetR,2))

	targetR20 = 0.5
	targetR0 = sqrt(targetR20)
	MR0 = minRtarget(targetR0)
	targetR21 = 0.7
	targetR1 = sqrt(targetR21)
	MR1 = minRtarget(targetR1)
	targetR22 = 0.9
	targetR2 = sqrt(targetR22)
	MR2 = minRtarget(targetR2)

	fig  = plt.figure(figsize=(13,13))
	i = -1
	prev_group = 0
	for model in models:
		try:
			metrics_model = models[model]

			if prev_group != models[model]['group']: i = 0
			else: i+=1
			
			if models[model]['group'] > len(colors)-1: 
				color_group = colors[models[model]['group']-len(colors)]
			else: 
				color_group = colors[models[model]['group']]
			marker_group = markers[i]
			plt.scatter(metrics_model['sign_sigma']*metrics_model['RMSD_norm_unb'], metrics_model['normalised_bias'], 
				label = model, color = color_group, marker = marker_group, s = 100, alpha = 0.7)
			prev_group = models[model]['group']
		except:
			traceback.print_exc()
			print ('Cannot plot model {}'.format(model))
	## Display and others
	plt.axhline(0, color='gray', linewidth = 0.8)
	plt.axvline(0, color='gray', linewidth = 0.8)

	## Add circles
	ax = plt.gca()
	circle1 =plt.Circle((0, 0), 1, linewidth = 1.4, color='gray', fill =False)
	circleMR0 =plt.Circle((0, 0), MR0, linewidth = 1.4, color='r', fill=False)
	circleMR1 =plt.Circle((0, 0), MR1, linewidth = 1.4, color='y', fill=False)
	circleMR2 =plt.Circle((0, 0), MR2, linewidth = 1.4, color='g', fill=False)
	
	circle3 =plt.Circle((0, 0), 0.01, color='g', fill=True)
	
	## Add annotations
	ax.add_artist(circle1)
	ax.annotate('R2 < 0',
				xy=(1, 0), xycoords='data',
				xytext=(-35, 10), textcoords='offset points')
	
	ax.add_artist(circleMR0)
	ax.annotate('R2 < ' + str(targetR20),
				xy=(MR0, 0), xycoords='data',
				xytext=(-35, 10), textcoords='offset points', color = 'r')
	
	ax.add_artist(circleMR1)
	ax.annotate('R2 < ' + str(targetR21),
				xy=(MR1, 0), xycoords='data',
				xytext=(-45, 10), textcoords='offset points', color = 'y')
	
	
	ax.add_artist(circleMR2)
	ax.annotate('R2 < ' + str(targetR22),
				xy=(MR2, 0), xycoords='data',
				xytext=(-45, 10), textcoords='offset points', color = 'g')
	ax.add_artist(circle3)
	
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.xlim([-1.1,1.1])
	plt.ylim([-1.1,1.1])
	plt.title('Target Diagram')
	plt.ylabel('Normalised Bias (-)')
	plt.xlabel("RMSD*'")
	plt.grid(True)
	plt.show()

	return fig

def scatterDiagram(fig, gs, n, dataframeTrain, dataframeTest):
	ax = fig.add_subplot(gs[n])

	plt.plot(dataframeTrain['reference'], dataframeTrain['prediction'], 'go', label = 'Train ' + model_name, alpha = 0.3)
	plt.plot(dataframeTest['reference'], dataframeTest['prediction'], 'bo', label = 'Test ' + model_name, alpha = 0.3)
	plt.plot(dataframeTrain['reference'], dataframeTrain['reference'], 'k', label = '1:1 Line', linewidth = 0.4, alpha = 0.3)

	plt.legend(loc = 'best')
	plt.ylabel('Prediction (-)')
	plt.xlabel('Reference (-)')