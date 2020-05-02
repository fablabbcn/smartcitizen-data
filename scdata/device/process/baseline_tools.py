import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import pandas as pd
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats.stats import linregress
from scipy.optimize import curve_fit

import datetime
from dateutil import relativedelta
from src.models.formulas import exponential_smoothing

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt, isnan

from src.data.constants import *



def find_max(iterable = list()):
	'''
		Input: list to obtain maximum value
		Output: value and index of maximum in the list
	'''
	
	value = np.max(iterable)
	index = np.argmax(iterable)
	
	return value, index

def createBaselines(_dataBaseline, baselineType, hyperparameters, _type_regress = 'linear', _plots = False, _verbose = False):
	'''
		Input:
			_dataBaseline: dataframe containing signal and index to be baselined
			_type_regress= 'linear', 'exponential', 'best' (based on p_value of both)
			_numberDeltas : vector of floats for deltas (N periods)
			_plots:  display plots or not
			_verbose: print info or not
			_type_regress: regression type (linear, exp, ... )
		Output:
			baseline: pandas dataframe baseline
		TODO:
			implement other types of regression
	'''
	
	resultData = _dataBaseline.copy()

	name = resultData.iloc[:,0].name
	pearsons =[]
	
	if _plots: figX, axX = plt.subplots(figsize=(20,8))

	if baselineType == 'deltas':
		_numberDeltas = hyperparameters
		
		for delta in _numberDeltas:
			resultData[(name +'_' +str(delta))] = ExtractBaseline(resultData.iloc[:,0], delta)
			if _plots: axX.plot(resultData.iloc[:,0].index, resultData[(name +'_' +str(delta))], label = resultData[(name +'_' +str(delta))].name)

			_, _, r_value, _, _ = linregress(np.transpose(resultData[(name +'_' +str(delta))]), np.transpose(resultData.iloc[:,1].values))

			pearsons.append(r_value)

	elif baselineType == 'als':
		lambdaALSs = hyperparameters[0]
		pALS = hyperparameters[1]

		for lambdaALS in lambdaALSs:
			resultData[(name +'_' +str(lambdaALS))] = ExtractBaseline_ALS(resultData.iloc[:,0], lambdaALS, pALS)

			if _plots: axX.plot(resultData.iloc[:,0].index, resultData[(name +'_' +str(lambdaALS))], label = resultData[(name +'_' +str(lambdaALS))].name)

			# slope, intercept, r_value, p_value, std_err = linregress(np.transpose(resultData[(name +'_' +str(delta))]), np.transpose(resultData.iloc[:,1].values))
			_, _, r_value, _, _ = linregress(np.transpose(resultData[(name +'_' +str(lambdaALS))]), np.transpose(resultData.iloc[:,1].values))
			pearsons.append(r_value)
	
	if _plots:
		axX.plot(resultData.iloc[:,0].index, resultData.iloc[:,0], label = 'Working Electrode')
		axX.plot(resultData.iloc[:,0].index, resultData.iloc[:,1], label = resultData.iloc[:,1].name)
		
		axX.axis('tight')
		axX.legend(loc='best')
		axX.set_xlabel('Date')
		axX.set_ylabel('Baselines')
		axX.grid(True)
	
		plt.show()
	## Find Max in the pearsons - correlation can be negative, so use absolute of the pearson
	valMax, indexMax = findMax(np.abs(pearsons))

	if baselineType == 'deltas': listIterations = _numberDeltas
	elif baselineType == 'als' : listIterations = lambdaALSs

	# print (valMax, indexMax)
	if _type_regress == 'linear':
		## Fit with y = A + Bx
		slope, intercept, r_value, p_value, std_err = linregress(np.transpose(resultData.iloc[:,1].values),resultData[(name + '_'+str(listIterations[indexMax]))])
		baseline = intercept + slope*resultData.iloc[:,1].values
		# print (r_value)
	elif _type_regress == 'exponential':
		## Fit with y = Ae^(Bx) -> logy = logA + Bx
		logy = np.log(resultData[(name + '_'+str(listIterations[indexMax]))])
		slope, intercept, r_value, p_value, std_err = linregress(np.transpose(resultData.iloc[:,1].values), logy)
		baseline = exponential_func(np.transpose(resultData.iloc[:,1].values), np.exp(intercept), slope, 0)
		# print (r_value)
	elif _type_regress == 'best':
		## Find linear r_value
		slope_lin, intercept_lin, r_value_lin, p_value_lin, std_err_lin = linregress(np.transpose(resultData.iloc[:,1].values),resultData[(name + '_'+str(listIterations[indexMax]))])
		
		## Find Exponential r_value
		logy = np.log(resultData[(name + '_'+str(listIterations[indexMax]))])
		slope_exp, intercept_exp, r_value_exp, p_value_exp, std_err_exp = linregress(np.transpose(resultData.iloc[:,1].values), logy)
		
		## Pick which one is best
		if ((not isnan(r_value_exp)) and (not isnan(r_value_lin))):
			if r_value_lin > r_value_exp:
				if _verbose:
					print ('Using linear regression')
				baseline = intercept_lin + slope_lin*resultData.iloc[:,1].values
			else:
				if _verbose:
					print ('Using exponential regression')
				baseline = exponential_func(np.transpose(resultData.iloc[:,1].values), np.exp(intercept_exp), slope_exp, 0)
		elif not isnan(r_value_lin):
			if _verbose:
				print ('Using linear regression')
			baseline = intercept_lin + slope_lin*resultData.iloc[:,1].values
		elif not isnan(r_value_exp):
			if _verbose:
				print ('Using exponential regression')
			baseline = exponential_func(np.transpose(resultData.iloc[:,1].values), np.exp(intercept_exp), slope_exp, 0)
		else:
			print ('Exponential and linear regression are nan')
	
	# Put baseline in the dataframe
	resultData['Zero'] = 0
	# Avoid baselines higher than the working electrode and lower than zero
	resultData[(name + '_' + 'baseline_raw' +  _type_regress)] = baseline
	resultData[(name + '_' + 'baseline_min' + _type_regress)] = resultData[[(name + '_' + 'baseline_raw' +  _type_regress), resultData.iloc[:,0].name]].min(axis=1)
	resultData[(name + '_' + 'baseline_' + _type_regress)] = resultData[[(name + '_' + 'baseline_min' + _type_regress), 'Zero']].max(axis = 1)
	
	if _plots == True:
		with plt.style.context('seaborn-white'):
			fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8))
			
			ax1.plot(resultData.iloc[:,1].values, resultData[(name + '_'+str(listIterations[indexMax]))], label = 'Baseline ' + str(listIterations[indexMax]), linewidth=0, marker='o')
			ax1.plot(resultData.iloc[:,1].values, resultData[(name + '_' + 'baseline_' +  _type_regress)] , label = 'Regressed value', linewidth=0, marker='o')
			legend = ax1.legend(loc='best')
			ax1.set_xlabel(resultData.iloc[:,1].name)
			ax1.set_ylabel('Regression values')
			ax1.grid(True)
			
			lns1 = ax2.plot(resultData.iloc[:,0].index, resultData.iloc[:,0].values, label = "Actual", linestyle=':', linewidth=1, marker=None)
			#[ax2.plot(resultData.index, resultData[(name +'_' +str(delta))].values, label="Delta {}".format(delta), marker=None,  linestyle='-', linewidth=1) for delta in _numberDeltas]
			lns2 = ax2.plot(resultData.index, resultData[(name + '_' + 'baseline_' +  _type_regress)], label='Baseline', marker = None)

			ax2.axis('tight')
			ax2.set_title("Baseline Extraction")
			ax2.grid(True)
			
			ax22 = ax2.twinx()
			lns22 = ax22.plot(resultData.index, resultData.iloc[:,1].values, color = 'red', label = resultData.iloc[:,1].name, linestyle='-', linewidth=1, marker=None)
			ax22.set_ylabel(resultData.iloc[:,1].name, color = 'red')
			ax22.set_ylim(ax2.get_ylim())
			ax22.tick_params(axis='y', labelcolor='red')

			lns = lns1+lns2+lns22
			labs = [l.get_label() for l in lns]
			ax2.legend(lns, labs, loc='best')
			
			fig2, ax3 = plt.subplots(figsize=(20,8)) # two axes on figure
			ax3.plot(listIterations, pearsons)

			if baselineType == 'deltas':
				ax3.set_title("R2 vs. Delta")
				ax3.set_xlabel('Delta')

			elif baselineType == 'als':
				ax3.set_title("R2 vs. λ")
				ax3.set_xlabel('λ')
				ax3.set_xscale('log')

			ax3.axis('tight')
			ax3.set_ylabel('R2')
			ax3.grid(True)
			plt.show()

	return resultData[(name + '_' + 'baseline_' +  _type_regress)], indexMax

def decompose(_data, _plots = False):
	'''
			Function to decompose a signal into it's trend and normal variation
			Input:
				_data: signal to decompose
				_plots: print plots or not (default False)
			Output:
				DataDecomp = _data - slope*_data.index
				slope, intercept = linear regression coefficients
	'''
	indexDecomp = np.arange(len(_data))

	slope, intercept, r_value, p_value, std_err = linregress(indexDecomp, np.transpose(_data.values))
	dataDecomp=pd.DataFrame(index = _data.index)
	name = _data.name
	result = []
	
	for n in range(len(_data)):
		result.append(float(_data.values[n]-slope*n))
	dataDecomp[(name + '_' + '_flat')] = result
	
	trend = slope*indexDecomp + intercept
	if _plots == True:
		
		with plt.style.context('seaborn-white'):
			fig, ax = plt.subplots(figsize=(20,10))
			ax.plot(_data.index, _data.values, label = "Actual", marker = None)
			ax.plot(_data.index, dataDecomp[(name + '_' +'_flat')], marker = None, label = 'Flattened')
			ax.plot(_data.index, trend, label = 'Trend')
			ax.legend(loc="best")
			ax.axis('tight')
			ax.set_title("Signal Decomposition - "+ name)
			ax.set_xlabel('Index')
			ax.set_ylabel('Signal')
			ax.grid(True)

			plt.show()
			
	return dataDecomp, slope, intercept

def calculateBaselineDay(_dataFrame, _listNames, baselineType, hyperparameters, _type_regress, _plots = False, _verbose = True):
	'''
		Function to calculate day-based baseline corrections
		Input:
			_dataFrame: pandas dataframe with datetime index containing 1 day of measurements + overlap
			_listNames: list containing column names of /WE / AE / TEMP / HUM) or (MICS / TEMP / HUM)
			_baselined: channel to calculate the baseline of
			_baseliner: channel to use as input for the baselined (baselined = f(baseliner)) 
			_type_regress: type of regression to perform (linear, exponential, best)
			_plot: plot analytics or not
			_verbose: print analytics or not
		Output:
			_data_baseline: dataframe with baseline
			_baseline_corr: metadata containing analytics for long term analysis
	'''
	
	## Create Baselines

	dataframeCalc = _dataFrame.copy()

	data_baseline, indexMax = createBaselines(dataframeCalc, baselineType, hyperparameters, _type_regress, _plots, _verbose)

	## COMMENTED FOR SIMPLICITY - NOT USED
	# ## Un-pack list names
	# alphaW, alphaA, temp, hum = _listNames

	# ## Correlation between Baseline and original auxiliary
	# slopeAuxBase, interceptAuxBase, rAuxBase, pAuxBase, std_errAuxBase = linregress(np.transpose(data_baseline.values), np.transpose(dataframeCalc.iloc[:,1].values))

	# # Add metadata for further research
	# deltaAuxBase_avg = np.mean(data_baseline.values-dataframeCalc.iloc[:,1].values)
	# ratioAuxBase_avg = np.mean(data_baseline.values/dataframeCalc.iloc[:,1].values)
   
	# # Pre filter based on the metadata itself
	# if slopeAuxBase > 0 and rAuxBase > 0.3:
	# 	valid = True
	# else:
	# 	valid = False
   
	# corrMetrics = {}

	# corrMetrics['slopeAuxBase'] = slopeAuxBase
	# corrMetrics['rAuxBase'] = rAuxBase
	# corrMetrics['deltaAuxBase_avg'] = deltaAuxBase_avg
	# corrMetrics['ratioAuxBase_avg'] = ratioAuxBase_avg
	# corrMetrics['indexMax'] = indexMax
	# corrMetrics['0_valid'] = valid

	# if _verbose == True:
		
	# 	print ('-------------------')
	# 	print ('Auxiliary Electrode')
	# 	print ('-------------------')
	# 	print ('Correlation coefficient of Baseline and Original auxiliary: {}'.format(rAuxBase))
		
	# 	print ('Average Delta: {} \t Average Ratio: {}'.format(deltaAuxBase_avg, deltaAuxBase_avg))

	if _plots == True:
		with plt.style.context('seaborn-white'):
			
			fig2, ax3 = plt.subplots(figsize=(20,8))

			ax3.plot(data_baseline.index, data_baseline.values, label='Baseline', marker = None)
			ax3.plot(dataframeCalc.index, dataframeCalc.iloc[:,0], label='Original Working', marker = None)
			ax3.plot(dataframeCalc.index, dataframeCalc.iloc[:,1], label='Original Auxiliary', marker = None)

			ax3.legend(loc="best")
			ax3.axis('tight')
			ax3.set_title("Baseline Not Compensated")
			ax3.set(xlabel='Time', ylabel='Ouput-mV')
			ax3.grid(True)
			plt.show()

	return data_baseline #, corrMetrics

def findDates(_dataframe):
	'''
		Find minimum, maximum dates in the dataframe and the amount of days in between
		Input: 
			_dataframe: pandas dataframe with datetime index
		Output: 
			rounded up min day, floor max day and number of days between the min and max dates
	'''
	range_days = (_dataframe.index.max()-_dataframe.index.min()).days
	min_date_df = _dataframe.index.min().floor('D')
	max_date_df = _dataframe.index.max().ceil('D')
	
	return min_date_df, max_date_df, range_days

def calculatePollutantsAlpha(_dataframe, _sensorIDs, _variables, _refAvail, _dataframeRef, _type_regress = 'best', _filterExpSmoothing = 0.2, _plotsInter = False, _plotResult = True, _verbose = False, _printStats = False, _calibrationDataPath = '', _currentSensorNames = ''):

	dataframeResult = _dataframe.copy()
	correlationMetrics = dict()

	alpha_calData = getCalData('alphasense', _calibrationDataPath)
	
	for pollutant in _sensorIDs.keys():
		
		# Get Sensor 
		sensorID = _sensorIDs[pollutant][0]
		slot = _sensorIDs[pollutant][1]

		method = _variables[pollutant][0]
		if method == 'baseline':
			baselineSignal = _variables[pollutant][1]
			baselineType = _variables[pollutant][2]
			hyperparameters = _variables[pollutant][3]
			overlapHours = _variables[pollutant][4]
			append = _variables[pollutant][5]
			try:
				clean_negatives = _variables[pollutant][6]
			except:
				clean_negatives = False
				pass

		else:
			append = _variables[pollutant][2]
			try:
				clean_negatives = _variables[pollutant][3]
			except:
				clean_negatives = False
				pass

		# Give name to pollutant column
		pollutant_column = (pollutant + '_' + append)
		print (f'Final pollutant name: {pollutant_column}')
		
		# Get Sensor data
		Sensitivity_1 = alpha_calData.loc[sensorID,'Sensitivity 1']
		Sensitivity_2 = alpha_calData.loc[sensorID,'Sensitivity 2']
		Target_1 = alpha_calData.loc[sensorID,'Target 1']
		Target_2 = alpha_calData.loc[sensorID,'Target 2']
		nWA = alpha_calData.loc[sensorID,'Zero Current']/alpha_calData.loc[sensorID,'Aux Zero Current']

		if not Target_1 == pollutant:
			print ('Sensor ID ({}) and pollutant type ({}) not matching'.format(Target_1, pollutant))
			return

		## Retrieve alphasense ids from table (for the slot we are calculating)
		for item in as_ids_table:
			if slot == item[0]:
				id_alphaW = item[1]
				id_alphaA = item[2]

		## Retrieve alphasense name
		for currentSensorName in _currentSensorNames:        
			if id_alphaW == _currentSensorNames[currentSensorName]['id']:
				alphaW = _currentSensorNames[currentSensorName]['shortTitle']
			if id_alphaA == _currentSensorNames[currentSensorName]['id']:
				alphaA = _currentSensorNames[currentSensorName]['shortTitle']
	   
		# Retrieve temperature and humidity names
		temp = ''
		hum = ''
		for item in th_ids_table:
			for currentSensorName in _currentSensorNames:      
				if temp == '' and item[1] == _currentSensorNames[currentSensorName]['id'] and _currentSensorNames[currentSensorName]['shortTitle'] in dataframeResult.columns:
					temp = _currentSensorNames[currentSensorName]['shortTitle']
					break
		
		for item in th_ids_table:
			for currentSensorName in _currentSensorNames:   
				if hum == '' and item[2] == _currentSensorNames[currentSensorName]['id'] and _currentSensorNames[currentSensorName]['shortTitle'] in dataframeResult.columns:
					hum = _currentSensorNames[currentSensorName]['shortTitle']
					break

		_listNames = (alphaW, alphaA, temp, hum)

		if pollutant == 'O3':
			# Check if NO2 is already present in the dataset
			if ('NO2'+ '_' + append) in dataframeResult.columns:
				pollutant_column_2 = ('NO2' + '_' + append)
			else:
				print ('Change list order to [(CO,sensorID_CO, ...), (NO2, sensorID_NO2, ...), (O3, sensorID_O3, ...)]')
				return
		
		# Get units for the pollutant in questions
		for pollutantItem in alphaUnitsFactorsLUT:
			
			if pollutant == pollutantItem[0]:
				factor_unit_1 = pollutantItem[1]
				factor_unit_2 = pollutantItem[2]

		## Find min, max and range of days
		min_date_df, max_date_df, range_days = findDates(dataframeResult)
		if _verbose:
			print ('Calculation of ' + '\033[1m{:10s}\033[0m'.format(pollutant))
			print ('Data Range from {} to {} with {} days'.format(min_date_df, max_date_df, range_days))

		if range_days == 0: print ('No data between these days'); continue
		
		if method == 'baseline':
			# Select baselined - baseliner depending on the baseline type
			if baselineSignal == 'single_temp': baseliner = temp
			elif baselineSignal == 'single_hum': baseliner = hum
			elif baselineSignal == 'single_aux': baseliner = alphaA
			
			baselined = alphaW
			if _verbose:
				print ('Baseline method {}'. format(baselineType))
				print ('Using {} as baselined channel'.format(baselined))
				print ('Using {} as baseliner channel'.format(baseliner))
			
			# Iterate over days
			for day in range(range_days):
			
				# Calculate overlap dates for that day
				min_date_ovl = max(_dataframe.index.min(), (min_date_df + pd.DateOffset(days=day) - pd.DateOffset(hours = overlapHours)))
				max_date_ovl = min(_dataframe.index.max(), (min_date_ovl + pd.DateOffset(days=1) + pd.DateOffset(hours = overlapHours + relativedelta.relativedelta(min_date_df + pd.DateOffset(days=day),min_date_ovl).hours)))
				
				# Calculate non overlap dates for that day
				min_date_novl = max(min_date_df, (min_date_df + pd.DateOffset(days=day)))
				max_date_novl = min(max_date_df, (min_date_novl + pd.DateOffset(days=1)))
			
				if _verbose: print ('Calculating day {}, with range: {} \t to {}'.format(day, min_date_ovl, max_date_ovl))

				## Trim dataframe to overlap dates
				dataframeTrim = dataframeResult[dataframeResult.index > min_date_ovl]
				dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_ovl]

				# Make 
				dataframeBaseline = dataframeTrim.loc[:,(baselined, baseliner)].dropna(how = 'any', axis = 0)

				# Init stuff
				if day == 0: corrMetrics = list() # Init corrMetrics
				
				corrMetricsTrim = dict()	
				if dataframeBaseline.empty:
					
					corrMetricsTrim['0_valid'] = False
					if _verbose: print ('No data between these dates' )
				
				else:

					# CALCULATE THE BASELINE PER DAY
					# dataframeTrim[alphaW + '_BASELINE_' + _append], corrMetricsTrim = calculateBaselineDay(dataframeBaseline, _listNames, baselineType, hyperparameters, _type_regress, _plotsInter, _verbose)
					dataframeTrim[alphaW + '_BASELINE_' + append] = calculateBaselineDay(dataframeBaseline, _listNames, baselineType, hyperparameters, _type_regress, _plotsInter, _verbose)

					# TRIM IT TO NO-OVERLAP
					dataframeTrim = dataframeTrim[dataframeTrim.index > min_date_novl]
					dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_novl]

					# CALCULATE ACTUAL POLLUTANT CONCENTRATION
					if pollutant == 'CO': # Not recommended for CO
						dataframeTrim[pollutant_column + '_PRE'] = backgroundConc_CO + factor_unit_1*factorPCB*(dataframeTrim[alphaW] - dataframeTrim[alphaW + '_BASELINE_' + append])/abs(Sensitivity_1)
					elif pollutant == 'NO2':
						dataframeTrim[pollutant_column + '_PRE'] = backgroundConc_NO2 + factor_unit_1*factorPCB*(dataframeTrim[alphaW] - dataframeTrim[alphaW + '_BASELINE_' + append])/abs(Sensitivity_1)
					elif pollutant == 'O3':
						dataframeTrim[pollutant_column + '_PRE'] = backgroundConc_OX + factor_unit_1*(factorPCB*(dataframeTrim[alphaW] - dataframeTrim[alphaW + '_BASELINE_' + append]) - (dataframeTrim[pollutant_column_2])/factor_unit_2*abs(Sensitivity_2))/abs(Sensitivity_1)
					dataframeResult = dataframeResult.combine_first(dataframeTrim)
				
					# Extract metrics from reference
					if _refAvail:
						pollutant_ref = (pollutant + '_' + ref_append)

						## Trim ref dataframe to no-overlap dates
						prediction = dataframeTrim[pollutant_column + '_PRE']
						reference = _dataframeRef[pollutant_ref][_dataframeRef.index >= dataframeTrim.index.min()]

						dataframeRefTrim = pd.concat([prediction, reference], axis = 1)	
						dataframeRefTrim.dropna(axis = 0, inplace = True, how = 'any')

						if pollutant_ref in dataframeRefTrim.columns:
							corrMetricsTrim['r2_ref'] = r2_score(dataframeRefTrim[pollutant_column+ '_PRE'], dataframeRefTrim[pollutant_ref]) if not dataframeRefTrim.empty else np.nan
							corrMetricsTrim['rmse'] = sqrt(mean_squared_error(dataframeRefTrim[pollutant_column+ '_PRE'], dataframeRefTrim[pollutant_ref])) if not dataframeRefTrim.empty else np.nan
							corrMetricsTrim['bias'] = (dataframeRefTrim[pollutant_column+ '_PRE'] - dataframeRefTrim[pollutant_ref]).mean() if not dataframeRefTrim.empty else np.nan
							corrMetricsTrim['reference_avg'] = dataframeRefTrim[pollutant_ref].mean() if not dataframeRefTrim.empty else np.nan
					else:
						if _verbose:
							print ('No Ref available')
					
					## Get some metrics 
					corrMetricsTrim['temp_avg'] = dataframeTrim[temp].mean() if not dataframeTrim.empty else np.nan
					corrMetricsTrim['hum_avg'] = dataframeTrim[hum].mean() if not dataframeTrim.empty else np.nan
					corrMetricsTrim['prediction_avg'] = dataframeTrim[pollutant_column+ '_PRE'].mean() if not dataframeTrim.empty else np.nan
					# corrMetricsTrim['pollutant_std'] = dataframeTrim[pollutant_column].std() if not dataframeTrim.empty else np.nan
					# corrMetricsTrim['pollutant_min'] = dataframeTrim[pollutant_column].min() if not dataframeTrim.empty else np.nan
					# corrMetricsTrim['pollutant_max'] = dataframeTrim[pollutant_column].max() if not dataframeTrim.empty else np.nan
				
				corrMetrics.append(corrMetricsTrim)
			
			correlationMetrics_df = pd.DataFrame(corrMetrics, index = [(min_date_df+ pd.DateOffset(days=day)).strftime('%Y-%m-%d') for day in range(range_days)])

			# Eliminate negative values and unnecessary columns
			dataframeResult[pollutant_column] = dataframeResult[pollutant_column + '_PRE'][dataframeResult[pollutant_column + '_PRE'] > 0].dropna()
			dataframeResult.drop(pollutant_column + '_PRE', axis=1, inplace=True)
			dataframeResult[pollutant_column + '_FILTER'] = exponential_smoothing(dataframeResult[pollutant_column].fillna(0), filterExpSmoothing)
					
		elif method == 'classic' or method == 'classic_no_zero':

			if method == 'classic':
				factor_zero_current = 1
			elif method == 'classic_no_zero':
				factor_zero_current = nWA
			## corrMetrics
			corrMetricsTrim = list()
			dataframeResult ['Zero'] = 0
			
			if pollutant == 'CO': 
				dataframeResult[pollutant_column + '_PRE'] = factor_unit_1*factorPCB*(dataframeResult[alphaW] - nWA/factor_zero_current*dataframeResult[alphaA])/abs(Sensitivity_1)+backgroundConc_CO
			elif pollutant == 'NO2':
				dataframeResult[pollutant_column + '_PRE'] = factor_unit_1*factorPCB*(dataframeResult[alphaW] - nWA/factor_zero_current*dataframeResult[alphaA])/abs(Sensitivity_1)+backgroundConc_NO2
			elif pollutant == 'O3':
				dataframeResult[pollutant_column + '_PRE'] = factor_unit_1*(factorPCB*(dataframeResult[alphaW] - nWA/factor_zero_current*dataframeResult[alphaA]) - (dataframeResult[pollutant_column_2])/factor_unit_2*abs(Sensitivity_2))/abs(Sensitivity_1) + backgroundConc_OX
			
			# Eliminate negative values and unnecessary columns
			if clean_negatives == True:
				dataframeResult[pollutant_column] = dataframeResult[pollutant_column + '_PRE'][dataframeResult[pollutant_column + '_PRE'] > 0].dropna()
			else:
				dataframeResult[pollutant_column] = dataframeResult[pollutant_column + '_PRE']
			
			dataframeResult.drop(pollutant_column + '_PRE', axis=1, inplace=True)
			dataframeResult[pollutant_column + '_FILTER'] = exponential_smoothing(dataframeResult[pollutant_column].fillna(0), filterExpSmoothing)

			## Calculate stats day by day to avoid stationarity
			min_date_df, max_date_df, range_days = findDates(dataframeResult)
			print ('Data Range from {} to {} with {} days'.format(min_date_df, max_date_df, range_days))

			for day in range(range_days):
				## corrMetrics
				corrMetricsTrim = {}

				# Calculate non overlap dates for that day
				min_date_novl = max(min_date_df, (min_date_df + pd.DateOffset(days=day)))
				max_date_novl = min(max_date_df, (min_date_novl + pd.DateOffset(days=1)))
				
				## Trim dataframe to no-overlap dates
				dataframeTrim = dataframeResult[dataframeResult.index > min_date_novl]
				dataframeTrim = dataframeTrim[dataframeTrim.index <= max_date_novl]
					
				# Extract metrics from reference
				if _refAvail:
					pollutant_ref = (pollutant + '_' + ref_append)

					## Trim ref dataframe to no-overlap dates
					prediction = dataframeResult[pollutant_column]
					reference = _dataframeRef[pollutant_ref][_dataframeRef.index >= dataframeTrim.index.min()]

					dataframeRefTrim = pd.concat([prediction, reference], axis = 1)	
					dataframeRefTrim.dropna(axis = 0, inplace = True, how = 'any')

					if pollutant_ref in dataframeRefTrim.columns:
						corrMetricsTrim['r2_ref'] = r2_score(dataframeRefTrim[pollutant_column], dataframeRefTrim[pollutant_ref]) if not dataframeRefTrim.empty else np.nan
						corrMetricsTrim['rmse'] = sqrt(mean_squared_error(dataframeRefTrim[pollutant_column ], dataframeRefTrim[pollutant_ref])) if not dataframeRefTrim.empty else np.nan
						corrMetricsTrim['bias'] = (dataframeRefTrim[pollutant_column ] - dataframeRefTrim[pollutant_ref]).mean() if not dataframeRefTrim.empty else np.nan
						corrMetricsTrim['reference_avg'] = dataframeRefTrim[pollutant_ref].mean() if not dataframeRefTrim.empty else np.nan
				else:
					if _verbose:
						print ('No Ref available')
					
				## Get some metrics 
				corrMetricsTrim['temp_avg'] = dataframeTrim[temp].mean() if not dataframeTrim.empty else np.nan
				corrMetricsTrim['hum_avg'] = dataframeTrim[hum].mean() if not dataframeTrim.empty else np.nan
				corrMetricsTrim['prediction_avg'] = dataframeTrim[pollutant_column].mean() if not dataframeTrim.empty else np.nan
				# corrMetricsTrim['pollutant_std'] = dataframeTrim[pollutant_column].std() if not dataframeTrim.empty else np.nan
				# corrMetricsTrim['pollutant_min'] = dataframeTrim[pollutant_column].min() if not dataframeTrim.empty else np.nan
				# corrMetricsTrim['pollutant_max'] = dataframeTrim[pollutant_column].max() if not dataframeTrim.empty else np.nan
				
				if day == 0: corrMetrics = list()
				corrMetrics.append(corrMetricsTrim)
			
			correlationMetrics_df = pd.DataFrame(corrMetrics, index = [(min_date_df+ pd.DateOffset(days=day)).strftime('%Y-%m-%d') for day in range(range_days)])
			
			# SHOW SOME METADATA FOR THE BASELINES FOUND
			if _printStats:
				display(correlationMetrics_df)
		
		## RETRIEVE METADATA
		correlationMetrics[pollutant_column] = correlationMetrics_df
		
		# PLOT THINGS IF REQUESTED
		if _plotResult:

			fig1 = tls.make_subplots(rows=4, cols=1, shared_xaxes=True, print_grid=False)
			
			fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaW], 'type': 'scatter', 'line': dict(width = 2), 'name': dataframeResult[alphaW].name}, 1, 1)
			fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaA], 'type': 'scatter', 'line': dict(width = 2), 'name': dataframeResult[alphaA].name}, 1, 1)
			fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaA] * nWA, 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': 'AuxCor Alphasense'}, 1, 1)
			if method == 'baseline':
				fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[alphaW + '_BASELINE_' + append], 'type': 'scatter', 'line': dict(width = 2), 'name': 'Baseline'}, 1, 1)
			
			fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[pollutant_column], 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': dataframeResult[pollutant_column].name}, 2, 1)
			fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[pollutant_column + '_FILTER'], 'type': 'scatter', 'name': (dataframeResult[pollutant_column + '_FILTER'].name)}, 2, 1)
			
			if _refAvail:
				# take the reference and check if it's available
				pollutant_ref = pollutant + '_' + ref_append
				if pollutant_ref in _dataframeRef.columns:
					# If all good, plot it
					fig1.append_trace({'x': _dataframeRef.index, 'y': _dataframeRef[pollutant_ref], 'type': 'scatter', 'name': _dataframeRef[pollutant_ref].name}, 2, 1)
				
			fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[temp], 'type': 'scatter', 'line': dict(width = 1, dash = 'dot'), 'name': dataframeResult[temp].name}, 3, 1)
			fig1.append_trace({'x': dataframeResult.index, 'y': dataframeResult[hum], 'type': 'scatter', 'name': (dataframeResult[hum].name)}, 4, 1)
			
			fig1['layout'].update(height = 1500, 
								  legend=dict(x=-.1, y=0.9), 
								  xaxis=dict(title='Time'), 
								  title = 'Baseline Correction for {}'.format(pollutant),
								  yaxis1 = dict(title='Sensor Output - mV'), 
								  yaxis2 = dict(title='Pollutant - ppm'),
								  yaxis3 = dict(title='Temperature - degC'),
								  yaxis4 = dict(title='Humidity - %'),
								 )
								   
			ply.offline.iplot(fig1)

	return dataframeResult, correlationMetrics