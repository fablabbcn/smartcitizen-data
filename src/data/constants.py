### ---------------------------------------
### -------------SMART CITIZEN-------------
### ---------------------------------------

# Smart Citizen API kit ids: 
station_kit_ids = [19, 21, 29]
kit_kit_id = [11, 28]

# Convertion table from API SC to Pandas
# https://stackoverflow.com/questions/35339139/where-is-the-documentation-on-pandas-freq-tags
# https://developer.smartcitizen.me/#get-historical-readings
frequency_convert_LUT = (['y','A'],
					    ['M','M'],
					    ['w','W'],
					    ['d','D'],
					    ['h','H'],
					    ['m','Min'],
					    ['s','S'],
					    ['ms','ms'])

# Alphasense ID table (Slot, Working, Auxiliary)
as_ids_table = ([1,'64','65'], 
			 [2,'61','62'], 
			 [3,'67','68'])

# External temperature table (this table is by priority)
th_ids_table = (['EXT_DALLAS','96',''], 
				 ['EXT_SHT31','79', '80'], 
				 ['SENSOR_TEMPERATURE','55','56'],
				 ['GASESBOARD_TEMPERATURE','79', '80'])

pollutant_LUT = (['CO', 28, 'ppm'],
				['NO', 30, 'ppb'],
				['NO2', 46, 'ppb'],
				['O3', 48, 'ppb'])

convertion_LUT = (['ppm', 'ppb', 1000],
					['mg/m3', 'ug/m3', 1000],
					['mg/m3', 'ppm', 24.45],
					['ug/m3', 'ppb', 24.45],
					['mg/m3', 'ppb', 1000*24.45],
					['ug/m3', 'ppm', 1./1000*24.45])

### --------------------------------------
### -------------SENSORS DATA-------------
### --------------------------------------
# Units Look Up Table - ['Pollutant', unit factor from ppm to target 1, unit factor from ppm to target 2]
alpha_factors_LUT = (['CO', 1, 0],
						['NO2', 1000, 0],
						['O3', 1000, 1000])

# AlphaDelta PCB factor (converstion from mV to nA)
factor_PCB = 6.36

# Background Concentration (model assumption)
# (from Modelling atmospheric composition in urban street canyons - Vivien Bright, William Bloss and Xiaoming Cai)
background_conc_CO = 0 # ppm
background_conc_NO2 = 8 # ppb
background_conc_OX = 40 # ppb

# Filter Smoothing 
filter_exponential_smoothing = 0.2