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

# This look-up table is comprised of channels you want always want to have with the same units and that might come from different sources
# i.e. pollutant data in various units (ppm or ug/m3) from different analysers
# The table should be used as follows:
# (['int_channel', 'molecular_weight', 'target_unit'], ...)
# - int_channel is the internal channel that will lately be used in the analysis. This has to be defined in the target_channel_names when creating the test (see below)
# - molecular_weight is only for chemical components. It can be left to 1 for other types of signals
# - target_unit is the unit you want this channel to be and that will be converted in case of it being found in the channels list of your source

channel_LUT = (['CO', 28, 'ppm'],
				['NO', 30, 'ppb'],
				['NO2', 46, 'ppb'],
				['O3', 48, 'ppb'])

# Target channel name definition
# This dict has to be specified when you create a test.
# 'channels': {'source_channel_names' : ('air_temperature_celsius', 'battery_percent', 'calibrated_soil_moisture_percent', 'fertilizer_level', 'light', 'soil_moisture_percent', 'water_tank_level_percent'), 
#              'units' : ('degC', '%', '%', '-', 'lux', '%', '%'),
#              'target_channel_names' : ('TEMP', 'BATT', 'Cal Soil Moisture', 'Fertilizer', 'Soil Moisture', 'Water Level')
# source_channel_names: is the actual name you can find in your csv file
# units: the units of this channel. These will be converted using the LUT below
# target_channel_names: how you want to name your channels after the convertion. A suffix ('_CONV') will be added to them in case they are matching the source csv names

# Can be targetted to convert the units with the channel_LUT below
# This table is used to convert units
# ['from_unit', 'to_unit', 'multiplicative_factor']
# - 'from_unit'/'to_unit' = 'multiplicative_factor'
# It accepts reverse operations - you don't need to put them twice but in reverse
convertion_LUT = (['ppm', 'ppb', 1000],
					['mg/m3', 'ug/m3', 1000],
					['mgm3', 'ugm3', 1000],
					['mg/m3', 'ppm', 24.45],
					['mgm3', 'ppm', 24.45],
					['ug/m3', 'ppb', 24.45],
					['ugm3', 'ppb', 24.45],
					['mg/m3', 'ppb', 1000*24.45],
					['mgm3', 'ppb', 1000*24.45],
					['ug/m3', 'ppm', 1./1000*24.45],
					['ugm3', 'ppm', 1./1000*24.45])

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