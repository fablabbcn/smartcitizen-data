from numpy import arange

# Avoid negative pollutant concentrations
avoid_negative_conc = False

# Background concentrations
background_conc = {
    'CO':   0,
    'NO2':  8,
    'O3':   40
}

# SC AlphaDelta PCB factor (mV/nA)
alphadelta_pcb = 6.36

# Deltas for baseline deltas algorithm
baseline_deltas = arange(30, 45, 5)

# Lambdas for baseline ALS algorithm
baseline_als_lambdas = [1e5]

# Alphasense sensor codes
alphasense_sensor_codes =  {
    '132':  'ASA4_CO',
    '133':  'ASA4_H2S',
    '130':  'ASA4_NO',
    '212':  'ASA4_NO2',
    '214':  'ASA4_OX',
    '134':  'ASA4_SO2',
    '162':  'ASB4_CO',
    '133':  'ASB4_H2S',#
    '130':  'ASB4_NO', #
    '202':  'ASB4_NO2',
    '204':  'ASB4_OX',
    '164':  'ASB4_SO2'
}

# Alphasense temperature channels (in order of priority)
# These are the "renamed" channels
alphasense_temp_channel = [
    "ASPT1000",
    "SHT31_EXT_TEMP",
    "SHT35_EXT_TEMP",
    "PM_DALLAS_TEMP",
    "TEMP"
]

# Alphasense pcb gains in mV/nA - TBR
as_pcb_gains = {
    'CO': 0.8,
    'H2S': 0.8,
    'SO2': 0.8,
    'NO': 0.8,
    'NO2': -0.73,
    'OX': -0.73
}

# Alphasense pcb offsets in mV
as_pcb_offsets = {
    'CO': 0,
    'H2S': 0,
    'SO2': 0,
    'NO': 20,
    'NO2': 0,
    'OX': 0,
}

# Alphasense temperature compensations
# From Tables 2 and 3 of AAN 803-04
as_t_comp = [-30, -20, -10, 0, 10, 20, 30, 40, 50]

as_sensor_algs = {
    'ASA4_CO':
                {
                    1: ['n_t',      [1.0, 1.0, 1.0, 1.0, -0.2, -0.9, -1.5, -1.5, -1.5]],
                    4: ['kpp_t',    [13, 12, 16, 11, 4, 0, -15, -18, -36]],
                },
    'ASA4_H2S':
                {
                    2: ['k_t',      [-1.5, -1.5, -1.5, -0.5, 0.5, 1.0, 0.8, 0.5, 0.3]],
                    1: ['n_t',      [3.0, 3.0, 3.0, 1.0, -1.0, -2.0, -1.5, -1.0, -0.5]]
                },
    'ASA4_NO':
                {
                    3: ['kp_t',     [0.7, 0.7, 0.7, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6]],
                    4: ['kpp_t',    [-25, -25, -25, -25, -16, 0, 56, 200, 615]]
                },
    'ASA4_NO2':
                {
                    1: ['n_t',      [0.8, 0.8, 1.0, 1.2, 1.6, 1.8, 1.9, 2.5, 3.6]],
                    3: ['kp_t',     [0.2, 0.2, 0.2, 0.2, 0.7, 1.0, 1.3, 2.1, 3.5]]
                },
    'ASA4_OX':
                {
                    3: ['kp_t',     [0.1, 0.1, 0.2, 0.3, 0.7, 1.0, 1.7, 3.0, 4.0]],
                    1: ['n_t',      [1.0, 1.2, 1.2, 1.6, 1.7, 2.0, 2.1, 3.4, 4.6]]
                },
    'ASA4_SO2':
                {
                    4: ['kpp_t',    [0, 0, 0, 0, 0, 0, 5, 25, 45]],
                    1: ['n_t',      [1.3, 1.3, 1.3, 1.2, 0.9, 0.4, 0.4, 0.4, 0.4]]
                },
    'ASB4_CO':
                {
                    1: ['n_t',      [0.7, 0.7, 0.7, 0.7, 1.0, 3.0, 3.5, 4.0, 4.5]],
                    2: ['k_t',      [0.2, 0.2, 0.2, 0.2, 0.3, 1.0, 1.2, 1.3, 1.5]]
                },
    'ASB4_H2S':
                {
                    1: ['n_t',      [-0.6, -0.6, 0.1, 0.8, -0.7, -2.5, -2.5, -2.2, -1.8]],
                    2: ['k_t',      [0.2, 0.2, 0.0, -0.3, 0.3, 1.0, 1.0, 0.9, 0.7]]
                },
    'ASB4_NO':
                {
                    2: ['k_t',      [1.8, 1.8, 1.4, 1.1, 1.1, 1.0, 0.9, 0.9, 0.8]],
                    3: ['kp_t',     [0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
                },
    'ASB4_NO2':
                {
                    1: ['n_t',      [1.3, 1.3, 1.3, 1.3, 1.0, 0.6, 0.4, 0.2, -1.5]],
                    3: ['kp_t',     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, -0.1, -4.0]]
                },
    'ASB4_OX':
                {
                    1: ['n_t',      [0.9, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.7]],
                    3: ['kp_t',     [0.5, 0.5, 0.5, 0.6, 0.6, 1.0, 2.8, 5.0, 5.3]]
                },
    'ASB4_SO2':
                {
                    4: ['kpp_t',    [-4, -4, -4, -4, -4, 0, 20, 140, 450]],
                    1: ['n_t',      [1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.9, 3.0, 5.8]]
                }
    }
