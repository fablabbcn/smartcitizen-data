# Calibrations file description

## Alphasense Sensors

```
'SENSORID':
  # Electronic zero in mV for aux. electrode - i.e. ISB offset without sensor
  ae_electronic_zero_mv: ''
  # Aux sensor zero in mV - i.e. sensor withou ISB
  ae_sensor_zero_mv: '-20.8'
  # Sensor + ISB total aux. zero in mV
  ae_total_zero_mv: ''
  # PCB gain
  pcb_gain_mv_na: ''
  # Working electrode cross sensitivity to NO2 in mV
  we_cross_sensitivity_no2_mv_ppb: ''
  # Working electrode cross sensitivity to NO2 in nA  
  we_cross_sensitivity_no2_na_ppb: '0'
  # Electronic zero in mV for working electrode - i.e. ISB offset without sensor
  we_electronic_zero_mv: ''
  # Working electrode sensitivity in mV/ppb
  we_sensitivity_mv_ppb: ''
  # Working electrode sensitivity in na/ppb
  we_sensitivity_na_ppb: '568.3'
  # Working sensor zero in mV - i.e. sensor withou ISB  
  we_sensor_zero_mv: '-34'
  # Sensor + ISB total working zero in mV
  we_total_zero_mv: ''
```

## Alphasense AFE Boards

```
'BOARDID':
  # t20 - t20 temperature for v20 voltage level
  t20: ''
  # v20 - v20 voltage factor - voltage at which the t20 is seen
  v20: ''
```

Formula (PT1000 is the positive voltage reading of the PT1000 in mV):

k = cal_data['v20'] - (cal_data['t20'] - 20.0) / 1000.0 # in mV
T = (PT1000 - k) * 1000.0 + 20.0 # in degC

### Connections to the SmartCitizen ASB

When using the ASB 4-ch, the measurements are as follows:

- Working electrode is either CH0 or CH2 of the ADC (depending on the row)  
- Auxiliary electrode is either CH1 or CH3 of the ADC (same as above)