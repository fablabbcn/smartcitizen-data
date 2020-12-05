# Hardware description

Below is the explanation of hardware description in pseudo-json format. Each hardware now is a json, in order to have an url for each:

```
{
    "1":
        {
            # Date from
            "from": "2020-10-28",
            # Date to
            "to": null,
            # Description
            "description": "2PMS5003-4ELEC-GPS",
            # Sensor IDS - Sensors are calculated in sequential order - if a sensor has cross-sensitivity, needs to be placed after (i.e. O3 after NO2)
            "ids":
                {
                "AS_48_01": "162031254",
                "AS_48_23": "162031256",
                "AS_4A_01": "162031257",
                "AS_4A_23": "162581706"
                }
        }
}
```

## Device Names

_Examples_:
**SCAS210001**: Smart Citizen Air Station Kit 2.1 #1

## Description

2PMS5003-4ELEC-GPS-CO2-UVX
   ├──-----├──--├──-├──-└── UV Index sensor
   ├──-----├──--├──-└──---- CO2
   ├──-----├──--└──-------- GPS
   ├──-----└──------------- Electrochemical sensors
   └──--------------------- PM Sensor