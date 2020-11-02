# Hardware description

Below is the explanation of hardware description in pseudo-json format.

{"SCTEST":
  {
    # Sensor blueprint as in blueprints.yaml
    "blueprint": "sc_21_station_box",
    # Hardware versions (sensors)
    "versions": {
                    "1":
                        {
                            # Date from
                            "from": "2020-10-28",
                            # Date to
                            "to": null,
                            # Sensor IDS - Sensors are calculated in sequential order - if a sensor has cross-sensitivity, needs to be placed after (i.e. O3 after NO2)
                            "ids":
                                {
                                "AS_48_01": "162031254",
                                "AS_48_23": "162031256",
                                "AS_49_01": "162031257",
                                "AS_49_23": "162581706"
                                }
                        }
                }
  }
}