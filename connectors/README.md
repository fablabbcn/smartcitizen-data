# Data connectors

This folder specifies data connectors with other APIs that might interact in some way with the Smart Citizen API. Flexible json format handled by the `configure` method provided by the ApiDevice Handler.

## Currently implemented connections

- [nilu](https://iflink.nilu.no/en/home/): https://sensors.nilu.no/api/doc

## Definition

```
{
  "handler": "NiluApiDevice", # Handler class in device_api.py
  "sensors": {
    "14": { # SC ID (see https://api.smartcitizen.me/v0/sensors/?per_page=200)
      "id": 56, # target ID (for NILU see https://sensors.nilu.no/api/components)
      "unitid": 28, # target unit ID
      "level": 1 # target level ID
    },
    ...
}
```