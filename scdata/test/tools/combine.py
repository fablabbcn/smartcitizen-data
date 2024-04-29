from pandas import DataFrame
from scdata.tools.custom_logger import logger
from scdata.device import Device

def combine(self, devices = None, readings = None):
    """
    Combines devices from a test into a new dataframe, following the
    naming as follows: DEVICE-NAME_READING-NAME
    Parameters
    ----------
        devices: list or None
            None
            If None, includes all the devices in self.devices
        readings: list or None
            None
            If None, includes all the readings in self.readings
    Returns
    -------
        Dataframe if successful or False otherwise
    """

    dfc = DataFrame()

    if devices is None:
        dl = list(self.devices.keys())
    else:
        # Only pick the ones that are actually present
        dl = list(set(devices).intersection(list(self.devices.keys())))
        if len(dl) != len(devices):
            logger.warning('Requested devices are not all present in devices')
            logger.info(f'Discarding {set(devices).difference(list(self.devices.keys()))}')

    for device in dl:
        new_names = list()

        if readings is None:
            rl = list(self.devices[device].readings.columns)
        else:
            # Only pick the ones that are actually present
            rl = list(set(readings).intersection(list(self.devices[device].readings.columns)))

            if any([reading not in rl for reading in readings]):
                logger.warning(f'Requested readings are not all present in readings for device {device}')
                logger.warning(f'Discarding {list(set(readings).difference(list(self.devices[device].readings.columns)))}')

        rename = dict()

        for reading in rl:
            rename[reading] = reading + '_' + self.devices[device].id

        df = self.devices[device].readings[rl].copy()
        df.rename(columns = rename, inplace = True)
        dfc = dfc.combine_first(df)

    if dfc.empty:
        logger.error('Error ocurred while combining data. Review data')
        return False
    else:
        logger.info('Data combined successfully')
        return dfc