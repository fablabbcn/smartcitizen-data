from pandas import DataFrame
from scdata.tools.custom_logger import logger
from scdata.device import Device

def combine(self, devices = None, channels = None, resample = True, frequency = '1Min'):
    """
    Combines devices from a test into a new dataframe, following the
    naming as follows: DEVICE-NAME_READING-NAME
    Parameters
    ----------
        devices: list or None
            None
            If None, includes all the devices in self.devices
        channels: list or None
            None
            If None, includes all the readings in self.readings
    Returns
    -------
        Dataframe if successful or False otherwise
    """

    dfc = DataFrame()
    if not self.loaded:
        logger.error('Cannot combine data if test is not loaded. Maybe test.load() first?')

    if devices is None:
        dl = [device.id for device in self.devices]
    else:
        # Only requested AND available
        dl = list(set(devices).intersection([device.id for device in self.devices]))
        if len(dl) != len(devices):
            logger.warning('Requested devices are not all present in devices')
            logger.info(f'Discarding {set(devices).difference([device.id for device in self.devices])}')

    for device in dl:
        new_names = list()

        if channels is None:
            channel_list = list(self.get_device(device).data.columns)
        else:
            # Only pick the ones that are actually present
            channel_list = list(set(channels).intersection(list(self.get_device(device).data.columns)))

            if any([channel not in channel_list for channel in channels]):
                logger.warning(f'Requested channels are not all present in readings for device {device}')
                logger.warning(f'Discarding {list(set(channels).difference(list(self.get_device(device).data.columns)))}')

        rename = dict()

        for channel in channel_list:
            rename[channel] = f'{channel}_{self.get_device(device).id}'

        df = self.get_device(device).data[channel_list].copy()
        df.rename(columns = rename, inplace = True)
        if resample:
            df = df.resample(frequency).mean()
        dfc = dfc.combine_first(df)

    if dfc.empty:
        logger.error('Error ocurred while combining data. Review data')
        return False
    else:
        logger.info('Data combined successfully')
        return dfc