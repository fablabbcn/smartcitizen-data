from scdata.tools.custom_logger import logger
import matplotlib.pyplot as plt
from pandas import to_datetime, DataFrame
from scdata.test.plot.plot_tools import prepare_data
from scdata._config import config
from scdata.tools.dictmerge import dict_fmerge

def get_common_channels(self, devices = None, ignore_missing_channels = False, pop_zero_readings_devices = False, detailed = False, verbose = True):
    '''
        Convenience method get the common channels of the devices in the test
        Params:
            devices: list
                None
                List of devices to get common channels from. Passing None means 'all'
            ignore_missing_channels: bool
            	False
            	In case there is a device with lower amount of channels, ignore the missing channels and keep going
            pop_zero_readings_devices: bool
            	False
            	Remove devices from test that have no readings
			verbose: bool
				True
				Print extra info
        Returns:
			List containing the common channels to all devices
    '''

	# Get list of devices
    if devices is None:
        list_devices = list(self.devices.keys())
        return_all = True
    else:
        list_devices = devices
        return_all = False

    # Init list of common channels. Get just the first one
    list_channels = self.devices[list_devices[0]].readings.columns

    # Extract list of common channels
    len_channels = len(list_channels)
    show_warning = False
    channels_devices = {}

    for device in list_devices:

        if ignore_missing_channels:
            # We don't reduce the list in case the new list is smaller
            list_channels = list(set(list_channels) | set(self.devices[device].readings.columns))
        else:
            # We reduce it
            list_channels = list(set(list_channels) & set(self.devices[device].readings.columns))

        channels_devices[device] = {len(self.devices[device].readings.columns)}
        logger.info (f'Device {device}')
        logger.info (f'Min reading at {self.devices[device].readings.index[0]}')
        logger.info (f'Max reading at {self.devices[device].readings.index[-1]}')
        logger.info (f'Number of dataframe points {len(self.devices[device].readings.index)}')
        if detailed:
            for column in list_channels:
                logger.info ('\tColumn {}'.format(column))
                nas = self.devices[device].readings[column].isna()
                logger.info ('\tNumber of nas {}'.format(nas.sum()))

        ## Eliminate devices with no points
        if (len(self.devices[device].readings.index) == 0):
            logger.warning (f'Device {device} for insufficient data points')
            if pop_zero_readings_devices: self.devices.pop(device)
        # Check the number of channels
        elif len_channels != len(self.devices[device].readings.columns):
            logger.warning(f"Device {device} has {len(self.devices[device].readings.columns)}. Current common channel length is {len_channels}")
            len_channels = len(list_channels)
            show_warning = True
            if ignore_missing_channels: logger.warning ("Ignoring missing channels")
        logger.error ('---------')

    if return_all:

        self.common_channels = list_channels

        logger.info(f'Final list of channels:\n {self.common_channels}')
        if show_warning:
            logger.warning (f'Some devices show less amount of sensors')
            print (channels_devices)

        return self.common_channels

    else:

        return list_channels
