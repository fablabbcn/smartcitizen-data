from scdata.utils import std_out
import matplotlib.pyplot as plt
import missingno as msno
from pandas import to_datetime, DataFrame
from scdata.test.plot.plot_tools import prepare_data
from scdata._config import config
from scdata.utils.dictmerge import dict_fmerge

def gaps_check(self, devices = None, channels = None, groupby = 'channel', **kwargs):
    if config.framework == 'jupyterlab': plt.ioff();
    plt.clf();

    if 'formatting' not in kwargs:
        std_out('Using default formatting')
        formatting = config._missingno_def_fmt
    else:
        formatting = dict_fmerge(config._missingno_def_fmt, kwargs['formatting'])

	# Get list of devices
    if devices is None:
        _devices = list(self.devices.keys())
    else:
        _devices = devices

    if channels is None:
        std_out('Need some channels to check gaps for', 'ERROR')
        return

    if groupby == 'device':

        for device in _devices:
            if device not in self.devices:
                std_out('Device not found in test', 'WARNING')
                continue
            msno.matrix(self.devices[device].readings)

    elif groupby == 'channel':

        for channel in channels:
            traces = {"1": {"devices": _devices, "channel": channel, "subplot": 1}}
            options = config._plot_def_opt
            df, _ = prepare_data(self, traces, options)
            fig, ax = plt.subplots(1, len(_devices), figsize=(formatting['width'], formatting['height']))
            for device in _devices:
                if device not in self.devices:
                    std_out('Device not found in test', 'WARNING')
                    continue
                msno.matrix(df = DataFrame(df[f'{channel}_{device}']), sparkline=False, ax = ax[_devices.index(device)], fontsize=formatting['fontsize'])
                ax[_devices.index(device)].set_yticks([i for i in range(len(df))], [i for i in df.index.values])
            plt.show()

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
        std_out ('Device {}'.format(device), force = verbose)
        std_out (f'Min reading at {self.devices[device].readings.index[0]}', force = verbose)
        std_out (f'Max reading at {self.devices[device].readings.index[-1]}', force = verbose)
        std_out (f'Number of dataframe points {len(self.devices[device].readings.index)}', force = verbose)
        if detailed:
            for column in list_channels:
                std_out ('\tColumn {}'.format(column), force = verbose)
                nas = self.devices[device].readings[column].isna()
                std_out ('\tNumber of nas {}'.format(nas.sum()), force = verbose)

        ## Eliminate devices with no points
        if (len(self.devices[device].readings.index) == 0):
            std_out (f'Device {device} for insufficient data points', 'WARNING')
            if pop_zero_readings_devices: self.devices.pop(device)
        # Check the number of channels
        elif len_channels != len(self.devices[device].readings.columns):
            std_out(f"Device {device} has {len(self.devices[device].readings.columns)}. Current common channel length is {len_channels}", 'WARNING')
            len_channels = len(list_channels)
            show_warning = True
            if ignore_missing_channels: std_out ("Ignoring missing channels", 'WARNING')
        std_out ('---------', force = verbose)


    if return_all:

        self.common_channels = list_channels
    
        std_out(f'Final list of channels:\n {self.common_channels}')
        if show_warning:
            std_out (f'Some devices show less amount of sensors', 'WARNING')
            print (channels_devices)

        return self.common_channels

    else:

        return list_channels
