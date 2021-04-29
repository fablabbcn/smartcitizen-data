from scdata.utils import std_out

def get_common_channels(self, ignore_missing_channels = False, pop_zero_readings_devices = False, verbose = True):
    '''
        Convenience method get the common channels of the devices in the test
        Params:
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
    list_devices = list(self.devices.keys())
    # Init list of common channels. Get just the first one
    list_channels = self.devices[list_devices[0]].readings.columns

    # Extract list of common channels
    len_channels = len(list_channels)
    show_warning = False
    channels_devices = {}

    for device in list_devices:

        try:

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
            std_out (f'Number of points {len(self.devices[device].readings.index)}', force = verbose)

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
        except:
            std_out ('Error on device, ignoring', 'WARNING')
            continue

    self.common_channels = list_channels
    
    std_out(f'Final list of channels:\n {self.common_channels}')
    if show_warning:
    	std_out (f'Some devices show less amount of sensors', 'WARNING')
    	print (channels_devices)

    return self.common_channels