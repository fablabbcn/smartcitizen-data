from scdata.tools.custom_logger import logger
from scdata.tools.date import localise_date
from pandas import DataFrame
from scdata._config import config

def dispersion_analysis(self, devices = None, min_date = None, max_date = None, timezone = 'Europe/Madrid', smooth_window = 5):
    '''
        Creates channels on a new dataframe for each device/channel combination, and makes the average/std of each
        in a point-by-point fashion

        Parameters:
        -----------
        devices: list
            Default: None
            If list of devices is None, then it will use all devices in self.devices
        min_date: String
            Default: None
            Minimum date from which to perform the analysis

        max_date: String
            Default: None
            Maximum date from which to perform the analysis

        timezone: String
            Default: None
            Sensors for timezone

        smooth_window: int
            Default: 5
            If not None, performs smoothing of the channels with rolling average.

        Returns:
        ---------

    '''
    dispersion_df = DataFrame()

    # Get common channels for this group
    if devices is not None:
        common_ch = self.get_common_channels(devices = devices)
        _devices = devices
    else:
        common_ch = self.get_common_channels()
        _devices = self.devices

    # Localise dates
    min_date = localise_date(min_date, timezone)
    max_date = localise_date(max_date, timezone)

    # Calculate the dispersion for the sensors present in the dataset
    warning = False

    for channel in common_ch:
        columns = list()

        if channel in config._dispersion['ignore_channels']: continue

        for device in _devices:
            if channel in self.devices[device].readings.columns and len(self.devices[device].readings.loc[:,channel]) >0:
                # Important to resample and bfill for unmatching measures
                if smooth_window is not None:
                    # channel_new = self.devices[device].readings[channel].resample('1Min').bfill().rolling(window=smooth_window).mean()
                    channel_new = self.devices[device].readings[channel].bfill().rolling(window=smooth_window).mean()
                    dispersion_df[channel + '-' + device] = channel_new[channel_new > 0]
                else:
                    dispersion_df[channel + '-' + device] = self.devices[device].readings[channel].resample('1Min').bfill()

                columns.append(channel + '-' + device)
            else:
                logger.warning(f'Device {device} does not contain {channel}</p>')
                warning = True

        dispersion_df.index = localise_date(dispersion_df.index, timezone)

        # Trim dataset to min and max dates (normally these tests are carried out with _minutes_ of differences)
        if min_date is not None: dispersion_df = dispersion_df[dispersion_df.index > min_date]
        if max_date is not None: dispersion_df = dispersion_df[dispersion_df.index < max_date]

        # Calculate Metrics
        dispersion_df[channel + '_AVG'] = dispersion_df.loc[:,columns].mean(skipna=True, axis = 1)
        dispersion_df[channel + '_STD'] = dispersion_df.loc[:,columns].std(skipna=True, axis = 1)

    if not warning:
        logger.info(f'All devices have the provided channels list recorded')
    else:
        logger.warning(f'Missing channels, review data')

    if devices is None:
        self.dispersion_df = dispersion_df
        return self.dispersion_summary

    group_dispersion_summary = dict()

    for channel in common_ch:
        if channel in config._dispersion['ignore_channels']: continue
        # Calculate
        group_dispersion_summary[channel] = dispersion_df[channel + '_STD'].mean()

    return group_dispersion_summary

@property
def dispersion_summary(self):
    self._dispersion_summary = dict()

    if self.dispersion_df is None:
        logger.error('Perform dispersion analysis first!')
        return None
    for channel in self.common_channels:
        if channel in config._dispersion['ignore_channels']: continue
        # Calculate
        self._dispersion_summary[channel] = self.dispersion_df[channel + '_STD'].mean()

    return self._dispersion_summary
