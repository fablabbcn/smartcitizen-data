from scdata.utils import std_out, localise_date
from pandas import DataFrame
from scdata._config import config

def dispersion_analysis(self, min_date = None, max_date = None, location = 'Europe/Madrid', smooth_window = 5):
    '''
        Creates channels on a new dataframe for each device/channel combination, and makes the average/std of each
        in a point-by-point fashion

        Parameters:
        -----------        
        min_date: String
            Default: None
            Minimum date from which to perform the analysis
        
        max_date: String
            Default: None
            Maximum date from which to perform the analysis
        
        location: String
            Default: None
            Location of the sensors for timezone
        
        smooth_window: int
            Default: 5
            If not None, performs smoothing of the channels with rolling average. 

        Returns:
        ---------

    '''
    self.dispersion_df = DataFrame()

    # Get channels if not done yet
    if len(self.common_channels) == 0: self.get_common_channels() 

    # Localise dates
    min_date = localise_date(min_date, location)
    max_date = localise_date(max_date, location)
    
    # Calculate the dispersion for the sensors present in the dataset
    warning = False

    for channel in self.common_channels:
        columns = list()

        if channel in config._dispersion['ignore_channels']: continue

        for device in self.devices:
            if channel in self.devices[device].readings.columns and len(self.devices[device].readings.loc[:,channel]) >0:
                # Important to resample and bfill for unmatching measures
                if smooth_window is not None:
                    # channel_new = self.devices[device].readings[channel].resample('1Min').bfill().rolling(window=smooth_window).mean()
                    channel_new = self.devices[device].readings[channel].bfill().rolling(window=smooth_window).mean()
                    self.dispersion_df[channel + '-' + device] = channel_new[channel_new > 0]
                else:
                    self.dispersion_df[channel + '-' + device] = self.devices[device].readings[channel].resample('1Min').bfill()

                columns.append(channel + '-' + device)
            else:
                std_out(f'Device {device} does not contain {channel}</p>', 'WARNING')
                warning = True

        self.dispersion_df.index = localise_date(self.dispersion_df.index, location)

        # Trim dataset to min and max dates (normally these tests are carried out with _minutes_ of differences)
        if min_date is not None: self.dispersion_df = self.dispersion_df[self.dispersion_df.index > min_date]
        if max_date is not None: self.dispersion_df = self.dispersion_df[self.dispersion_df.index < max_date]

        # Calculate Metrics
        self.dispersion_df[channel + '_AVG'] = self.dispersion_df.loc[:,columns].mean(skipna=True, axis = 1)
        self.dispersion_df[channel + '_STD'] = self.dispersion_df.loc[:,columns].std(skipna=True, axis = 1)
    
    if not warning:
        std_out(f'All devices have the provided channels list recorded')
    else:
        std_out(f'Missing channels, review data', 'WARNING')

    return self.dispersion_summary

@property
def dispersion_summary(self):
    self._dispersion_summary = dict()

    if self.dispersion_df is None:
        std_out('Perform dispersion analysis first!', 'ERROR')
        return None
    for channel in self.common_channels:
        if channel in config._dispersion['ignore_channels']: continue
        # Calculate 
        self._dispersion_summary[channel] = self.dispersion_df[channel + '_STD'].mean()

    return self._dispersion_summary
