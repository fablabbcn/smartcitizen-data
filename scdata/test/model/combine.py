def combine_devices(self):

        try: 
            std_out(f'Combining devices for {self.name}')

            dfc = DataFrame()
            
            ## And then add it again
            for device in self.devices.keys():
                
                new_names = list()
                
                for name in self.devices[device].readings.columns:
                    new_names.append(name + '_' + self.devices[device].name)
                
                df = self.devices[device].readings.copy()
                df.columns = new_names
                dfc = dfc.combine_first(df)
            
            self.devices[config.combined_devices_name] = Device({'id': config.combined_devices_name,
                                                                 'frequency': '1Min'}
                                                                )

            self.devices[config.combined_devices_name].readings = dfc
        
        except:
            std_out('Error ocurred while combining data. Review data', 'ERROR')
            print_exc()
            return False
        else:
            std_out('Data combined successfully', 'SUCCESS')
            return True