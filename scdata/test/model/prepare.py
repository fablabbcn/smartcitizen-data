def prepare_dataframe_model(self, model):

    test_names = list(model.data['train'].keys())

    # Create structure for multiple training
    if len(test_names) > 1: 
        multiple_training = True
        combined_name = model.name + '_' + config.name_multiple_training_data
        frames = list()
    else:
        multiple_training = False

    for test_name in test_names:

        device = model.data['train'][test_name]['devices']
        reference = model.data['train'][test_name]['reference_device']

        std_out('Preparing dataframe model for test {}'.format(test_name))
    
        if self.combine_devices(test_name):

            ## Send only the needed features
            list_features = list()
            list_features_multiple = list()
            features = model.data['features']
            try:
                # Create the list of features needed
                for item in features.keys():

                    # Dirty horrible workaround
                    if type(device) == list: device = device[0]

                    if item == 'REF': 
                        feature_name = features[item] + '_' + reference
                        feature_name_multiple = features[item]
                        reference_name = feature_name
                        reference_name_multiple = feature_name_multiple
                    else: 
                        feature_name = features[item] + '_' + device
                        feature_name_multiple = features[item]
                    list_features.append(feature_name)
                    list_features_multiple.append(feature_name_multiple)
                
                # Get features from data only and pre-process non-numeric data
                dataframeModel = self.tests[test_name].devices[config.combined_devices_name].readings.loc[:,list_features]
                
                # Remove device names if multiple training
                if multiple_training:
                    for i in range(len(list_features)):
                        dataframeModel.rename(columns={list_features[i]: list_features_multiple[i]}, inplace=True)

                dataframeModel = dataframeModel.apply(to_numeric, errors='coerce')   

                # Resample
                dataframeModel = dataframeModel.resample(model.data['data_options']['frequency'], limit = 1).mean()
                
                # Remove na
                if model.data['data_options']['clean_na']:
                    std_out('Cleaning na with {}'.format(model.data['data_options']['clean_na_method']))
                    if model.data['data_options']['clean_na_method'] == 'fill':
                        dataframeModel = dataframeModel.fillna(method='ffill')
                    
                    elif model.data['data_options']['clean_na_method'] == 'drop':
                        dataframeModel.dropna(axis = 0, how='any', inplace = True)

                if model.data['data_options']['min_date'] is not None:
                    dataframeModel = dataframeModel[dataframeModel.index > model.data['data_options']['min_date']]
                if model.data['data_options']['max_date'] is not None:
                    dataframeModel = dataframeModel[dataframeModel.index < model.data['data_options']['max_date']]

                if multiple_training:
                    frames.append(dataframeModel)
                
            except:
                std_out(f'Dataframe model failed for {test_name}')
                print_exc()
                return None
            else: 
                # Set flag
                self.tests[test_name].ready_to_model = True
                std_out(f'Dataframe model generated successfully for {test_name}')
                
    if multiple_training:
        std_out('Multiple training datasets requested. Combining')
        # Combine everything
        model.dataframe = concat(frames)
        model.features = features       
        model.reference = reference_name_multiple

        return combined_name
    else:
        model.dataframe = dataframeModel
        model.features = features
        model.reference = reference_name
        
        return test_name    