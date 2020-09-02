# from src.data.data import Data
# from src.models.model import Model
# from src.models.model_tools import metrics
# from src.visualization.visualization import Plot

import traceback, json, itertools
from termcolor import colored

class Batch(object):
    append_alphasense = 'PRE'
    append_derivative = 'DERIV'

    def __init__(self, tasks_file, verbose = False):
        try:
            self.verbose = verbose
            self.tasks = json.load(open(tasks_file, 'r'))
            
            self.re_processing_needed = False
            self.results = dict()
        except:
            traceback.print_exc()
            raise SystemError('Could not initialise object')
        
        else:
            self.std_out('Object initialised successfully', 'SUCCESS')
    
    def std_out(self, msg, type_message = None, force = False):
        if self.verbose or force: 
            if type_message is None: print(msg) 
            elif type_message == 'SUCCESS': print(colored(msg, 'green'))
            elif type_message == 'WARNING': print(colored(msg, 'yellow')) 
            elif type_message == 'ERROR': print(colored(msg, 'red'))
    
    def load_data(self, task, tests, options):

        try:
            for test in tests:

                # Check if we should avoid loading processed data
                if 'avoid_processed' in options.keys():
                    if options['avoid_processed'] != True: load_processed = True
                    else: load_processed = False
                else: load_processed = True
                
                # Load each of the tests
                self.results[task]['data'].load_test(test, options)
                # Options:
                # 'load_cached_API'
                # 'store_cached_API'
                # 'clean_na'
                # 'clean_na_method'
                # 'frequency'
        
        except:
            self.std_out('Problem loading data', 'ERROR')
            traceback.print_exc()
            return False
        else:
            self.std_out('Data load OK', 'SUCCESS')
            return True

    def pre_process_data(self, task, tests, data, task_has_model):
        
            for test in tests:
                self.std_out('Pre-processing {}'.format(test))
                try:
                    # Assume we need to re-process:
                    self.re_processing_needed = True
                            
                    # Find out if in the processed test, we have something that matches our variables
                    if data['data_options']['use_cache']:
                        try:
                            with open(join(self.results[task]['data'].available_tests()[test], 'cached', 'cached_info.json')) as handle:
                                cached_info = json.loads(handle.read())
                        except:
                            cached_info = None
                            pass

                        self.std_out('Checking if we can use previously processed test')

                    if 'avoid_processed' in data['data_options'].keys():
                        if not data['data_options']['avoid_processed']:
                            try:
                                # Open info file
                                with open(join(self.results[task]['data'].available_tests()[test], 'processed', 'processed_info.json')) as handle:
                                    processed_info = json.loads(handle.read())

                                # Check if there is pre-processing done in info
                                if 'pre-processing' in processed_info.keys():
                                    
                                    # Check if it has alphasense within it
                                    if 'alphasense' in processed_info['pre-processing'].keys():
                                        # Check if the pre-processed parameters are the right ones. Also, that there is no new data to process
                                        if processed_info['pre-processing']['alphasense'] == self.tasks[task]['pre-processing']['alphasense'] and cached_info['new_data_to_process'] == False:
                                            
                                            self.std_out('Using cached test, no need to preprocess')
                                            self.re_processing_needed = False

                                    # Not used for now
                                    # TO-DO put other pre-processing options here - filtering, smoothing?
                                    if 'custom' in processed_info['pre-processing'].keys():
                                        for pre_process in processed_info['pre_processing']['custom'].keys():
                                            if processed_info['pre-processing']['custom'][pre_process] == self.tasks[task]['pre-processing']['custom'][pre_process] and cached_info['new_data_to_process'] == False:
                                                self.std_out('Using cached test, no need to preprocess')
                                                self.re_processing_needed = False
                            
                            except:
                                processed_info = dict()
                                processed_info['pre-processing'] = dict()
                                # traceback.print_exc()
                                if not exists(join(self.results[task]['data'].available_tests()[test], 'processed', 'processed_info.json')):
                                    self.std_out('Processed Info json file does not exist') 
                                pass
                        else:
                            processed_info = dict()
                            processed_info['pre-processing'] = dict()
                       
                    # If we need to re-process data, do it nicely
                    if self.re_processing_needed:
                        self.std_out('Pre processing cannot be loaded from cached, calculating')
                        if cached_info is not None:
                            if cached_info['new_data_to_process']: self.std_out('Reason, new data to process')

                        if 'alphasense' in self.tasks[task]['pre-processing']:

                            # Variables
                            variables = self.tasks[task]['pre-processing']['alphasense']['variables']
                            overlapHours = self.tasks[task]['pre-processing']['alphasense']['overlapHours']
                            clean_negatives = self.tasks[task]['pre-processing']['alphasense']['clean_negatives']

                            for pollutant in variables.keys():
                                if variables[pollutant][0] == 'baseline': 
                                    baseline_method = variables[pollutant][2]
                                    if baseline_method == 'deltas':
                            
                                        # append_name = baseline_method[:].upper() + '_OVL_' + str(overlapHours) + '-' + str(self.tasks[task]['pre-processing']['alphasense']['parameters'][0]) + '-' +str(self.tasks[task]['pre-processing']['alphasense']['parameters'][1])
                                        variables[pollutant].append(self.tasks[task]['pre-processing']['alphasense']['parameters'])
                                        variables[pollutant].append(overlapHours)
                                    elif baseline_method == 'als':
                            
                                        append_name = baseline_method[:].upper() + '_OVL_' + str(overlapHours) + '-LAMBDA_' + str(lam_als[0]) + '-' +str(lam_als[-1]) + '_P_'+ str(p_als)
                                        variables[pollutant].append(self.tasks[task]['pre-processing']['alphasense']['parameters'])
                                        variables[pollutant].append(overlapHours)
                                # else:
                                #     append_name = 'CLASSIC'
                                
                                # Override for batch case
                                append_name = "PRE"    
                                variables[pollutant].append(append_name)
                                variables[pollutant].append(clean_negatives)

                            # Display options
                            options_alphasense = dict()
                            options_alphasense['checkBoxDecomp'] = False
                            options_alphasense['checkBoxPlotsIn'] = False
                            options_alphasense['checkBoxPlotsResult'] = False
                            options_alphasense['checkBoxVerb'] = False
                            options_alphasense['checkBoxStats'] = False
                            
                            # Calculate Alphasense
                            self.results[task]['data'].calculateAlphaSense(test, variables, options_alphasense)
                            
                            # Store what we used as parameters for the pre-processing
                            processed_info['pre-processing']['alphasense'] = self.tasks[task]['pre-processing']['alphasense']

                        if 'custom' in self.tasks[task]['pre-processing']:

                            # For later info
                            processed_info['pre-processing']['custom'] = dict()

                            for pre_process in self.tasks[task]['pre_processing']['custom']:
                                self.std_out('Performing custom pre-process')
                                # TO-DO Put here custom
                                # Store what we used as parameters for the pre-processing
                                processed_info['pre-processing']['custom'][pre_process] = self.tasks[task]['pre-processing']['custom'][pre_process]
                        
                        if 'derivate' in self.tasks[task]['pre-processing']:
                            self.std_out('Calculating derivative')
                            for channel in self.tasks[task]['pre-processing']['derivate']['channels'].keys():
                                window = self.tasks[task]['pre-processing']['derivate']['channels'][channel]
                                for device in self.results[task]['data'].tests[test]['devices']:
                                    if channel in self.results[task]['data'].tests[test]['devices'][device]['data'].columns:
                                        index = self.results[task]['data'].tests[test]['devices'][device]['data'].index
                                        self.results[task]['data'].tests[test]['devices'][device]['data'][channel + '_DERIV'] = self.results[task]['data'].tests[test]['devices'][device]['data'].loc[:,channel].diff().rolling(window = window).mean()/ index.to_series().diff().dt.total_seconds()                                

                        # Store the data in the info file and export csvs for next time
                        if data['data_options']['export_data'] is not None:
                            if cached_info is not None:
                                # Set the flag of pre-processed test to False
                                cached_info['new_data_to_process'] = False
                            
                            # Store what we used as parameters for the pre-processing
                            self.std_out('Saving info file for processed test')

                            if not exists(join(self.results[task]['data'].available_tests()[test], 'processed')):
                                self.std_out('Making dir for processed files')
                                mkdir(join(self.results[task]['data'].available_tests()[test], 'processed'))

                            # Dump processed info file
                            with open(join(self.results[task]['data'].available_tests()[test], 'processed', 'processed_info.json'), 'w') as file:
                                json.dump(processed_info, file)

                            # Dump cached info file
                            if cached_info is not None:
                                with open(join(self.results[task]['data'].available_tests()[test], 'cached', 'cached_info.json'), 'w') as file:
                                    json.dump(cached_info, file)

                            self.std_out('Saving cached and processed info done')

                            # Export is done later on
       
                    # Get list of devices
                    if task_has_model:
                        
                        # Do it for both, train and test, if they exist
                        for dataset in ['train', 'test']:
                            # Check for datasets that are not reference
                            if dataset in self.tasks[task]['model']['data'].keys():
                                if test in self.tasks[task]['model']['data'][dataset].keys():
                                    for device in self.tasks[task]['model']['data'][dataset][test]:
                                        if device + '_PROCESSED' in self.results[task]['data'].tests[test]['devices'].keys():
                                            # print ('PROCESSED columns')
                                            # print (self.results[task]['data'].tests[test]['devices'][device + '_PROCESSED']['data'].columns)
                                            # print ('TARGET columns')
                                            # print (self.results[task]['data'].tests[test]['devices'][device]['data'].columns)
                                            self.std_out("Combining processed data in test {}. Merging {} with {}".format(test, device, device + '_PROCESSED'))                           
                                            self.results[task]['data'].tests[test]['devices'][device]['data'] = self.results[task]['data'].tests[test]['devices'][device]['data'].combine_first(self.results[task]['data'].tests[test]['devices'][device + '_PROCESSED']['data'])
                                            self.results[task]['data'].tests[test]['devices'].pop(device + '_PROCESSED')
                                            # print ('Final columns')
                                            # print (self.results[task]['data'].tests[test]['devices'][device]['data'].columns)

                                        else:
                                            self.std_out("No available PROCESSED data to combine with")
                    else:

                        for device in data["datasets"][test]:
                            if device + '_PROCESSED' in self.results[task]['data'].tests[test]['devices'].keys():
                                self.std_out("Combining processed data in test {}. Merging {} with {}".format(test, device, device + '_PROCESSED'))
                                # print ('PROCESSED columns')
                                # print (self.results[task]['data'].tests[test]['devices'][device + '_PROCESSED']['data'].columns)
                                # print ('TARGET columns')
                                # print (self.results[task]['data'].tests[test]['devices'][device]['data'].columns)
                                cols_to_use = self.results[task]['data'].tests[test]['devices'][device + '_PROCESSED']['data'].columns.difference(self.results[task]['data'].tests[test]['devices'][device]['data'].columns)
                                # self.std_out('cols_to_use')
                                # self.std_out(cols_to_use)
                                # self.results[task]['data'].tests[test]['devices'][device]['data'].join(self.results[task]['data'].tests[test]['devices'][device  + '_PROCESSED']['data'][cols_to_use], left_index=True, right_index=True, how='outer')
                                self.results[task]['data'].tests[test]['devices'][device]['data'] = self.results[task]['data'].tests[test]['devices'][device]['data'].combine_first(self.results[task]['data'].tests[test]['devices'][device + '_PROCESSED']['data'])
                                self.results[task]['data'].tests[test]['devices'].pop(device + '_PROCESSED')
                                # print ('FINAL columns')
                                # print (self.results[task]['data'].tests[test]['devices'][device]['data'].columns)
                            else:
                                self.std_out("No available PROCESSED data to combine with")

                except:
                    self.std_out("Problem pre-processing test {}".format(test))
                    traceback.print_exc()
                    return False
                else:
                    self.std_out("Pre-processing test {} OK ".format(test))
            
            return True

    def sanity_checks(self, task, tests, task_has_model):

        # Sanity check for test presence
        if not all([self.results[task]['data'].available_tests().__contains__(i) for i in tests]):
            self.std_out ('Not all tests are available, review data input', 'ERROR')
            return False
        
        else:
            # Cosmetic output
            self.std_out('Test presence check passed', 'SUCCESS')
            
            # Check here if all the tuple_features are in each of the tests (accounting for the pre_processing)
            if task_has_model:
                
                # Get features
                features = self.tasks[task]['model']['data']['features']
                features_names = [features[key] for key in features.keys() if key != 'REF']
                reference_name = features['REF']

                datasets = ['train', 'test']

                pollutant_index = {'CO': 1, 'NO2': 2, 'O3': 3}

                # Do it for both, train and test, if they exist
                for dataset in datasets:
                    self.std_out('Checking validity of {} input'.format(dataset))
                    # Check for datasets that are not reference
                    if dataset in self.tasks[task]['model']['data'].keys():
                        for test in self.tasks[task]['model']['data'][dataset].keys():
                            for device in self.tasks[task]['model']['data'][dataset][test]['devices']:
                                # Get columns of the test
                                all_columns = list(self.results[task]['data'].tests[test].devices[device].readings.columns)

                                # Add the pre-processing ones and if we can pre-process
                                if 'pre-processing' in self.tasks[task].keys(): 
                                    if 'alphasense'in self.tasks[task]['pre-processing'].keys():
                                        # Check for each pollutant
                                        for pollutant in self.tasks[task]['pre-processing']['alphasense']['variables'].keys():
                                            self.std_out('Checking pre-processing for {}'.format(pollutant))
                                            # Define who should be the columns
                                            minimum_alpha_columns = ['GB_{}W'.format(pollutant_index[pollutant]), 'GB_{}A'.format(pollutant_index[pollutant])]
                                            # Check if we can pre-process alphasense data with working and auxiliary electrode

                                            if not all([all_columns.__contains__(i) for i in minimum_alpha_columns]): return False
                                            # We know we can pre-process, add the result of the pre-process to the columns
                                            all_columns.append(pollutant + '_' + self.append_alphasense)

                                    if 'derivate' in self.tasks[task]['pre-processing'].keys():
                                        for channel in self.tasks[task]['pre-processing']['derivate']['channels'].keys():
                                            all_columns.append(channel + '_' + self.append_derivative)

                                    if not all([all_columns.__contains__(i) for i in features_names]): return False
                                    

                            # In case of training dataset, check that the reference exists
                            if dataset == 'train':
                                found_ref = False
                                for device in self.results[task]['data'].tests[test].devices.keys():
                                    if self.results[task]['data'].tests[test].devices[device].type == 'OTHER':
                                        reference_dataframe = device
                                        if not (self.tasks[task]['model']['data']['features']['REF'] in self.results[task]['data'].tests[test].devices[device].readings.columns) and not found_ref: 
                                            found_ref = False
                                            self.std_out('Reference presence check not passed')
                                        else:
                                            found_ref = True
                                            self.std_out('Reference presence check passed')
                                if not found_ref: return False
            
            self.std_out('All checks passed')
            return True

    def run(self):
        
        # Process task by task
        for task in self.tasks.keys():
            self.std_out('-------------------------------')
            self.std_out('Evaluating task {}'.format(task))
            self.results[task] = dict()
            self.results[task]['data'] = data_wrapper(verbose = self.verbose)

            if 'model' in self.tasks[task].keys():
                model_name = self.tasks[task]['model']['model_name'] 
                
                # Create model_wrapper instance
                self.std_out('Task {} involves modeling, initialising model'.format(task))
                current_model = model_wrapper(self.tasks[task]['model'], verbose = self.verbose)
                
                # Parse current model instance
                task_has_model = True

                # Ignore extra data in json if present in the task
                if 'data' in self.tasks[task].keys(): self.std_out('Ignoring additional data in task', 'WARNING')

                # Model dataset names and options
                tests = list(set(itertools.chain(self.tasks[task]['model']['data']['train'], self.tasks[task]['model']['data']['test'])))
                data_dict = self.tasks[task]['model']['data']

            else:
                self.std_out('No model involved in task {}'.format(task))
     
                # Model dataset names and options
                tests = list(self.tasks[task]['data']['datasets'])
                data_dict = self.tasks[task]['data']
                task_has_model = False

            # Cosmetic output
            self.std_out('Loading data...')

            # Load data
            if not self.load_data(task, tests, data_dict['data_options']):
                self.std_out('Failed loading data', 'SUCCESS')
                return

            # Cosmetic output
            self.std_out('Sanity checks...')
            # Perform sanity check
            if not self.sanity_checks(task, tests, task_has_model):
                self.std_out('Failed sanity checks')
                return
            
            # TO-DO: Fix                               
            # # Pre-process data
            # if 'pre-processing' in self.tasks[task].keys():
            #     # Cosmetic output
            #     self.std_out('Pre-processing requested...')

            #     if not self.pre_process_data(task, tests, data_dict, task_has_model): 
            #         self.std_out('Failed preprocessing')
            #         return
            
            # Perform model stuff
            if task_has_model:
                try:
                    # Prepare dataframe for training
                    train_dataset = self.results[task]['data'].prepare_dataframe_model(current_model)
                    self.std_out (f'Train dataset: {train_dataset}')
                    
                    if train_dataset is None:
                        self.std_out('Failed training dataset dataframe preparation for model', 'ERROR')
                        return
                    
                    # Train Model based on training dataset
                    current_model.train()
                    
                    # Evaluate Model in test data
                    for test_dataset in current_model.data['test'].keys():
                        # Check if there is a reference
                        if 'reference_device' in current_model.data['test'][test_dataset].keys():
                            reference = self.results[task]['data'].tests[test_dataset].devices[current_model.data['test'][test_dataset]['reference_device']].readings[current_model.data['features']['REF']]
                        else:
                            reference = None
                        
                        for device in current_model.data['test'][test_dataset]['devices']:
                            prediction_name = f'{device}_{model_name}'
                            self.std_out('-----------------------------------------')
                            self.std_out('Predicting {} for device {} in {}'.format(prediction_name, device, test_dataset))

                            # Get prediction for test          
                            prediction = current_model.predict(self.results[task]['data'].tests[test_dataset].devices[device].readings, prediction_name, reference)

                            # Combine it in tests
                            self.results[task]['data'].tests[test_dataset].devices[device].readings.combine_first(prediction)

                    # Export model data if requested
                    if data_dict['data_options']["export_data"] is not None:
                        dataFrameExport = current_model.dataFrameTrain.copy()
                        dataFrameExport = dataFrameExport.combine_first(current_model.dataFrameTest)
                    else:
                        dataFrameExport = None
                    
                    # Save model in session
                    if current_model.options['session_active_model']:
                        self.std_out (f'Saving model in session:{train_dataset}')
                        self.results[task]['data'].archive_model(current_model)

                    # Export model if requested
                    if current_model.options['export_model']:
                        current_model.export(self.results[task]['data'].models)
                except:
                    traceback.print_exc()
                    pass
            
            # Export data
            if data_dict['data_options']["export_data"] is not None:
                try: 
                    if data_dict['data_options']["export_data"] == 'Only Generic':
                        all_channels = False
                        include_raw = True
                        include_processed = True
                    if data_dict['data_options']["export_data"] == 'Only Processed':
                        all_channels = False
                        include_raw = False
                        include_processed = True
                    elif data_dict['data_options']["export_data"] == 'Only Raw':
                        all_channels = False
                        include_raw = True
                        include_processed = False
                    elif data_dict['data_options']["export_data"] == 'All':
                        all_channels = True
                        include_raw = True
                        include_processed = False
                    else:
                        all_channels = False
                        include_raw = False
                        include_processed = False
                        
                    if self.re_processing_needed:
                        # Including pre-processed for next time
                        include_processed = True
                        self.std_out('Including preprocessed for next time')

                    # Export data to disk once tasks completed
                    if task_has_model:
                        # Do it for both, train and test, if they exist
                        for dataset in ['train', 'test']:
                            # Check for datasets that are not reference
                            if dataset in self.tasks[task]['model']['data'].keys():
                                for test in self.tasks[task]['model']['data'][dataset].keys():
                                    for device in self.tasks[task]['model']['data'][dataset][test]['devices']:
                                        self.std_out('Exporting data of device {} in test {} to processed folder'.format(device, test))
                                        self.results[task]['data'].export_data(test, device, all_channels = all_channels, 
                                                            include_raw = include_raw, include_processed = include_processed, 
                                                            rename = data_dict['data_options']['rename_export_data'], 
                                                            to_processed_folder = True, 
                                                            forced_overwrite = True)
                    else:
                        for test in tests:
                            # Get devices
                            list_devices = self.tasks[task]['data']['datasets'][test]

                            # Iterate and export them
                            for device in list_devices:
                                self.std_out('Exporting data of device {} in test {} to processed folder'.format(device, test))
                                self.results[task]['data'].export_data(test, device, all_channels = all_channels, 
                                    include_raw = include_raw, include_processed = include_processed, 
                                    rename = data_dict['data_options']['rename_export_data'],
                                    to_processed_folder = True, 
                                    forced_overwrite = True)
                except:
                    traceback.print_exc()
                    pass
            
            # Plot data
            if "plot" in self.tasks[task].keys():
                self.std_out('Processing plotting task')
                try:
                    for plot_description in self.tasks[task]["plot"].keys():

                        # Separate device plots
                        if self.tasks[task]["plot"][plot_description]["options"]["separate_device_plots"] == True:
                            original_filename = self.tasks[task]['plot'][plot_description]['options']['file_name']
                            # Each trace it's own
                            for trace in self.tasks[task]["plot"][plot_description]['data']['traces']:
                                if self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["device"] == 'all':
                                    list_devices_plot = list()
                                    for device in self.results[task]['data'].tests[self.tasks[task]["plot"][plot_description]['data']['test']].devices.keys():
                                        channel = self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["channel"]
                                        if channel in self.results[task]['data'].tests[self.tasks[task]["plot"][plot_description]['data']['test']].devices[device].readings.columns:
                                            list_devices_plot.append(device)
                                        else:
                                            self.std_out('Trace ({}) not in tests in device {}'.format(channel, device))
                                else:
                                    list_devices_plot = self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["device"]      
                            # Make a plot for each device
                            for device in list_devices_plot:
                                # Rename the traces
                                for trace in self.tasks[task]["plot"][plot_description]['data']['traces']:
                                    self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["device"] = device

                                # plot it
                                plot_object = plot_wrapper(self.tasks[task]["plot"][plot_description], True)
                                plot_object.plot(self.results[task]['data'])
                                
                                # Export if we have how to
                                if self.tasks[task]["plot"][plot_description]['options']['export_path'] is not None and self.tasks[task]["plot"][plot_description]['options']['file_name'] is not None:
                                    self.tasks[task]['plot'][plot_description]['options']['file_name'] = original_filename + '_' + device
                                    plot_object.export_plot()
                                plot_object.clean_plot()
                        # Or only one
                        else:    
                            plot_object = plot_wrapper(self.tasks[task]["plot"][plot_description], True)
                            plot_object.plot(self.results[task]['data'])
                            if self.tasks[task]["plot"][plot_description]['options']['export_path'] is not None and self.tasks[task]["plot"][plot_description]['options']['file_name'] is not None:
                                self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["device"] = device
                                plot_object.export_plot()
                            plot_object.clean_plot()
                except:
                    traceback.print_exc()
                    pass
        
        self.std_out('Finished task {}'.format(task))
        self.std_out('---')