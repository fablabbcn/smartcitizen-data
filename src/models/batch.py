import itertools
from os.path import basename, normpath, join, exists
from os import getcwd, walk, mkdir
import json
from src.data.recording import recording
from src.models.model_tools import model_wrapper
from src.visualization.visualization import plot_wrapper
import numpy as np
import traceback

from src.data.signal_tools import metrics

class batch_analysis:
    append_alphasense = 'PRE'

    def __init__(self, tasksFile, verbose = False):
        try:
            self.verbose = verbose
            self.records = recording(verbose = self.verbose)
            self.tasks = json.load(open(tasksFile, 'r'))
            self.available_tests = self.records.available_tests()
            self.re_processing_needed = False
        
        except:
            raise SystemError('Could not initialise object')
        
        else:
            self.std_out('Object initialised successfully')

    def std_out(self, msg):
        if self.verbose: print(msg)

    def load_data(self, tests, options):

        try:
            for test in tests:

                # Check if we should avoid loading processed data
                if 'avoid_processed' in options.keys():
                    if options['avoid_processed'] != True: load_processed = True
                    else: load_processed = False
                else: load_processed = True
                
                # Load each of the tests
                self.records.load_recording_database(test, self.available_tests[test], 
                                                    target_raster = options['target_raster'],
                                                    clean_na = options['clean_na'],
                                                    clean_na_method = options['clean_na_method'],
                                                    load_cached_API = options['use_cache'], 
                                                    cache_API = options['use_cache'],
                                                    load_processed = load_processed)
        
        except:
            self.std_out('Problem loading data')
            traceback.print_exc()
            return False
        else:
            self.std_out('Data load OK')
            return True

    def pre_process_data(self, task, tests, data, has_model):
        
            for test in tests:
                self.std_out('Pre-processing {}'.format(test))
                try:
                    # Assume we need to re-process:
                    self.re_processing_needed = True
                            
                    # Find out if in the processed test, we have something that matches our variables
                    if data['options']['use_cache']:
                        try:
                            with open(join(self.available_tests[test], 'cached', 'cached_info.json')) as handle:
                                cached_info = json.loads(handle.read())
                        except:
                            cached_info = None
                            pass

                        self.std_out('Checking if we can use previously processed test')

                        try:
                            # Open info file
                            with open(join(self.available_tests[test], 'processed', 'processed_info.json')) as handle:
                                processed_info = json.loads(handle.read())

                            # Check if there is pre-processing done in info
                            if 'pre-processing' in processed_info.keys():
                                # Check if it has alphasense within it
                                if 'alphasense' in processed_info['pre-processing'].keys():
                                    # Check if the pre-processed parameters are the right ones. Also, that there is no new data to process
                                    if processed_info['pre-processing']['alphasense'] == task['pre-processing']['alphasense'] and cached_info['new_data_to_process'] == False:
                                        
                                        self.std_out('Using cached test, no need to preprocess')
                                        self.re_processing_needed = False

                                if 'custom' in processed_info['pre-processing'].keys():
                                    for pre_process in processed_info['pre_processing']['custom'].keys():
                                        if processed_info['pre-processing']['custom'][pre_process] == task['pre-processing']['custom'][pre_process] and cached_info['new_data_to_process'] == False:
                                            self.std_out('Using cached test, no need to preprocess')
                                            self.re_processing_needed = False
                        
                        except:
                            processed_info = dict()
                            processed_info['pre-processing'] = dict()
                            # traceback.print_exc()
                            if not exists(join(self.available_tests[test], 'processed', 'processed_info.json')):
                                self.std_out('Processed Info json file does not exist') 
                            pass
                       
                    # If we need to re-process data, do it nicely
                    if self.re_processing_needed:
                        self.std_out('Pre processing cannot be loaded from cached, calculating')
                        if cached_info is not None:
                            if cached_info['new_data_to_process']: self.std_out('Reason, new data to process')

                        if 'alphasense' in task['pre-processing']:

                            # Variables for the two methods (deltas or ALS)
                            variables = list()
                            
                            baseline_method = task['pre-processing']['alphasense']['baseline_method']
                            parameters = task['pre-processing']['alphasense']['parameters']
                            
                            if baseline_method == 'deltas':
                                variables.append(np.arange(parameters[0], parameters[1], parameters[2]))
                            elif baseline_method == 'als':
                                variables.append([parameters['lambda'], parameters['p']])
                            
                            variables.append(task['pre-processing']['alphasense']['methods'])
                            variables.append(task['pre-processing']['alphasense']['overlapHours'])
                            variables.append(baseline_method)

                            # Display options
                            options_alphasense = dict()
                            options_alphasense['checkBoxDecomp'] = False
                            options_alphasense['checkBoxPlotsIn'] = False
                            options_alphasense['checkBoxPlotsResult'] = False
                            options_alphasense['checkBoxVerb'] = False
                            options_alphasense['checkBoxStats'] = False
                            
                            # Calculate Alphasense
                            self.records.calculateAlphaSense(test, self.append_alphasense, variables, options_alphasense)
                            
                            # Store what we used as parameters for the pre-processing
                            processed_info['pre-processing']['alphasense'] = task['pre-processing']['alphasense']

                        if 'custom' in task['pre-processing']:

                            # For later info
                            processed_info['pre-processing']['custom'] = dict()

                            for pre_process in task['pre_processing']['custom']:
                                self.std_out('Performing custom pre-process')
                                # TO-DO Put here custom
                                # Store what we used as parameters for the pre-processing
                                processed_info['pre-processing']['custom'][pre_process] = task['pre-processing']['custom'][pre_process]
                             

                        # Store the data in the info file and export csvs for next time
                        if data['options']['export_data'] is not None:
                            if cached_info is not None:
                                # Set the flag of pre-processed test to False
                                cached_info['new_data_to_process'] = False
                            
                            # Store what we used as parameters for the pre-processing
                            self.std_out('Saving info file for processed test')

                            if not exists(join(self.available_tests[test], 'processed')):
                                self.std_out('Making dir for processed files')
                                mkdir(join(self.available_tests[test], 'processed'))

                            # Dump processed info file
                            with open(join(self.available_tests[test], 'processed', 'processed_info.json'), 'w') as file:
                                json.dump(processed_info, file)

                            # Dump cached info file
                            with open(join(self.available_tests[test], 'cached', 'cached_info.json'), 'w') as file:
                                json.dump(cached_info, file)

                            self.std_out('Saving cached and processed info done')

                            # Export is done later on
       
                    # Get list of devices
                    if has_model:
                        
                        # Do it for both, train and test, if they exist
                        for dataset in ['train', 'test']:
                            # Check for datasets that are not reference
                            if dataset in task['model']['data'].keys():
                                if test in task['model']['data'][dataset].keys():
                                    for device in task['model']['data'][dataset][test]:
                                        if device + '_PROCESSED' in self.records.readings[test]['devices'].keys():
                                            # print ('PROCESSED columns')
                                            # print (self.records.readings[test]['devices'][device + '_PROCESSED']['data'].columns)
                                            # print ('TARGET columns')
                                            # print (self.records.readings[test]['devices'][device]['data'].columns)
                                            self.std_out("Combining processed data in test {}. Merging {} with {}".format(test, device, device + '_PROCESSED'))                           
                                            self.records.readings[test]['devices'][device]['data'] = self.records.readings[test]['devices'][device]['data'].combine_first(self.records.readings[test]['devices'][device + '_PROCESSED']['data'])
                                            self.records.readings[test]['devices'].pop(device + '_PROCESSED')
                                            # print ('Final columns')
                                            # print (self.records.readings[test]['devices'][device]['data'].columns)

                                        else:
                                            self.std_out("No available PROCESSED data to combine with")
                    else:

                        for device in data["datasets"][test]:
                            if device + '_PROCESSED' in self.records.readings[test]['devices'].keys():
                                self.std_out("Combining processed data in test {}. Merging {} with {}".format(test, device, device + '_PROCESSED'))
                                print ('PROCESSED columns')
                                print (self.records.readings[test]['devices'][device + '_PROCESSED']['data'].columns)
                                print ('TARGET columns')
                                print (self.records.readings[test]['devices'][device]['data'].columns)
                                cols_to_use = self.records.readings[test]['devices'][device + '_PROCESSED']['data'].columns.difference(self.records.readings[test]['devices'][device]['data'].columns)
                                # self.std_out('cols_to_use')
                                # self.std_out(cols_to_use)
                                # self.records.readings[test]['devices'][device]['data'].join(self.records.readings[test]['devices'][device  + '_PROCESSED']['data'][cols_to_use], left_index=True, right_index=True, how='outer')
                                self.records.readings[test]['devices'][device]['data'] = self.records.readings[test]['devices'][device]['data'].combine_first(self.records.readings[test]['devices'][device + '_PROCESSED']['data'])
                                self.records.readings[test]['devices'].pop(device + '_PROCESSED')
                                print ('FINAL columns')
                                print (self.records.readings[test]['devices'][device]['data'].columns)
                            else:
                                self.std_out("No available PROCESSED data to combine with")

                # TO-DO put other pre-processing options here - filtering, smoothing?
                except:
                    self.std_out("Problem pre-processing test {}".format(test))
                    traceback.print_exc()
                    return False
                else:
                    self.std_out("Pre-processing test {} OK ".format(test))
            
            return True

    def sanity_checks(self, task, tests, has_model):

        # Sanity check for test presence
        if not all([self.available_tests.__contains__(i) for i in tests]):
            self.std_out ('Not all tests are available, review data input')
            return False
        
        else:
            # Cosmetic output
            self.std_out('Test presence check passed')
            
            # Check here if all the tuple_features are in each of the tests (accounting for the pre_processing)
            if has_model:
                
                # Get features
                features = task['model']['data']['features']
                features_names = [features[key] for key in features.keys() if key != 'REF']
                reference_name = features['REF']

                datasets = ['train', 'test']

                pollutant_index = {'CO': 1, 'NO2': 2, 'O3': 3}

                # Do it for both, train and test, if they exist
                for dataset in datasets:
                    self.std_out('Checking validity of {} input'.format(dataset))
                    # Check for datasets that are not reference
                    if dataset in task['model']['data'].keys():
                        for test in task['model']['data'][dataset].keys():
                            for device in task['model']['data'][dataset][test]['devices']:
                                # Get columns of the test
                                all_columns = list(self.records.readings[test]['devices'][device]['data'].columns)

                                # Add the pre-processing ones and if we can pre-process
                                if 'pre-processing' in task.keys(): 
                                    if 'alphasense'in task['pre-processing'].keys():
                                        # Check for each pollutant
                                        for pollutant in task['pre-processing']['alphasense']['methods'].keys():
                                            self.std_out('Checking pre-processing for {}'.format(pollutant))
                                            # Define who should be the columns
                                            minimum_alpha_columns = ['GB_{}W'.format(pollutant_index[pollutant]), 'GB_{}A'.format(pollutant_index[pollutant])]
                                            # Check if we can pre-process alphasense data with working and auxiliary electrode
                                            if not all([all_columns.__contains__(i) for i in minimum_alpha_columns]): return False
                                            # We know we can pre-process, add the result of the pre-process to the columns
                                            
                                            all_columns.append(pollutant + '_' + self.append_alphasense)
                                        
                                        if not all([all_columns.__contains__(i) for i in features_names]): return False
                                        
                            # In case of training dataset, check that the reference exists
                            if dataset == 'train':
                                found_ref = False
                                for device in self.records.readings[test]['devices'].keys():
                                    if 'is_reference' in self.records.readings[test]['devices'][device].keys():
                                        reference_dataframe = device
                                        if not (reference_name in self.records.readings[test]['devices'][device]['data'].columns) and not found_ref: 
                                            found_ref = False
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

            if 'model' in self.tasks[task].keys():
                model_name = self.tasks[task]['model']['model_name'] 
                
                # Create model_wrapper instance
                self.std_out('Task {} involves modeling, initialising model'.format(task))
                current_model = model_wrapper(self.tasks[task]['model'], verbose = self.verbose)
                
                # Parse current model instance
                has_model = True

                # Ignore extra data in json if present in the task
                if 'data' in self.tasks[task].keys(): self.std_out('Ignoring additional data in task')

                # Model dataset names and options
                tests = list(set(itertools.chain(self.tasks[task]['model']['data']['train'], self.tasks[task]['model']['data']['test'])))
                data_dict = self.tasks[task]['model']['data']
            else:
                self.std_out('No model involved in task {}'.format(task))
     
                # Model dataset names and options
                tests = list(self.tasks[task]['data']['datasets'])
                data_dict = self.tasks[task]['data']
                has_model = False

            # Cosmetic output
            self.std_out('Loading data...')

            # Load data
            if not self.load_data(tests, data_dict['options']):
                self.std_out('Failed loading data')
                return

            # Cosmetic output
            self.std_out('Sanity checks...')
            # Perform sanity check
            if not self.sanity_checks(self.tasks[task], tests, has_model):
                self.std_out('Failed sanity checks')
                return
                               
            # Pre-process data
            if 'pre-processing' in self.tasks[task].keys():
                # Cosmetic output
                self.std_out('Pre-processing requested...')

                if not self.pre_process_data(self.tasks[task], tests, data_dict, has_model): 
                    self.std_out('Failed preprocessing')
                    return
            
            # Perform model stuff
            if has_model:
                try:
                    # Prepare dataframe for training
                    train_dataset = list(current_model.data['train'].keys())[0]
                    self.std_out (f'Train dataset: {train_dataset}')

                    if not self.records.prepare_dataframe_model(current_model):
                        self.std_out('Failed training dataset dataframe preparation for model')
                        return
                    
                    # Train Model based on training dataset
                    current_model.training(self.records.readings[train_dataset]['models'][model_name])
                    # Evaluate Model in train data
                    device = current_model.data['train'][train_dataset]['devices']
                    # Dirty horrible workaround
                    if type(device) == list: device = device[0]
                    prediction_name = device + '_' + model_name

                    self.std_out('Predicting {} for device {} in {}'.format(prediction_name, device, train_dataset))
                    # Get prediction for train
                    prediction = current_model.predict_channels(self.records.readings[train_dataset]['devices'][device]['data'], prediction_name)
                    # Combine it in readings
                    self.records.readings[train_dataset]['devices'][device]['data'].combine_first(prediction)

                    # Evaluate Model in test data
                    for test_dataset in current_model.data['test'].keys():
                        for device in current_model.data['test'][test_dataset]['devices']:
                            prediction_name = device + '_' + model_name
                            self.std_out('Predicting {} for device {} in {}'.format(prediction_name, device, test_dataset))

                            # Get prediction for test                            
                            if current_model.data['test'][test_dataset]['reference_device'] in self.records.readings[test_dataset]['devices'].keys():
                                reference = self.records.readings[test_dataset]['devices'][current_model.data['test'][test_dataset]['reference_device']]['data'][current_model.data['features']['REF']]
                            else:
                                reference = None
                            
                            prediction = current_model.predict_channels(self.records.readings[test_dataset]['devices'][device]['data'], prediction_name, reference)

                            # Combine it in readings
                            self.records.readings[test_dataset]['devices'][device]['data'].combine_first(prediction)

                    # Export model data if requested
                    if data_dict['options']["export_data"] is not None:
                        dataFrameExport = current_model.dataFrameTrain.copy()
                        dataFrameExport = dataFrameExport.combine_first(current_model.dataFrameTest)
                    else:
                        dataFrameExport = None
                    
                    # Save model in session
                    if current_model.options['session_active_model']:
                        self.std_out ('Saving model in session records...')
                        self.records.archive_model(train_dataset, current_model, dataFrameExport)

                    # Export model if requested
                    if current_model.options['export_model']:
                        current_model.export(self.records.modelDirectory)
                except:
                    traceback.print_exc()
                    pass
            
            if data_dict['options']["export_data"] is not None:
                try: 
                    if data_dict['options']["export_data"] == 'Only Generic':
                        all_channels = False
                        include_raw = True
                        include_processed = True
                    if data_dict['options']["export_data"] == 'Only Processed':
                        all_channels = False
                        include_raw = False
                        include_processed = True
                    elif data_dict['options']["export_data"] == 'Only Raw':
                        all_channels = False
                        include_raw = True
                        include_processed = False
                    elif data_dict['options']["export_data"] == 'All':
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
                    if has_model:
                        # Do it for both, train and test, if they exist
                        for dataset in ['train', 'test']:
                            # Check for datasets that are not reference
                            if dataset in self.tasks[task]['model']['data'].keys():
                                for test in self.tasks[task]['model']['data'][dataset].keys():
                                    for device in self.tasks[task]['model']['data'][dataset][test]:
                                        self.std_out('Exporting data of device {} in test {} to processed folder'.format(device, test))
                                        self.records.export_data(test, device, all_channels = all_channels, 
                                                            include_raw = include_raw, include_processed = include_processed, 
                                                            rename = data_dict['options']['rename_export_data'], 
                                                            to_processed_folder = True, 
                                                            forced_overwrite = True)
                    else:
                        for test in tests:
                            # Get devices
                            list_devices = self.tasks[task]['data']['datasets'][test]

                            # Iterate and export them
                            for device in list_devices:
                                self.std_out('Exporting data of device {} in test {} to processed folder'.format(device, test))
                                self.records.export_data(test, device, all_channels = all_channels, 
                                    include_raw = include_raw, include_processed = include_processed, 
                                    rename = data_dict['options']['rename_export_data'],
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
                                    for device in self.records.readings[self.tasks[task]["plot"][plot_description]['data']['test']]['devices'].keys():
                                        channel = self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["channel"]
                                        if channel in self.records.readings[self.tasks[task]["plot"][plot_description]['data']['test']]['devices'][device]['data'].columns:
                                            list_devices_plot.append(device)
                                        else:
                                            self.std_out('Trace ({}) not in readings in device {}'.format(channel, device))
                                else:
                                    list_devices_plot = self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["device"]      
                            # Make a plot for each device
                            for device in list_devices_plot:
                                # Rename the traces
                                for trace in self.tasks[task]["plot"][plot_description]['data']['traces']:
                                    self.tasks[task]['plot'][plot_description]['data']['traces'][trace]["device"] = device
                                
                                # Rename the export name
                                self.tasks[task]['plot'][plot_description]['options']['file_name'] = original_filename + '_' + device

                                # plot it
                                plot_object = plot_wrapper(self.tasks[task]["plot"][plot_description], True)
                                plot_object.plot(self.records)
                                
                                # Export if we have how to
                                if self.tasks[task]["plot"][plot_description]['options']['export_path'] is not None and self.tasks[task]["plot"][plot_description]['options']['file_name'] is not None:
                                    plot_object.export_plot()
                                plot_object.clean_plot()
                        # Or only one
                        else:    
                            plot_object = plot_wrapper(self.tasks[task]["plot"][plot_description], True)
                            plot_object.plot(self.records)
                            if self.tasks[task]["plot"][plot_description]['options']['export_path'] is not None and self.tasks[task]["plot"][plot_description]['options']['file_name'] is not None:
                                plot_object.export_plot()
                            plot_object.clean_plot()
                except:
                    traceback.print_exc()
                    pass
        
        self.std_out('Finished task {}'.format(task))
        self.std_out('---')