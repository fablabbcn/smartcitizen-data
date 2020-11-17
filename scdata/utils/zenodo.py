''' Implementation of zenodo export '''

from scdata._config import config
from scdata.utils import std_out, get_tests_log, include_footer
from scdata import Test
import json, yaml
from os.path import join, dirname, getsize, exists
from requests import post
from os import environ

def zenodo_upload(upload_descritor, sandbox = True, dry_run = True):
	'''
	This section uses the code inspired by this repo https://github.com/darvasd/upload-to-zenodo
	Uploads a series of tests to zenodo.org using a template in /zenodo_templates and the descriptor
	file in data/uploads. It will need a ZENODO_TOKEN environment variable to work
	The submission needs an additional "Publish" step. 
	This can also be done from a script, but to be on the safe side, it is not included. 
	(The attached file cannot be changed after publication)
	Parameters
	----------
		upload_descritor: string
			The descriptor's filename (yaml) in the data/uploads. Check options in the example yaml
			Option defaults:
				include_processed_data: True (for tests)
				include_footer_doi: True (for pdfs)
				include_td_html: False (for yaml test_description in html)
			Upload types:
				publication: not implemented
				dataset: implemented (can contain several tests in it)
		sandbox: boolean
			True uses zenodo's sandbox at sandbox.zenodo.org
		dry_run:
			True fakes uploads everything to check
	Returns
	----------
		True if all data is uploaded, False otherwise

	'''
	
	def fill_template(individual_descriptor, descriptor_file_name, upload_type = 'dataset'):
		# Open base template with all keys

		if upload_type == 'dataset': template_file_name = 'template_zenodo_dataset'
		elif upload_type == 'publication': template_file_name = 'template_zenodo_publication'

		with open (join(dirname(__file__), 'zenodo_templates', f'{template_file_name}.json'), 'r') as template_file:
			template = json.load(template_file)

		filled_template = template

		# Fill it up for each key
		for key in individual_descriptor.keys():

			value = individual_descriptor[key]

			if key in filled_template['metadata'].keys():
				filled_template['metadata'][key] = value

		with open (join(config.paths['uploads'], descriptor_file_name), 'w') as descriptor_json:
			json.dump(filled_template, descriptor_json, ensure_ascii=True)
			std_out(f'Created descriptor file for {descriptor_file_name}', 'SUCCESS')
		
		return json.dumps(filled_template)

	def get_submission_id(metadata, base_url):
		url = f"{base_url}/api/deposit/depositions"

		headers = {"Content-Type": "application/json"}

		response = post(url, params={'access_token': environ['ZENODO_TOKEN']}, data = metadata, headers = headers)
		if response.status_code > 210:
			std_out("Error happened during submission, status code: " + str(response.status_code), 'ERROR')
			std_out(response.json()['message'], 'ERROR')
			return None

		# Get the submission ID
		submission_id = json.loads(response.text)["id"]

		return submission_id

	def upload_file(url, upload_metadata, files):
		response = post(url, params={'access_token': environ['ZENODO_TOKEN']}, data = upload_metadata, files=files)
		return response.status_code		

	std_out(f'Uploading {upload_descritor} to zenodo')

	if dry_run: std_out(f'Dry run. Verify output before setting dry_run to False', 'WARNING')
	
	# Sandbox or not
	if sandbox: 
		std_out(f'Using sandbox. Verify output before setting sandbox to False', 'WARNING')
		base_url = config.zenodo_sandbox_base_url
	else: base_url = config.zenodo_real_base_url
	
	if '.yaml' not in upload_descritor: upload_descritor = upload_descritor + '.yaml'
	
	with open (join(config.paths['uploads'], upload_descritor), 'r') as descriptor_file:
		descriptor = yaml.load(descriptor_file, Loader = yaml.SafeLoader)

	for key in descriptor:

		# Set options for processed and raw uploads
		stage_list = ['base']
		
		if 'options' in descriptor[key].keys(): options = descriptor[key]['options']
		else: options = {'include_processed_data': False, 'include_footer_doi': True, 'include_td_html': False}

		# Defaults
		if 'include_processed_data' not in options: options['include_processed_data'] = False
		if 'include_footer_doi' not in options: options['include_footer_doi'] = True
		if 'include_td_html' not in options: options['include_td_html'] = False
		
		if options['include_processed_data']: stage_list.append('processed')
		std_out(f'Options {options}')

		# Fill template
		if 'upload_type' in descriptor[key].keys(): upload_type = descriptor[key]['upload_type']
		else: 
			std_out(f'Upload type not set for key {key}. Skipping', 'ERROR')
			continue

		metadata = fill_template(descriptor[key], key, upload_type = upload_type)
		
		# Get submission ID
		if not dry_run: submission_id = get_submission_id(metadata, base_url)
		else: submission_id = 0

		if submission_id is not None:
			
			# Dataset upload
			if upload_type == 'dataset':
				# Get the tests to upload
				tests = descriptor[key]['tests']
				
				# Get url where to post the files
				url = f"{base_url}/api/deposit/depositions/{submission_id}/files"

				test_logs = get_tests_log()			

				for test_name in tests:
					
					# Get test path
					std_out(f'Uploading data from test {test_name}')
					
					test_path = test_logs[test_name]['path']

					# Upload the test descriptor (yaml (and html) format)
					td_upload = ['yaml']
					with open (join(test_path, 'test_description.yaml'), 'r') as td: 
						yaml_td = yaml.load(td, Loader = yaml.SafeLoader)
					
					if options['include_td_html']:
						html_td = td_to_html(yaml_td, test_path)
						if html_td: td_upload.append('html')

					for td_format in td_upload:

						upload_metadata = {'name': f'test_description_{test_name}.{td_format}'}
					
						files = {'file': open(join(test_path, f'test_description.{td_format}'), 'rb')}
						file_size = getsize(join(test_path, f'test_description.{td_format}'))/(1024*1024.0*1024)
					
						if file_size > 50: std_out(f'File size for {test_name} is over 50Gb ({file_size})', 'WARNING')
					
						if not dry_run: status_code = upload_file(url, upload_metadata, files)
						else: status_code = 200
					
						if status_code > 210: 
							std_out ("Error happened during file upload, status code: " + str(status_code), 'ERROR')
							return
						else:
							std_out(f"{upload_metadata['name']} submitted with submission ID = \
								 	{submission_id} (DOI: 10.5281/zenodo.{submission_id})" ,"SUCCESS")
					
					# Load the api devices to have them up to date in the cache
					if any(yaml_td['devices'][device]['source'] == 'api' for device in yaml_td['devices'].keys()): 
						test = Test(test_name)
						test.load(options = {'store_cached_api': True})
					
					for device in yaml_td['devices'].keys():
						
						std_out(f'Uploading data from device {device}')
						
						# Upload basic and processed data
						for file_stage in stage_list:
							
							file_path = ''
							
							try:

								# Find device files
								if file_stage == 'processed': 

									file_name = f'{device}.csv'
									file_path = join(test_path, 'processed', file_name)
									upload_metadata = {'name': f'{device}_PROCESSED.csv'}
								
								elif file_stage == 'base':
									if 'csv' in yaml_td['devices'][device]['source']:

										file_name = yaml_td['devices'][device]['processed_data_file']
										file_path = join(test_path, file_name)
									
									elif yaml_td['devices'][device]['source'] == 'api':
										file_name = f'{device}.csv'
										file_path = join(test_path, 'cached', file_name)
									
									upload_metadata = {'name': file_name}

								if file_path != '':

									files = {'file': open(file_path, 'rb')}
									file_size = getsize(file_path)/(1024*1024.0*1024)
									
									if file_size > 50: std_out(f'File size for {file_name} over 50Gb ({file_size})', 'WARNING')
									
									if not dry_run: status_code = upload_file(url, upload_metadata, files)
									else: status_code = 200
									
									if status_code > 210: 
										std_out (f"Error happened during file upload, status code: {status_code}. Skipping", 'ERROR')
										continue

									std_out(f"{upload_metadata['name']} submitted with submission ID =\
												{submission_id} (DOI: 10.5281/zenodo.{submission_id})" ,"SUCCESS")    
							except:
								if not exists(file_path): std_out(f'File {file_name} does not exist (type = {file_stage}). Skipping', 'ERROR')
								# print_exc()
								pass
				
				# Check if we have a report in the keys
				if 'report' in descriptor[key].keys():

					for file_name in descriptor[key]['report']:

						file_path = join(config.paths['uploads'], file_name)
						
						if options['include_footer_doi'] and file_name.endswith('.pdf'):

							output_file_path = file_path[:file_path.index('.pdf')] + '_doi.pdf'
							include_footer(file_path, output_file_path, link = f'https://doi.org/10.5281/zenodo.{submission_id}')
							file_path = output_file_path
						
						upload_metadata = {'name': file_name}
						files = {'file': open(file_path, 'rb')}
						file_size = getsize(file_path)/(1024*1024.0*1024)
						
						if file_size > 50: std_out(f'File size for {file_name} is over 50Gb({file_size})', 'WARNING')
						
						if not dry_run: status_code = upload_file(url, upload_metadata, files)
						else: status_code = 200

						if status_code > 210: 
							std_out (f"Error happened during file upload, status code: {status_code}. Skipping", 'ERROR')
							continue

						std_out(f"{upload_metadata['name']} submitted with submission ID = \
								{submission_id} (DOI: 10.5281/zenodo.{submission_id})" ,"SUCCESS")

			if upload_type == 'publication':
				std_out('Not implemented')
				return False
			
			std_out(f'Submission completed - (DOI: 10.5281/zenodo.{submission_id})', 'SUCCESS')
			std_out(f'------------------------------------------------------------')
		else:
			std_out(f'Submission ID error', 'ERROR')
			continue
	return True