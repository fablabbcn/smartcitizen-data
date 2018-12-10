# TestCreator
Log - Test Folder structure creation for logs from the kits, with or without reference.

## Usage

1. Do not delete any of the folders below:
- RAW_DATA
- data/*

2. Put the raw csv files from the KIT in the RAW_DATA directory
3. Launch the python concant_script.py with the files you want to concatenate in a folder (it will not know which files are from which kit)
4. Launch the Test Creation.ipynb and go through the cells. 
5. Input your data manually (WIP). Reference to them in the cell inside the jupyter notebook.

This will create a yaml file with the necessary test description.
