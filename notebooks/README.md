## Notebooks content

Each notebook in this folder uses scripts from ../src and other notebooks (with widgets) from the ./src_ipynb folder. These notebooks are here for convenience, and their use is not mandatory.

### Main

The main notebook for data analysis is main.ipynb, although other can be configured with the items from src_ipynb. This notebooks contains a generic workflow for data analysis including:

- Data load
- Calculator
- Data export
- Alphasense baseline calibration
- Exploratory data analysis (visualisation)
- Model calibration, export and application

### Dispersion Analysis

This notebook allows to load certain devices and perform basic dispersion analysis and faulty device detection. It loads the data from local tests or the API and calculates automatically the necessary outputs

### Batch Analysis

This notebook is an example of batch analysis, for the tasks contained in /tasks and used by the class `batch.py`. Check the guides in [the official documentation](https://docs.smartcitizen.me/Sensor%20Analysis%20Framework/guides/Analyse%20your%20data%20in%20batch/) for more information.

### Test creation

This notebook allows the creation of a test, as defined in [the official documentation](https://docs.smartcitizen.me/Sensor%20Analysis%20Framework/guides/Organise%20your%20data/)

## Notebooks export

Tools are provided to generate test or analysis reports, with a custom template. These are generated with the `jupyter nbconvert` using the preprocessor and tools in the `notebooks` and `template` folder. To generate a report, follow the steps:

1. Tag the cells in your notebook. You can use the [Jupyter Lab Celltags](https://github.com/jupyterlab/jupyterlab-celltags) extension. Don't tag the cells you want to hide, and tag the ones you want to show with `show_only_output`. This can be changed and add more tags, but we keep it this way for simplicity
2. Go to the notebooks folder:
```
cd notebooks
```
3. Type the command:
```
jupyter nbconvert --config sc_nbconvert_config.py notebook.ipynb --sc_Preprocessor.expression="show_only_output" --to html --TemplateExporter.template_file=./templates/full_sc --output-dir=../reports --output=OUTPUT_NAME
```

Where:

- `sc_nbconvert_config.py` is the config
- `notebook.ipynb` is the notebook you want
- `"show_only_output"` is a boolean expression that is evaluated for each of the cells. If true, the cell is shown
- `./templates/full_sc` is the default template we have created
- `../reports` is the directory where we will put the `html` report
- `OUTPUT_NAME` is the name for the export

This generates an html export containing only the mkdown or code cell outputs, without any code. An example can be seen in `reports/example.html`.
