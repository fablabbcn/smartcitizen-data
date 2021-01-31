## About this folder

You can use this folder to put your working notebooks. Some examples are also found in the `/examples` folder.

## [DEPRECATED] Notebooks export

Tools are provided to generate test or analysis reports, with a custom template. These are generated with the `jupyter nbconvert` using the preprocessor and tools in the `notebooks` and `template` folder. To generate a report, follow the steps:

1. Decide how you want the notebook to look: either only with markdown notes, with cell outputs, or with the code included.
2. Tag the cells in your notebook. You can use the [Jupyter Lab Celltags](https://github.com/jupyterlab/jupyterlab-celltags) extension. Don't tag the cells you want to hide, and tag the `code cells` you want to show with `show_only_output`. This can be changed and add more tags, but we keep it this way for simplicity
3. Go to the notebooks folder and run the command:
```
cd notebooks
jupyter nbconvert --config sc_nbconvert_config.py <notebook_name.ipynb> --sc_Preprocessor.expression="show_only_output" --to html --TemplateExporter.template_file=./templates/full_sc --output-dir=../reports --output=<OUTPUT_NAME>
```

Where:

- `sc_nbconvert_config.py` is the config
- `<notebook_name.ipynb>` is the notebook you want to export, saved with the tags
- `"show_only_output"` is a boolean expression that is evaluated for each of the cells. If true, the cell is shown
- `./templates/full_sc` is the default template we have created
- `../reports` is the directory where we will put the `html` report
- `OUTPUT_NAME` is the name for the export
