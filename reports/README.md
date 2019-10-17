Welcome to the reports folder. Here you can find documentation on technical development of the sensors, as well as dispersion analysis for the different deliveries of kits that we have done. The raw ipynb for generating the reports is included, as well as a rendered pdf, ready to share. 

These reports are automatically generated with a templating system and the `jupyter nbconvert` tool, for more on this check below.

## Reports

Execute this command in your terminal for converting the notebooks. You should be in the `notebooks` folder of the framework:

```
> cd notebooks
> jupyter nbconvert --config sc_nbconvert_config.py <input_nb.ipynb> --sc_Preprocessor.expression="show_only_output" --to html --TemplateExporter.template_file="full_sc" --output-dir=../reports --output=<filename>
```