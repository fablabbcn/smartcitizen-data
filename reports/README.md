How to generate these reports:

jupyter nbconvert --config sc_nbconvert_config.py batch_dispersion_advanced.ipynb --sc_Preprocessor.expression="show_only_output" --to html --TemplateExporter.template_file="full_sc" --output-dir=../reports --output=1909_Deviveries