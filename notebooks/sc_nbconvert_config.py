c = get_config()

#Export all the notebooks in the current directory to the sphinx_howto format.
c.Exporter.preprocessors = ['sc_preprocessor.SCPreprocessor']
# c.NbConvertApp.writer_class = 'notebook_copy_writer.NotebookCopyWriter'