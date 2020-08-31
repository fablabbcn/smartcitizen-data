from scdata.utils import std_out
from joblib import dump, load
from scdata._config import config
from os.path import join, exists
from os import makedirs

def model_export(name = None, path = None, model = None, variables = None, hyperparameters = None, options = None, metrics = None):
    
    if name is None:
        std_out('No name specified', 'ERROR')
        return False

    if path is None: 
        path = config.paths['models']
    
    modeldir = join(path, name)

    if not exists(modeldir): 
        makedirs(modeldir)

    filename = join(modeldir, name)

    if hyperparameters is not None:
        std_out('Saving hyperparameters')
        dump(hyperparameters, filename + '_hyperparameters.sav')
    
    if variables is not None:
        std_out('Saving variables')
        dump(variables, filename + '_variables.sav')
    else: return False

    if model is not None:
        std_out('Saving model')
        dump(model, filename + '_model.sav', compress = 3)
    else: return False

    if options is not None:
        std_out('Saving options')
        dump(options, filename + '_options.sav')
    else: return False

    if metrics is not None:
        std_out('Saving metrics')
        dump(metrics, filename + '_metrics.sav')
    else: return False  

    std_out(f'Model: {name} saved in {modeldir}', 'SUCCESS')

    return True

def model_load(name = '', path = None):

    if path is None: 
        path = config.paths['models']
    
    modeldir = join(path, name)
    filename = join(modeldir, name)

    std_out('Loading hyperparameters')
    hyperparameters = load(filename + '_hyperparameters.sav')
    
    std_out('Loading variables')
    variables = load(filename + '_variables.sav')

    std_out('Loading model')
    model = load(filename + '_model.sav')

    std_out('Loading options')
    options = load(filename + '_options.sav')   

    std_out('Loading metrics')
    metrics = load(filename + '_metrics.sav')

    std_out(f'Model: {name} loaded', 'SUCCESS')

    return hyperparameters, variables, model, options, metrics