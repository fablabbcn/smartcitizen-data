from scdata.tools.custom_logger import logger
from joblib import dump, load
from scdata._config import config
from os.path import join, exists
from os import makedirs

def model_export(name = None, path = None, model = None, variables = None, hyperparameters = None, options = None, metrics = None):

    if name is None:
        logger.error('No name specified')
        return False

    if path is None:
        path = config.paths['models']

    modeldir = join(path, name)

    if not exists(modeldir):
        makedirs(modeldir)

    filename = join(modeldir, name)

    if hyperparameters is not None:
        logger.info('Saving hyperparameters')
        dump(hyperparameters, filename + '_hyperparameters.sav')

    if variables is not None:
        logger.info('Saving variables')
        dump(variables, filename + '_variables.sav')
    else: return False

    if model is not None:
        logger.info('Saving model')
        dump(model, filename + '_model.sav', compress = 3)
    else: return False

    if options is not None:
        logger.info('Saving options')
        dump(options, filename + '_options.sav')
    else: return False

    if metrics is not None:
        logger.info('Saving metrics')
        dump(metrics, filename + '_metrics.sav')
    else: return False

    logger.info(f'Model: {name} saved in {modeldir}')

    return True

def model_load(name = '', path = None):

    if path is None:
        path = config.paths['models']

    modeldir = join(path, name)
    filename = join(modeldir, name)

    logger.info('Loading hyperparameters')
    hyperparameters = load(filename + '_hyperparameters.sav')

    logger.info('Loading variables')
    variables = load(filename + '_variables.sav')

    logger.info('Loading model')
    model = load(filename + '_model.sav')

    logger.info('Loading options')
    options = load(filename + '_options.sav')

    logger.info('Loading metrics')
    metrics = load(filename + '_metrics.sav')

    logger.info(f'Model: {name} loaded')

    return hyperparameters, variables, model, options, metrics