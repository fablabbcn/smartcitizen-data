from scdata.tools.custom_logger import logger

def find_by_field(models, value, field):
    try:
        item = next(model for _, model in enumerate(models) if model.__getattribute__(field) == value)
    except StopIteration:
        # logger.info(f'Column {field} or value {value} not in models')
        pass
    else:
        return item
    return None