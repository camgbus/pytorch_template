################################################################################
# Dynamically select the model
################################################################################

def get_model(model_name):
    """Returns a model class, which must be instantiated.
    Note that for this to work, model classes must live in a module with the 
    same name as the class, and be located in the 'models' subdirectory"""
    module = __import__('ptt.models.'+model_name)
    module = getattr(module, 'models')
    module = getattr(module, model_name)
    class_ref = getattr(module, model_name)
    return class_ref