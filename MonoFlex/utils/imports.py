import sys
from importlib import util


def import_file(module_name, file_path, make_importable=False):
    spec = util.spec_from_file_location(module_name, file_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if make_importable:
        sys.modules[module_name] = module_name
        
    return module