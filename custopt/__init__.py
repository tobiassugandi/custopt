# __init__.py for mypackage

import os
import pkgutil
import importlib
import logging

# Package-wide constants
__version__ = '0.1.0'
__author__ = 'tobias'

# Optional: Setting up package-wide logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"{__name__} version {__version__} initialized.")

# Function to dynamically import all submodules and set __all__
def initialize():
    logger.info("Running package initialization code")

    # Discover and import all submodules
    package_dir = os.path.dirname(__file__)
    for module_info in pkgutil.iter_modules([package_dir]):
        if module_info.ispkg:
            continue
        module_name = module_info.name
        full_module_name = f"{__name__}.{module_name}"
        module = importlib.import_module(full_module_name)
        globals()[module_name] = module
        logger.info(f"Imported module: {full_module_name}")

    # Set __all__ to include all imported submodules
    global __all__
    __all__ = [name for _, name, _ in pkgutil.iter_modules([package_dir])]

# Run initialization code
initialize()
