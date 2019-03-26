import os
import sys

try:
    import fe621
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath('../'))
    os.chdir(os.path.abspath('../'))
    import fe621
