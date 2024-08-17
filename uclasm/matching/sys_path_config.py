# path_config.py

import sys

def extend_sys_path():
    additional_paths = [
        "./",
        "./GraphGPS/",
        "./uclasm/",
        "./NSUBS/",
    ]

    sys.path.extend(additional_paths)
