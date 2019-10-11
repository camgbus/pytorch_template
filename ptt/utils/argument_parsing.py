################################################################################
# Pases arguments
################################################################################

import argparse

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args(argv)
    return args

def parse_dict_as_args(dictionary):
    pass