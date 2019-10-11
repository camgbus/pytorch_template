################################################################################
# Defines and parses arguments
################################################################################

import argparse

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='InceptionV3')
    parser.add_argument('--pretrained', dest='pretrained', default=False, action='store_true', help='Use pretrained parameters. Only valid for transfer learning, i.e. with models imported from torchvision.')
    parser.add_argument('--freeze_weights', dest='freeze_weights', default=False, action='store_true', help='Freeze the weights from the first model layers to speed up training. Only valid for transfer learning, i.e. with models imported from torchvision.')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8, help='Nr. workers for data loading')
    parser.add_argument('--device', type=int, default=0, help='GPU nr.')
    return parser

def parse_args(argv):
    """Parses arguments passed from the console as, e.g.
    'python ptt/main.py --epochs 3' """
    parser = _get_parser()
    args = parser.parse_args(argv)
    return args

def parse_dict_as_args(dictionary):
    """Parses arguments given in a dictionary form"""
    argv = []
    for key, value in dictionary.items():
        if isinstance(value, bool):
            if value:
                argv.append('--'+key)
        else:
            argv.append('--'+key)
            argv.append(str(value))
    return parse_args(argv)