# ------------------------------------------------------------------------------
# Defines and parses arguments
# ------------------------------------------------------------------------------

import argparse

def _get_parser():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset_class_path', type=str)
    parser.add_argument('--agent_class_path', type=str, default='NormalNN', help="The class path of the agent.")
    parser.add_argument('--model_class_path', type=str, default='NormalNN', help="The class path of the model.")

    # Experiment
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--experiment_notes', type=str, default='')

    # Dataset
    parser.add_argument('--restore_dataset', dest='restore_dataset', default=False, action='store_true', help='Reload dataset object.')
    parser.add_argument('--nr_runs', type=int, default=1)
    parser.add_argument('--cross_validation', dest='cross_validation', default=False, action='store_true')
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data from train+validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help="Only relevant if 'cross_validation' if false.")

    # Training
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--schedule', nargs="+", type=int, default=[2],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")

    config = {'batch_size':128, 'lr':1e-3, 'momentum':0.9, 'device':'cuda:0', 'nr_epochs': 10, 'tracking_interval': 2}

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