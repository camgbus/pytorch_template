from ptt.utils.argument_parsing import parse_dict_as_args

def test_arg_parsing():
    dict_args = {'model_name': 'InceptionV3', 
        'freeze_weights': True, 
        'pretrained': False, 
        'epochs': 3}
    args = parse_dict_as_args(dict_args)
    assert args.model_name == 'InceptionV3'
    assert args.freeze_weights == True
    assert args.pretrained == False
    assert args.epochs == 3