import argparse
import json
import sys
import os
def parse_with_config(parser):
    args = parser.parse_args()

    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
            if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args,k,v)
    del args.config
    return args

class SharedConfigs(object):
    def __init__(self, desc="shared config class for pretraining and downstream"):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('--seed',type=int,default=42,help="random seed")
        self.cwd = os.getcwd()
        self.parser = parser
        
    def get_pretrain_args(self):
        pass

    def get_data_process_args(self):
        self.parser.add_argument('--config',type=str,default="../config/data_config.json",help="config file")
        args = self.parse_args()
        return args
    def parse_args(self):
        args = parse_with_config(self.parser)
        return args


shared_configs = SharedConfigs()