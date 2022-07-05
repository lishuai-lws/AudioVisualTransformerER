from ast import arg
import torch
from config.config import shared_configs
from utils.logger import LOGGER, add_log_to_file
from utils.misc import set_random_seed

def main(opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("device:{}".format(device))

    set_random_seed(opts.seed)
    LOGGER.info("Loading dataset...")









if __name__=="__main__":
    args = shared_configs.get_pretrain_args()
    main(args)