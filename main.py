# Test for bfloat16 training with the DOPE model

from src.new_torch.model import DopeNetwork

import sys
import datetime
import os
import random
import warnings
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

for parent in Path.cwd().parents:
    sys.path.append(str(parent))

import src.new_torch.args_parser as ar
import src.new_torch.custom_transform as ct
import src.new_torch.auxiliar as aux
import src.new_torch.run_network as rn
import multiprocessing as mp

def main():
    # Import the necessary modules
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    full_path = os.getcwd()
    sys.path.append(full_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse arguments
    opt = ar.parse_args(full_path, False)
    print(opt.outf)

    # set the manual seed.
    random.seed(opt.manualseed)
    torch.manual_seed(opt.manualseed)
    torch.cuda.manual_seed_all(opt.manualseed)
    # Create output folder and files
    aux.create_output_folder(opt)
    print ("start:" , datetime.datetime.now().time())

    # Initialize the image transforms
    transform, preprocessing_transform, mean, std  = ct.get_transform()

    # Get the DataLoaders
    train_dataset, test_dataset, trainingdata, testingdata = aux.get_DataLoaders(opt, preprocessing_transform, transform)

    net = DopeNetwork(stop_at_stage=6)
    net = net.to(device)

    # Load the weights if a pretrained model is provided
    aux.load_dicts(opt, net,device)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters,lr=opt.lr, weight_decay= 0.0005)

    nb_update_network = 0
    
    torch.backends.cudnn.benchmark = True
    pbar = tqdm(range(1, opt.epochs + 1))    

    for epoch in pbar:        
        rn._runnetwork(epoch, trainingdata, testingdata, pbar=pbar,
                    optimizer=optimizer, opt = opt,
                    net = net, device = device)
            
        nb_update_network += 1
        # Stop training if nb_update_network exceeds the limit
        if opt.nbupdates is not None and nb_update_network > int(opt.nbupdates):
            break

    print("end:", datetime.datetime.now().time())

if __name__ == "__main__":
    mp.freeze_support() # Required for Windows
    main()