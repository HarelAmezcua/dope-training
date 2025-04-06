# Test for bfloat16 training with the DOPE model
import sys
import datetime
import os
import random
import warnings
from pathlib import Path

import torch

for parent in Path.cwd().parents:
    sys.path.append(str(parent))

import src.new_torch.argument_parser as ar
import src.new_torch.custom_transform as ct
import src.new_torch.auxiliar as aux

warnings.filterwarnings("ignore")

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

iterator = iter(trainingdata)
batch = next(iterator)