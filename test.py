import torch
from src.model import DopeNetwork
import sys
from src.auxiliar import get_DataLoaders
import src.custom_transform as ct
import src.args_parser as ar
import src.auxiliar as aux

import os
import random
import datetime
import warnings
from pathlib import Path
import multiprocessing as mp



# Import the necessary modules


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

# Get the first batch of data from the training data loader
batch = next(iter(trainingdata))

print(batch['translations'])  # Should be (1, 3, 480, 640)
print(batch['rotations'])  # Should be torch.float32

# shapes
print(f"Batch shape: {batch['translations'].shape}")
print(f"Batch shape: {batch['rotations'].shape}")

print("Has object: ", batch['has_points_belief'])

model = DopeNetwork()


# Create a 480x640 RGB image (3 channels)
input_image = torch.randn(1, 3, 480, 640)  # Batch size of 1, 3 channels, 480x640

# Pass the image through the model
output = model(input_image)

print(output)

# Print the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")