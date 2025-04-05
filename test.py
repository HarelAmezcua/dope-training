import sys
import os
import random
from pathlib import Path
import warnings
import torch

from src.new_torch.argument_parser import parse_arguments
from src.new_torch.custom_transform import get_custom_transform
from src.new_torch.auxiliar import create_output_folder, get_DataLoaders


def setup_environment():
    """Set up the environment by configuring paths and warnings."""
    # Add all parent directories to sys.path
    sys.path.extend(map(str, Path.cwd().parents))
    full_path = os.getcwd()
    sys.path.append(full_path)
    warnings.filterwarnings("ignore")
    return full_path


def initialize_device():
    """Initialize the device (CPU or GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def set_random_seed(seed):
    """Set the manual seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """Main function to execute the script."""
    # Setup environment and device
    full_path = setup_environment()
    device = initialize_device()

    # Parse arguments
    options = parse_arguments(full_path, False)

    # Set the manual seed
    set_random_seed(options.manualseed)

    # Create output folder
    create_output_folder(options)

    # Initialize the image transforms
    transform, preprocessing_transform, mean, std = get_custom_transform()

    # Get the DataLoaders
    trainingdata, testingdata = get_DataLoaders(options, preprocessing_transform, transform)

    # Get the first batch of training data
    train_loader = iter(trainingdata)
    first_batch = next(train_loader)


if __name__ == "__main__":
    main()