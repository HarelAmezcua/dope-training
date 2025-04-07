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
import torch.nn as nn

# Import the necessary modules

def main():
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
    """
    iterator = iter(trainingdata)
    data = next(iterator)
    print(data['image'].shape)
    print(data['translations'].shape)
    print(data['rotations'].shape)
    print(data['has_points_belief'].shape)
    sys.exit(0)"""


    model = DopeNetwork()
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    criterion_2 = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)


    # Training loop
    for epoch in range(opt.epochs):
        for i, data in enumerate(trainingdata):
            # Get the input data and labels
            inputs = data['image'].to(device)
            translations = data['translations'].to(device)
            rotations = data['rotations'].to(device)
            has_points_belief = data['has_points_belief'].to(device)
            labels = torch.concat([has_points_belief.view(opt.subbatchsize, 1),rotations, translations], dim=1)
            labels_2 = torch.concat([rotations, translations], dim=1)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = 10*criterion(outputs[:,1:8], labels_2)
            loss += criterion_2(outputs[:, 0], labels[:, 0].float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Apply sigmoid to the first element of the batch output
            outputs[:, 0] = torch.sigmoid(outputs[:, 0])

            # Compute R^2 score
            ss_total = torch.sum((labels_2 - torch.mean(labels_2, dim=0)) ** 2)
            ss_residual = torch.sum((labels_2 - outputs[:, 1:8]) ** 2)
            r2_score = 1 - (ss_residual / ss_total)

            print(f"R^2 Score: {r2_score.item():.4f}")


            print(f"Epoch [{epoch+1}/{opt.epochs}], Batch [{i+1}/{len(trainingdata)}], Loss: {loss.item():.4f}, R^2 Score: {r2_score.item():.4f}")

            if i % 10 == 0:
                print("Label: ", labels)
                print("Output: ", outputs)


if __name__ == "__main__":
    mp.freeze_support()
    main()