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

    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    model.apply(initialize_weights)

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
            loss = 3*criterion(outputs, labels)            

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    
            print(f"Epoch [{epoch+1}/{opt.epochs}], Batch [{i+1}/{len(trainingdata)}], Loss: {loss.item():.4f}")

            if i % 10 == 0:
                print("Label: ", labels)
                print("Output: ", outputs)

        model.eval()
        with torch.no_grad():
            all_outputs = []
            all_labels = []
            for data in trainingdata:
                inputs = data['image'].to(device)
                translations = data['translations'].to(device)
                rotations = data['rotations'].to(device)
                has_points_belief = data['has_points_belief'].to(device)
                labels_2 = torch.concat([rotations, translations], dim=1)

                outputs = model(inputs)
                all_outputs.append(outputs[:, 1:8].cpu())
                all_labels.append(labels_2.cpu())

            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            ss_total = torch.sum((all_labels - torch.mean(all_labels)) ** 2)
            ss_residual = torch.sum((all_labels - all_outputs) ** 2)
            r2_score = 1 - (ss_residual / ss_total)

        print(f"Epoch [{epoch+1}/{opt.epochs}] R^2 Score for the dataset: {r2_score.item():.4f}")
        model.train()

if __name__ == "__main__":
    mp.freeze_support()
    main()