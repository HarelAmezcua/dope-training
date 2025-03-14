import sys


import datetime
import os
import random
import warnings
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.cuda import amp
from tqdm import tqdm


for parent in Path.cwd().parents:
    sys.path.append(str(parent))

from auxiliar_dope.model import DopeNetwork
from auxiliar_dope.utils import MultipleVertexJson, save_image
import src.args_parser as ar
import src.custom_transform as ct
import src.auxiliar as aux
import src.run_network as rn
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

    """"train_dataset.test = True
    for i in range(len(trainingdata)):
        images = next(iter(trainingdata))

        save_image(images['image'],'{}/train_{}.png'.format( opt.outf,str(i).zfill(5)),mean=mean[0],std=std[0])
        print ("Saving batch %d" % i)
    train_dataset.test = False

    print ('things are saved in {}'.format(opt.outf))
    for i, output in enumerate(trainingdata):
        print(f"Sample {i}: Keypoints shape = {output['keypoints'].shape}, Centroids shape = {output['centroids'].shape}")
        if i == 5:  # Muestra solo las primeras 5 iteraciones
            break
    quit()"""


    net = DopeNetwork(pretrained=opt.pretrained)
    net = net.to(device)

    # Load the weights if a pretrained model is provided
    aux.load_dicts(opt, net,device)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters,lr=opt.lr)

    nb_update_network = 0

    scaler = amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    pbar = tqdm(range(1, opt.epochs + 1))    

    for epoch in pbar:
        # Run training and testing as before
        if trainingdata is not None:
            rn._runnetwork(epoch, trainingdata, train=True, pbar=pbar,
                        optimizer=optimizer, scaler=scaler, opt = opt,
                        net = net, device = device, nb_update_network = nb_update_network)

        if opt.datatest != "":
            rn._runnetwork(epoch, testingdata, train=False, pbar=pbar,
                        optimizer=optimizer, scaler=scaler, opt = opt,
                        net = net, device = device, nb_update_network = nb_update_network)
            if opt.data == "":
                break  # Exit if only testing
        try:
            torch.save(net.state_dict(), f'{opt.outf}/net_{opt.namefile}_{epoch}.pth')
        except Exception as e:
            print(f"Error saving model at epoch {epoch}: {e}")

        # Stop training if nb_update_network exceeds the limit
        if opt.nbupdates is not None and nb_update_network > int(opt.nbupdates):
            break

    print("end:", datetime.datetime.now().time())


if __name__ == "__main__":
    mp.freeze_support() # Required for Windows
    main()
