import os
import argparse
import torch
from auxiliar_dope.utils import MultipleVertexJson
from collections import OrderedDict

def create_output_folder(opt): 
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # save the hyper parameters passed
    with open (opt.outf+'/header.txt','w', encoding='utf-8') as file:
        file.write(str(opt)+"\n")

    with open (opt.outf+'/header.txt','w', encoding='utf-8') as file:
        file.write(str(opt))
        file.write("seed: "+ str(opt.manualseed)+'\n')
        with open (opt.outf+'/test_metric.csv','w', encoding='utf-8') as file:
            file.write("epoch, passed,total \n")

    with open (opt.outf+'/loss_train.csv','w', encoding='utf-8') as file:
        file.write('epoch,batchid,loss\n')

    with open (opt.outf+'/loss_val.csv','w', encoding='utf-8') as file:
        file.write('id,batchid,loss\n')        


def get_DataLoaders(opt, preprocessing_transform, transform):
    #load the dataset using the loader in utils_pose
    trainingdata = None
    if not opt.data == "":
        train_dataset = MultipleVertexJson(
            root = opt.data,
            preprocessing_transform=preprocessing_transform,
            objectsofinterest=opt.object,
            sigma = opt.sigma,
            data_size = opt.datasize,
            save = opt.save,
            transform = transform,
        )

        print(f"Length of train_dataset: {len(train_dataset)}")

        trainingdata = torch.utils.data.DataLoader(train_dataset,
            batch_size = opt.subbatchsize,
            shuffle = True,
            num_workers = opt.workers,
            pin_memory = True,
            drop_last=True
        )

        if opt.save:
            print ('things are saved in {}'.format(opt.outf))
            quit()


    testingdata = None
    if not opt.datatest == "":
        test_dataset = MultipleVertexJson(
                root = opt.datatest,
                preprocessing_transform=preprocessing_transform,
                objectsofinterest=opt.object,
                sigma = opt.sigma,
                data_size = opt.datasize,
                save = opt.save,
                test = True
                )
        
        print(f"Length of test_dataset: {len(test_dataset)}")

        testingdata = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = opt.subbatchsize,
            shuffle = True,
            num_workers = opt.workers,
            pin_memory = True,
            drop_last=True)
        
    if not trainingdata is None:
        print('training data: {} batches'.format(len(trainingdata)))
    if not testingdata is None:
        print ("testing data: {} batches".format(len(testingdata)))
    return train_dataset, test_dataset, trainingdata, testingdata

def load_dicts(opt, net,device):
    if opt.net != '':
        # Load state dict from file
        state_dict = torch.load(opt.net, map_location=device)

        # If the state dict keys start with "module.", remove that prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            new_state_dict[new_key] = v

        # Use the new_state_dict directly
        net.load_state_dict(new_state_dict)