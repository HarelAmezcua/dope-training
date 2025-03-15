import torch
import torch.optim as optim
from tqdm import tqdm


def calculate_loss(output_belief, target_belief, output_affinities, target_affinity):
    loss = sum(((l - target_belief) ** 2).mean() for l in output_belief) # sum of the mean squared error of each belief map
    loss += sum(((l - target_affinity) ** 2).mean() for l in output_affinities)
    print("Exiting calculate_loss")
    return loss

def save_loss(epoch, batch_idx, loss, opt, train):
    namefile = '/loss_train.csv' if train else '/loss_test.csv'
    with open(opt.outf + namefile, 'a') as file:
        s = '{}, {},{:.15f}\n'.format(epoch, batch_idx, loss.data.item())
        file.write(s)

def save_model(net, opt):
    try:
        torch.save(net.state_dict(), f'{opt.outf}/net.pth')
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

def _runnetwork(epoch: int, loader, train: bool = True, 
                pbar: tqdm = None, opt = None, net = None, device = None, 
                optimizer: optim.Optimizer = None, nb_update_network: int = 0):  

    net.train() if train else net.eval()
    if train:
        optimizer.zero_grad()

    if nb_update_network is None:
        nb_update_network = 0  # Ensure initialization

    for batch_idx, targets in enumerate(loader):
        print(f"Processing batch {batch_idx}")

        data = targets['image'].to(device, non_blocking=True)
        target_belief = targets['beliefs'].to(device, non_blocking=True)
        target_affinity = targets['affinities'].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype= torch.bfloat16):  # Use explicit import
            output_belief, output_affinities = net(data)
            loss = calculate_loss(output_belief, target_belief, output_affinities, target_affinity)

        if not torch.isfinite(loss):  # Better NaN/Inf check
            print(f"Skipping batch {batch_idx} due to NaN/Inf loss")
            continue
        
        print("Hola")
        if train:
            loss.backward()
            print("Hola2")

            if batch_idx % (opt.batchsize // opt.subbatchsize) == 0:
                print("Hola3")
                optimizer.step()   
                print("Hola4")            
                optimizer.zero_grad()
                print("Hola5")
                nb_update_network += 1

        #save_loss(epoch, batch_idx, loss, opt, train)

        #if opt.nbupdates and nb_update_network > int(opt.nbupdates):  # Ensure no crash if None
            #torch.save(net.state_dict(), f'{opt.outf}/net_{opt.namefile}.pth')
            #break

        if pbar is not None:
            pbar.set_description(f"{'Training' if train else 'Testing'} loss: {loss.item():0.4f} ({batch_idx}/{len(loader)})")

        #if batch_idx % 10 == 0:
            #save_model(net, opt)

        print("Finishing batch")

    if train:
        optimizer.zero_grad()