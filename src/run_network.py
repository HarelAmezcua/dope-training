import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from torch.cuda import amp


def calculate_loss(output_belief, target_belief, output_affinities, target_affinity):
    loss = sum(((l - target_belief) ** 2).mean() for l in output_belief) # sum of the mean squared error of each belief map
    loss += sum(((l - target_affinity) ** 2).mean() for l in output_affinities)
    #print("Exiting calculate_loss")
    return loss

def save_loss(epoch, batch_idx, loss, opt, train):
    namefile = '/loss_train.csv' if train else '/loss_val.csv'
    with open(opt.outf + namefile, 'a') as file:
        s = '{}, {},{:.15f}\n'.format(epoch, batch_idx, loss)  # Removed .data.item()
        file.write(s)

def save_model(net, opt, epoch, batch_idx):
    try:
        torch.save(net.state_dict(), f'{opt.outf}/net_{epoch}_{batch_idx}.pth')
        print("Model saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")

def _runnetwork(epoch: int, loader, val_loader, scaler: GradScaler = None, 
                pbar: tqdm = None, opt = None, net = None, device = None, 
                optimizer: optim.Optimizer = None):  

    net.train()    
    optimizer.zero_grad()

    for batch_idx, targets in enumerate(loader):
        #print(f"Processing batch {batch_idx}")

        data = Variable(targets['image'].to(device, non_blocking=True).float())
        translations = targets['translations'].to(device, non_blocking=True).float()
        rotations = targets['rotations'].to(device, non_blocking=True).float()
        has_object = targets['has_points_belief'].to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():  # Use explicit import
            output_belief, output_affinities = net(data)
            loss = calculate_loss(output_belief, target_belief, output_affinities, target_affinity)

        save_loss(epoch, batch_idx, loss.item(), opt, train=True)

        if not torch.isfinite(loss):  # Better NaN/Inf check
            print(f"Skipping batch {batch_idx} due to NaN/Inf loss")
            continue

        scaler.scale(loss).backward()

        if batch_idx % (opt.batchsize // opt.subbatchsize) == 0:
            scaler.unscale_(optimizer)  # Prevent NaN gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()            
            optimizer.zero_grad()

        if pbar is not None:
            pbar.set_description(f"Training loss: {loss.item():0.4f} ({batch_idx}/{len(loader)})")
        
        # Every 10 batches, compute validation loss and save it
        if batch_idx % 10 == 0:
            net.eval()
            with torch.no_grad():
                val_loss = 0.0
                for val_targets in val_loader:
                    val_data = val_targets['image'].to(device, non_blocking=True)
                    val_target_belief = val_targets['beliefs'].to(device, non_blocking=True)
                    val_target_affinity = val_targets['affinities'].to(device, non_blocking=True)

                    val_output_belief, val_output_affinities = net(val_data)
                    val_loss += calculate_loss(val_output_belief, val_target_belief, val_output_affinities, val_target_affinity).item()

                val_loss /= len(val_loader)
                save_loss(epoch, batch_idx, val_loss, opt, train=False)
                #print(f"Validation loss after batch {batch_idx}: {val_loss:.4f}")
            net.train()
        if batch_idx % 500 == 0 and batch_idx > 0:
            try:
                torch.save(net.state_dict(), f'{opt.outf}/net_batch{opt.namefile}_{batch_idx}.pth')
                print("Model saved successfully")
            except Exception as e:
                print(f"Error saving model at epoch {epoch}: {e}")
    
    optimizer.zero_grad()