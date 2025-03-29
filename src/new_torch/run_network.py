import torch
import torch.optim as optim
from tqdm import tqdm

def calculate_loss(output, translations, rotations, has_object):
    """
    Calculate the loss for the network's output.

    Args:
        output (torch.Tensor or list): The network's output tensor or list.
        translations (torch.Tensor): Ground truth translations.
        rotations (torch.Tensor): Ground truth rotations.
        has_object (torch.Tensor): Ground truth object presence indicator.

    Returns:
        torch.Tensor: Computed loss value.
    """
    print("Type: ", type(output))
    print("len: ", len(output) if isinstance(output, list) else "N/A")
    
    # Ensure output is a tensor
    if isinstance(output, list):
        output = torch.cat(output, dim=0)  # Concatenate list elements into a tensor

    # Ensure each component of the output is a tensor
    if isinstance(output[0], torch.Tensor):
        object_output = output[0]
    elif isinstance(output[0], (list, tuple)):
        object_output = torch.tensor(output[0], device=has_object.device)
    else:
        raise ValueError(f"Unexpected type for output[0]: {type(output[0])}")
    
    print(f"Shape of output_belief: {[o.shape for o in output]}") if isinstance(output, list) else print(f"Shape of output: {output.shape}")
    print(f"Shape of translations: {translations.shape}")

    # Reshape or extract the correct part of the output for translations
    if isinstance(output, torch.Tensor):
        predicted_translations = output[1:4].T  # Transpose to match shape [8, 3]
    else:
        predicted_translations = torch.stack(output[1:4], dim=1)  # Stack along the second dimension
    print(f"Shape of predicted_translations: {predicted_translations.shape}")

    # Compute individual loss components
    object_loss = torch.nn.functional.mse_loss(object_output, has_object, reduction='mean')
    translation_loss = torch.nn.functional.mse_loss(predicted_translations, translations, reduction='mean')
    rotation_loss = torch.nn.functional.mse_loss(output[4:], rotations, reduction='mean')

    # Combine losses with equal weighting
    total_loss = (object_loss + translation_loss + rotation_loss) / 3.0

    return total_loss

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

def _runnetwork(epoch: int, loader, val_loader, 
                pbar: tqdm = None, opt = None, net = None, device = None, 
                optimizer: optim.Optimizer = None):  

    net.train()
    optimizer.zero_grad()

    for batch_idx, targets in enumerate(loader):        
        # Get the data and the target
        data = targets['image'].to(device, non_blocking=True)
        translations = targets['translations'].to(device, non_blocking = True)
        rotations = targets['rotations'].to(device, non_blocking = True)
        has_object = targets['has_points_belief'].to(device, non_blocking = True)


        with torch.autocast(device_type='cuda', dtype= torch.bfloat16):  # Use explicit bf16 precision
            output = net(data)
            loss = calculate_loss(output, translations, rotations, has_object)
        
        save_loss(epoch, batch_idx, loss.item(), opt, train=True)

        if not torch.isfinite(loss):  # Better NaN/Inf check
            print(f"Skipping batch {batch_idx} due to NaN/Inf loss")
            continue
                
        loss.backward()            

        if batch_idx % (opt.batchsize // opt.subbatchsize) == 0:                
            optimizer.step()                            
            optimizer.zero_grad()                             

        if pbar is not None:
            pbar.set_description(f"Training loss: {loss.item():0.4f} ({batch_idx}/{len(loader)})")

        if batch_idx % 100 == 0 and batch_idx > 0:
            save_model(net, opt,epoch, batch_idx)        

    optimizer.zero_grad()