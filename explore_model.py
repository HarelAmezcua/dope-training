import torch


# Load the .pth file
pth_file_path = "net_batchweights_ketchup_500.pth"  # Replace with the actual path to your .pth file
loaded_weights = torch.load(pth_file_path)


# Calculate the total number of parameters
total_parameters = sum(param.numel() for param in loaded_weights.values())
print(f"Total number of parameters: {total_parameters}")