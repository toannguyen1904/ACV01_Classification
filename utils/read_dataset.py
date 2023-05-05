import torch
from dataset import pre_data

def read_dataset(input_size, batch_size, data_root, train_folder_path, val_folder_path):
    trainset = pre_data.Dataset(input_size, data_root, train_folder_path, val_folder_path, mode='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=8, drop_last=False)

    valset = pre_data.Dataset(input_size, data_root, train_folder_path, val_folder_path, mode='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                  shuffle=False, num_workers=8, drop_last=False)

    return trainloader, valloader
