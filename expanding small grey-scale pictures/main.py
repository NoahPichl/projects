# -*- coding: utf-8 -*-
"""example_project/main.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.02.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Main file of example project.
"""

import os
import numpy as np
import torch
import torch.utils.data
from datasets import ImageData, custom_stacking, ImageNormalizer
from utils import plot_pil, plot
from architectures import CNN
from torch.utils.tensorboard import SummaryWriter
import tqdm
import pickle


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets = data
            masks = inputs[:, 1, :, :]
            minimal = float(inputs[:,0,:,:].min().data)
            maximal = float(inputs[:,0,:,:].max().data)

            inputs[:,0,:,:] = (inputs[:,0,:,:] - mean) / std
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get outputs for network
            outputs = model(inputs)
            outputs = outputs * std + mean

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance
            outputs = torch.clamp(outputs, minimal, maximal)

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            l = np.zeros((len(targets)))
            for output, target, mask, i in zip(outputs, targets, masks, range(0, len(targets))):
                mask = ~np.array(mask, dtype=bool)
                l[i] = mse(output[0][mask], target[0][mask])
            loss += l.sum()/len(dataloader.dataset)

    return loss

def main(data_path, results_path, network_config: dict, batchsize: int = 40, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # set seed
    np.random.seed(7)
    global mean, std

    # Prepare a path to plot to
    plotpath = os.path.join(results_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)
    
    # Prepare Images and create dataset
    image_dataset = ImageData(data_path)

    # Create inds for the subsets
    shuffled_indices = np.random.permutation(len(image_dataset))
    validationset_inds = shuffled_indices[:int(len(image_dataset) / 5)]
    trainingset_inds = shuffled_indices[int(len(image_dataset) / 5):]
    np.savez(os.path.join(os.getcwd(), "data", "data_split"), validationset_inds, trainingset_inds)

    # Split dataset into training and validation set
    trainingset = torch.utils.data.Subset(image_dataset, indices=trainingset_inds)
    validationset = torch.utils.data.Subset(image_dataset, indices=validationset_inds)

    # Create DataLoader
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=batchsize, shuffle=False, num_workers=12,
                                              collate_fn=custom_stacking)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=batchsize, shuffle=False, num_workers=12,
                                            collate_fn=custom_stacking)

    # Compute mean and std
    if os.path.isfile("data/norm.p"):
        mean, std = pickle.load(open("data/norm.p", "rb"))
    else:
        image_normalizer = ImageNormalizer(trainloader)
        mean, std = image_normalizer.analyze_images()
        pickle.dump((mean, std), open("data/norm.p", "wb"))

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard/test/9'))
    
    # Create Network
    net = CNN(**network_config)
    net.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer & learning rate scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 399, 0.7)

    print_stats_at = 1e2  # print status to tensorboard every x updates
    plot_at = 1e4  # plot every x updates
    validate_at = 249  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    
    # Train until n_updates update have been reached
    while update < n_updates:
        for data in trainloader:
            # Get next samples in `trainloader_augmented`
            inputs, targets = data      # batch is of form [input, targets]
            inputs[:,0,:,:] = (inputs[:,0,:,:] - mean) / std
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset gradients
            optimizer.zero_grad()
            
            # Get outputs for network
            outputs = net(inputs)
            outputs = outputs * std + mean
            
            # Calculate loss, do backward pass, and update weights
            mask = ~torch.tensor(inputs[:,1:2,:,:], dtype=bool)
            loss = mse(outputs[mask], targets[mask])
            loss.backward()
            optimizer.step()
            #scheduler.step()
            
            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)
                #writer.add_scalar(tag="training/lr",
                                  # scalar_value=scheduler.get_last_lr()[-1],
                                  # global_step=update)
            
            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valloader, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))
            
            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()
            
            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break

    update_progess_bar.close()
    print('Finished Training!')
    # plot pictures
    pictures = outputs.detach().cpu()
    for x, pic in enumerate(pictures.data):
        plot_pil(pic.numpy()[0], str(x))
    
    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader, device=device)
    
    print(f"Scores:")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")
    
    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)


# if __name__ == '__main__':
#import argparse
import json

#parser = argparse.ArgumentParser()
#parser.add_argument('config_file', help='path to config file', type=str)
#args = parser.parse_args()
#config_file = args.config_file
config_file = "working_config.json"

with open(config_file, 'r') as fh:
    config = json.load(fh)
main(**config)

