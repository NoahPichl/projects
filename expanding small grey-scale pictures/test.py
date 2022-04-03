import os
import numpy as np
import torch
import torch.utils.data
from datasets import ImageData, custom_stacking, ImageNormalizer
from utils import plot_pil, plot
import tqdm
import pickle
from datasets import ImageDataTest


def model_predicitons(model_path, data_path):
    np.random.seed(7)
    device = "cuda"
    model = torch.load(model_path)
    batch_size = 200

    try:
        mean, std = pickle.load(open("data/norm.p", "rb"))
    except FileNotFoundError:
        raise(FileNotFoundError("No file for mean and std found"))
    # create dataset
    testset = ImageDataTest(data_path)
    # Create DataLoader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1,
                                             collate_fn=custom_stacking)
    predicitons = []
    outputs_list = []
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(testloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets = data
            minimal = float(inputs[:, 0, :, :].min().data)
            maximal = float(inputs[:, 0, :, :].max().data)

            inputs[:, 0, :, :] = (inputs[:, 0, :, :] - mean) / std
            inputs = inputs.to(device)

            # Get outputs for network
            outputs = model(inputs)
            outputs = outputs * std + mean

            # clamp the outputs to the minimum and maximum values of inputs for better performance
            outputs = torch.clamp(outputs, minimal, maximal)
            outputs_list.append(outputs)

    data = pickle.load(open(data_path, "rb"))
    for x, batch in enumerate(outputs_list):
        for y, pic in enumerate(batch):
            idx = x * batch_size + y
            plot_pil(pic.data.cpu()[0,:,:].numpy(), "batch"+str(x)+"_pic"+str(y))
            image_array = pic.data.cpu().numpy().astype(dtype="uint8")[0,:,:]
            known_array = np.zeros_like(image_array)
            known_array[data["borders_x"][idx][0]:-data["borders_x"][idx][1], data["borders_y"][idx][0]:-data["borders_y"][idx][1]] = 1
            predicitons.append(image_array[~np.array(known_array, dtype=bool)])

    pickle.dump(predicitons, file=open("data/testset/predicitonsV3.pkl", "wb"))


model_predicitons("results/t6.pt", "data/testset/testset.pkl")
