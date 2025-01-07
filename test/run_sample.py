from utils.function import DDPM
from utils.function import ContextUnet

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def main():
    n_epoch = 20
    batch_size = 256
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './data/diffusion_outputs10/'
    ws_test = [0.0, 0.5, 1.0, 2.0] # strength of generative guidance
    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    # optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    model = ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes)
    weight = torch.load('/home/evan.chen/Conditional_Diffusion_MNIST/test/weight/model_39.pth')
    ddpm = DDPM(nn_model=model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    # print(weight.keys())
    ddpm.load_state_dict(weight)
    ddpm.to(device)
    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    # ws_test = [2]
    ddpm.eval()
    with torch.no_grad():
        for w_i, w in enumerate(tqdm(ws_test)):
            if Path(f"./gen_data/guide_{w:.2f}/").exists() == False:
                    Path(f"./gen_data/guide_{w:.2f}/").mkdir(parents = True)
        for i in range(0, 15):
            n_sample = 1000
            for w_i, w in enumerate(tqdm(ws_test)):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)
                pd.to_pickle(x_gen_store, f"./gen_data/guide_{w:.2f}/guide_{w:.2f}_{i}.pkl")

        # print(x_gen)

if __name__ == "__main__":
    main()