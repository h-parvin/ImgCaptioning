from data import Data
from settings import config
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
import gc
import multiprocessing

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
			nn.ReLU(),
            nn.Linear(512, 1)
			nn.ReLU()
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
		
		
if __name__ == "__main__":

	path_img = 'datasets/coco/image/'
	path_cp = 'datasets/coco/text/'

	data = Data(path_img,path_cp)
	model = Model(data)
	loss = Loss()
	optimizer = Optimizer(model)
	for epoch in range(config["num_epochs"]):
		print("[EPOCH]", epoch+1)

		# Process batches
		for caption, image_feature in data:
			pass			
			# Pass data through model
			embedded_caption, embedded_image_feature = model(caption, image_feature)
			# concatenation
			concatenation_vec = np.concatenate((embedded_caption, embedded_image_feature))
			mlp = MLP()
			score = mlp(concatenation_vec)
			cosine = np.dot(embedded_caption,embedded_image_feature)/(norm(embedded_caption, axis=1)*norm(embedded_image_feature))
			# Compute loss
			cost = loss(cosine, score)			
			optimizer.backprop(cost)
