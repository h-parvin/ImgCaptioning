from data import Data
from model import Model
from data import Data
from settings import config
from model import Model
from loss import PairwiseRankingLoss as Loss
from optimizer import Optimizer
import numpy as np
from numpy.linalg import norm

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

	path_img = 'datasets/coco/image'
	path_cp = 'datasets/coco/text'

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
			print(score)