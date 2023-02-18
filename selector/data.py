import numpy
import copy
from collections import OrderedDict
from settings import config
import torch
from torch.autograd import Variable
import pickle
from random import shuffle

def Data(path_img, path_cp):

	images = {}
	captions = {}
	file = open(path_cp,'r').read()

	for img in os.listdir(path_img):
	  image = os.path.join(path_img,img)
	  image = cv2.imread(image)
	  image = cv2.resize(image, (224,224))
	  images[img] = list()
	  images[img].append(image)
	  for line in file.split('\n'):
		row = line.split(',')     
		id, caption = row[0], row[1:]
		if id == img:
		  if id not in captions:
			captions[id]=list()
		  captions[id].append(caption)
	  break
	
	from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201
	from tensorflow.keras import Model

	def FeatureExtraction(image):
	  base_model = DenseNet201(weights="imagenet")
	  model = Model(inputs = base_model.input, outputs = base_model.layers[-2].output)
	  image = image.reshape(1,image.shape[0], image.shape[1], image.shape[2])
	  feature_vec = model.predict(image)
	  return feature_vec[0]


	feature_vec = {}
	for name, arr in images.items():
		img = np.array(arr[0])
		fv = FeatureExtraction(img)
		feature_vec[name] = list()
		feature_vec[name].append(fv) 
	
	import gensim
	from gensim.models import word2vec

	clean_caption = {}

	for name, cp in captions.items():
	  for sentences in cp:
		for sentence in sentences:
		  tokens = sentence.split()
		  tokens = [token.lower() for token in tokens]
		  tokens = [token for token in tokens if token.isalpha()]
		  tokens = [token.replace('[^A-Za-z]', '') for token in tokens] # delete special characters, digits, etc.
		  tokens = [token.replace('\s+', ' ') for token in tokens] # delete additional spaces
		  tokens = 'startseq ' + ' '.join(tokens) + ' endseq'
		  if name not in clean_caption:
			clean_caption[name]=list()
		  clean_caption[name].append(tokens)
	txt_feature =[]
	for cap in clean_caption:
		t = Word2Vec([cap], min_count=1)
		txt_feature.append(t)
	
	return feature_vec, txt_feature