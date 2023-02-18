config = {
	"num_epochs":100,
	"batch_size":100,
	"word_dimension":512, # The dimensionality of word embeddings
	"image_dimension":4096, # input image dimension : 4096 
	"model_dimension":1000, # The dimension of the embedding space,
	"learning_rate":0.01,
	"display_freq":50, h etc...
	"margin_pairwise_ranking_loss":0.2, # Should be between zero and 1,
	"dataset":"./datasets", 
	"cuda":True # enable cuda
}