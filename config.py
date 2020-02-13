#---Training setting---
epochs = 5
EGG_epochs = 20
batch_size = 128
learning_rate = 1e-3

CUDA=True
clip_norm = 5
num_class = 26 #알파벳 개수

#---Data setting---
data_path = "./az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data/A_Z Handwritten Data.csv"
EGG_data_path = "./EGG_data"
num_argu = 50
target_num = 25

