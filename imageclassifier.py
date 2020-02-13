# Python program to create 
# Image Classifier using CNN 

# Importing the required libraries 
import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm 
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 

import tensorflow as tf 
'''Setting up the env'''

TRAIN_DIR = "v_data\\train"
TEST_DIR = "v_data\\test"
IMG_SIZE = 100
LR = 1e-3

'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'smollan-{}-{}.model'.format(LR, '6conv-basic') 

def get_label_name(path):
    dirs = os.listdir(path)
    return np.identity(len(dirs),dtype=int), dirs

def create_train_data():
    i = 0
    training_data = []
    labels,dirs = get_label_name(TRAIN_DIR)
    images = []
    j = 0
    label_dirs = get_all_image_names(TRAIN_DIR)
    for _img in label_dirs:
        img = cv2.imread(_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if(_img.find("cars") != -1):
            training_data.append([np.array(img), np.array([1,0])])
        if(_img.find("planes") != -1):
            training_data.append([np.array(img), np.array([0,1])])
        # for label in dirs:
        #     if(_img.find(label) != -1):
        #         training_data.append([np.array(img), np.array(labels[j])])
        #         # print([np.array(img), np.array(labels[j])])
        #         print(j)    
    shuffle(training_data)
    print(training_data)
    np.save("training_data.npy",training_data)
    return training_data

def get_all_image_names(path):
    label_dirs = []
    images = []
    for root, dirs,files in os.walk(path):
        for name in files:
            # print(root)
            if(name.endswith(".jpg")):
                label_dirs.append(os.path.join(root,name))
    return label_dirs
def process_test_data():
    testing_data = [] 
    labels,dirs = get_label_name(TEST_DIR)
    label_dirs = get_all_image_names(TEST_DIR)
    for _img in label_dirs:
        img = cv2.imread(_img, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if(_img.find("cars") != -1):
            testing_data.append([np.array(img), np.array([1,0])])
        if(_img.find("planes") != -1):
            testing_data.append([np.array(img), np.array([0,1])])
    # print(testing_data)
    shuffle(testing_data)
    np.save("test_data.npy",testing_data)
    return testing_data
train_data = create_train_data()
test_data = process_test_data() 
# print(train_data)
# exit()
# def create_model():
tf.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 

convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 

convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.8) 

convnet = fully_connected(convnet, 2, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
    loss ='categorical_crossentropy', name ='targets') 

    # return convnet 

model = tflearn.DNN(convnet, tensorboard_dir ='log')

# if os.path.exists('D:/PythonProjects/ProjectFiles/HUL/{}.meta'.format(MODEL_NAME)):
#     model.load(MODEL_NAME)
#     print('model loaded!')
# print('model  Not loaded!')

train = train_data[:-50] 
test = train_data[-50:] 
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = [i[1] for i in train] 
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
test_y = [i[1] for i in test] 

'''Fitting the data into our model'''
# epoch = 5 taken 
model.fit({'input': X}, {'targets': Y}, n_epoch = 10, 
	validation_set =({'input': test_x}, {'targets': test_y}), 
	snapshot_step = 500, show_metric = True, run_id = MODEL_NAME) 
model.save(MODEL_NAME) 

import matplotlib.pyplot as plt 
# if you need to create the data: 
# test_data = process_test_data() 
# if you already have some saved: 
test_data = np.load('test_data.npy',allow_pickle=True) 
print(test_data.shape)
fig = plt.figure() 

for num, data in enumerate(test_data[:20]):
    img_num = data[1] 
    img_data = data[0] 
    y = fig.add_subplot(4, 5, num + 1) 
    orig = img_data 
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1),
    # model_out = model.predict([data])[0] 
    model_out = model.predict(data)
    print(model_out)
    if np.argmax(model_out) == 1: str_label ='planes'
    else: str_label ='cars'
    y.imshow(orig, cmap ='gray') 
    plt.title(str_label) 
    y.axes.get_xaxis().set_visible(False) 
    y.axes.get_yaxis().set_visible(False) 
plt.show() 
