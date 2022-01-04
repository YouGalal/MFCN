# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 2021
@author: Subin Park (subinn.park@gmail.com)
"""

import model


# Model
tag = 'v10.2'
#v6.2, v3.4, v8.0

loss_plot_filename_model = 'model_loss' + tag + '.png'
dataset = 0
division_trainset = 1
threshold_partial = 1
partial_dataset = False
gamma = 0.5
num_channel = 1
ratio_dropout = 0.2
weight_bk = 0.5

# input image folder name
Whole_Catheter = "_Whole_RANZCR"
Data_path = "_Whole_RANZCR"

# Directory
dir_data_root = "../model_input/"
dir_secon_data_root = "../output/"

dir_train_path = dir_data_root + "input_Catheter"+Whole_Catheter+"/train"
dir_valid_path = dir_data_root + "input_Catheter"+Whole_Catheter+"/validation"
dir_inference_path = dir_data_root + "input_Catheter"+Whole_Catheter+"/test"


dir_mask_path = ["/mask_Catheter"+Data_path]

dir_checkpoint = '../checkpoint/'
dir_save = "../output/"

original_dir = dir_data_root + "input_Catheter"+Data_path+"/test/PICC"
#test_normal_150

# Network
num_masks = 2
num_network = 1

test_secon_network_name = "FCDenseNet"
test_third_network_name = "FCDenseNet"
net = model.FCDenseNet(num_channel, num_masks)
#net = model.FCDenseNet(num_channel, num_masks, ratio_dropout)
net_label = ['BG', 'Catheter']

dir_First_test_path = dir_data_root + "input_Catheter"+Data_path+"/test"


net1 = model.FCDenseNet(num_channel, num_masks)
net2 = model.U_Net(num_channel, num_masks)
net4 = model.AttU_Net(num_channel, num_masks)

#first stage test network
#net_model = [net1, net2, net4]
First_net_model = [net1]
#net_model = [net1]
#net_string = ["FCDenseNet", "U_Net", "AttU_Net"]
First_net_string = ["FCDenseNet"]

# Dataset
#orig_height = 2048
#orig_width = 2048

resize_height = 512
resize_width = 512

rescale_bit = 8

# CPU
num_worker = 2

# Train schedule
num_batch_train = 1
epoch_max = 50
threshold_epoch = 100
learning_rate = 1e-4

# Validation schedule
train_split = 0.8
valid_split = 0.92

# Test schedule
num_batch_test = 8

filename_model = 'model_' + Whole_Catheter + '_'+ str(epoch_max)+ '_' + str(resize_height) + tag + '.pth'