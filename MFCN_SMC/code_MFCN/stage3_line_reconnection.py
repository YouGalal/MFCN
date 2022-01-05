# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 2020
@author: Yujin Oh (yujin.oh@kaist.ac.kr)
"""

import header

# common
import torch, torchvision
import numpy as np

# dataset
import mydataset
from torch.utils.data import DataLoader
from PIL import Image

# model
import torch.nn as nn

# post processing
import cv2
import correlation_code

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main():
    print("\nThird stage inference.py")

    ##############################################################################################################################
    # Semantic segmentation (inference)

    # Flag
    flag_eval_JI = True  # True #False # calculate JI
    flag_save_PNG = True  # preprocessR2U_Neted, mask
    connected_patch = True
    # GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_worker = header.num_worker
    else:
        device = torch.device("cpu")
        num_worker = 0

    # Model initialization
    net = header.net
    print(str(header.test_third_network_name), "inference")

    # network to GPU
    if torch.cuda.device_count() > 1:
        print("GPU COUNT = ", str(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
#FCDenseNet U_Net R2U_Net AttU_Net R2AttU_Net
    # Load model
    model_dir = header.dir_checkpoint + 'FCDenseNetmodel__1024_noise_RANZCR_100_1024v9.1.pth'

    print(model_dir)
    if os.path.isfile(model_dir):
        print('\n>> Load model - %s' % (model_dir))
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
        test_sampler = checkpoint['test_sampler']
        print("  >>> Epoch : %d" % (checkpoint['epoch']))
        # print("  >>> JI Best : %.3f" % (checkpoint['ji_best']))
    else:
        print('[Err] Model does not exist in %s' % (
                    header.dir_checkpoint + test_network_name + header.filename_model))
        exit()

    # network to GPU
    net.to(device)

    dir_third_test_path = header.dir_secon_data_root + 'output_inference_segmentation_endtoend' + test_network_name + header.Whole_Catheter + '/second_output_90/' + header.test_secon_network_name + '/data/input_Catheter_' + header.Whole_Catheter

    print('\n>> Load dataset -', dir_third_test_path)

    testset = mydataset.MyTestDataset(dir_third_test_path, test_sampler)
    testloader = DataLoader(testset, batch_size=header.num_batch_test, shuffle=False, num_workers=num_worker,
                            pin_memory=True)
    print("  >>> Total # of test sampler : %d" % (len(testset)))

    # inference
    print('\n\n>> Evaluate Network')
    with torch.no_grad():

        # initialize
        net.eval()

        for i, data in enumerate(testloader, 0):

            # forward
            outputs = net(data['input'].to(device))
            outputs = torch.argmax(outputs.detach(), dim=1)

            # one hot
            outputs_max = torch.stack(
                [mydataset.one_hot(outputs[k], header.num_masks) for k in range(len(data['input']))])

            # each case
            for k in range(len(data['input'])):

                # get size and case id
                original_size, dir_case_id, dir_results = mydataset.get_size_id(k, data['im_size'], data['ids'],
                                                                                header.net_label[1:])
                # '''
                # post processing
                post_output = [post_processing(outputs_max[k][j].numpy(), original_size) for j in
                               range(1, header.num_masks)]  # exclude background

                # original image processings
                save_dir = header.dir_save
                mydataset.create_folder(save_dir)
                image_original = testset.get_original(i * header.num_batch_test + k)

                # save mask/pre-processed image
                if flag_save_PNG:
                    save_dir = save_dir + '/output_inference_segmentation_endtoend' + str(
                        test_network_name) + header.Whole_Catheter + "/"+header.test_secon_network_name + 'third_output/' + str(
                        header.test_third_network_name) + '/Whole'

                    dir_case_id = dir_case_id.replace('/PICC', '')
                    dir_case_id = dir_case_id.replace('/Normal', '')
                    mydataset.create_folder(save_dir)
                    # '''
                    Image.fromarray(post_output[0] * 255).convert('L').save(save_dir + dir_case_id + '_mask.jpg')
                    Image.fromarray(image_original.astype('uint8')).convert('L').save(
                        save_dir + dir_case_id + '_image.jpg')


    if connected_patch:
        oup = save_dir.replace('/Whole',
                               '/First_connected_component')
        mydataset.create_folder(oup)
        connected_component(save_dir, oup)
        print("connected component process complete")

        make_correlation_check(oup)


def connected_component(inp, oup):
    import os.path
    import cv2
    import glob
    import numpy as np
    CAPTCHA_IMAGE_FOLDER = inp
    OUTPUT_FOLDER = oup

    # Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*_mask.jpg"))

    # loop over the image paths
    for (i, captcha_image_file) in enumerate(captcha_image_files):

        image = cv2.imread(captcha_image_file, cv2.IMREAD_UNCHANGED)

        image_zero = np.zeros(image.shape, dtype=image.dtype)

        image_zero[20:-20, 20:-20] = image[20:-20, 20:-20]

        image = image_zero

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        im = np.zeros((image.shape), np.uint8)
        check_value_list = []
        # getting mask with connectComponents
        ret, labels = cv2.connectedComponents(binary)
        for label in range(1, ret):
            checkvalue = np.array(labels, dtype=np.uint8)
            checkvalue[labels == label] = 255
            checkvalue = checkvalue.sum()
            check_value_list.append(checkvalue)
            check_value_list.sort()

        for label in range(1, ret):
            checkvalue = np.array(labels, dtype=np.uint8)
            checkvalue[labels == label] = 255
            checkvalue = checkvalue.sum()

            if check_value_list[-1] == checkvalue:
                    mask = np.array(labels, dtype=np.uint8)
                    mask[labels == label] = 255
                    mask[labels != label] = 0
                    im = im + mask

        filename = os.path.basename(captcha_image_file)
        p = os.path.join(OUTPUT_FOLDER, filename.replace('_mask', ''))

        cv2.imwrite(p, im)


def post_processing(raw_image, original_size, flag_pseudo=0):
    net_input_size = raw_image.shape
    raw_image = raw_image.astype('uint8')

    # resize
    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)
    else:
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)

    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, net_input_size, interpolation=cv2.INTER_NEAREST)

    return raw_image

def make_correlation_check(inp):
    import pandas as pd

    oup = inp.replace('/First_connected_component', '')

    for x in header.dir_mask_path:
        mask_path = header.dir_First_test_path.replace('/input_Catheter' + header.Data_path, x)

    image_name, dice, subin, point_rmse = correlation_code.correlation_Images(inp, mask_path, oup)

    df_image = pd.DataFrame([x for x in zip(image_name, dice, subin, point_rmse)],
                            columns=['image_name', 'dice', 'subin', 'point_rmse'])

    df_image.to_excel(oup + "third_image_rmse_jpg.xlsx", sheet_name='Sheet1')



if __name__ == '__main__':

    # test_network_name_list = ["FCDenseNet", "U_Net", "AttU_Net"]
    test_network_name_list = ["FCDenseNet"]

    for test_network_name in test_network_name_list:
        main()