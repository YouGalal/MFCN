# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 2021
@author: Subin Park (subinn.park@gmail.com)
"""
import header

# common
import torch
import numpy as np

# dataset
import mydataset
from torch.utils.data import DataLoader
from PIL import Image

# model
import torch.nn as nn

# post processing
import correlation_code
import cv2

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

Normal_save_param = False

def main():
    print("\nFirst stage inference.py")

    ##############################################################################################################################
    # Semantic segmentation (inference)

    # Flag
    flag_eval_JI = True  # True #False # calculate JI
    flag_save_PNG = True  # preprocessR2U_Neted, mask
    connected_patch = True
    softmax_thresh_value = 0.01



    # GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_worker = header.num_worker
    else:
        device = torch.device("cpu")
        num_worker = 0

    # Model initialization
    net = header.First_net_model[network_number]
    print(str(header.First_net_string[network_number]), "inference")

    # network to GPU
    if torch.cuda.device_count() > 1:
        print("GPU COUNT = ", str(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
#FCDenseNet U_Net R2U_Net AttU_Net R2AttU_Net
    # Load model
    model_dir = header.dir_checkpoint + str(header.First_net_string[network_number]) + 'model__RANZCR_100_Whole_100_1024v9.1.pth' # -> whole png

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
                    header.dir_checkpoint + str(header.First_net_string[network_number]) + header.filename_model))
        exit()

    # network to GPU
    net.to(device)


    # Dataset
    print('\n>> Load dataset -', header.dir_data_root)

    testset = mydataset.MyTestDataset(header.dir_First_test_path, test_sampler)

    testloader = DataLoader(testset, batch_size=header.num_batch_test, shuffle=False, num_workers=num_worker, pin_memory=True)
    print("  >>> Total # of test sampler : %d" % (len(testset)))

    # inference
    print('\n\n>> Evaluate Network')
    with torch.no_grad():

        # initialize
        net.eval()
        ji_test = []

        for i, data in enumerate(testloader, 0):

            # forward
            outputs = net(data['input'].to(device))
            snet = nn.Softmax(dim=1)
            soutputs = snet(outputs)

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

                post_output_mask = [post_processing(data['masks'][k][j].numpy(), original_size) for j in
                                    range(0, header.num_masks - 1)]  # exclude background

                # jaccard index (JI)
                if flag_eval_JI:
                    ji = tuple(get_JI(post_output[j], post_output_mask[j]) for j in range(len(post_output)))
                    ji_test.append(ji)
                # '''

                # original image processings
                save_dir = header.dir_save
                mydataset.create_folder(save_dir)
                image_original = testset.get_original(i * header.num_batch_test + k)

                # save mask/pre-processed image
                if flag_save_PNG:
                    save_dir = save_dir+'/output_inference_segmentation_endtoend' + str(
                        header.First_net_string[network_number]) + header.Whole_Catheter + "/" + 'First_output/Whole'


                    dir_case_id = dir_case_id.replace('/PICC', '')
                    dir_case_id = dir_case_id.replace('/Normal', '')
                    mydataset.create_folder(save_dir)
                    # '''
                    Image.fromarray(post_output[0] * 255).convert('L').save(save_dir + dir_case_id + '_mask.jpg')
                    Image.fromarray(image_original.astype('uint8')).convert('L').save(
                        save_dir + dir_case_id + '_image.jpg')

                    save_dir_th = save_dir.replace('/Whole',
                                           '/thresh')
                    mydataset.create_folder(save_dir_th)


                    save_file_th = np.where(soutputs[k][1].cpu() > softmax_thresh_value, 1., 0.)

                    save_file_th = cv2.resize(save_file_th, original_size, interpolation=cv2.INTER_NEAREST)

                    Image.fromarray(save_file_th * 255).convert('L').save(
                        save_dir_th + dir_case_id + '_thresh.jpg')


    if connected_patch:
        inp = header.original_dir
        oup = save_dir.replace('/Whole',
                               '/First_connected_component')
        mydataset.create_folder(oup)

        ouput2 = save_dir.replace('/Whole',
                                  '/random_crop')
        mydataset.create_folder(ouput2)


        make_correlation_check(save_dir)
        print("correlation complete")

        connected_component(save_dir_th, oup)
        print("connected component process complete")
        connedted_component_crop(inp, oup, ouput2)
        print("Crop process complete")






def connected_component(inp, oup):
    import os.path
    import cv2
    import glob
    import numpy as np
    CAPTCHA_IMAGE_FOLDER = inp
    OUTPUT_FOLDER = oup

    # Get a list of all the captcha images we need to process
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*_thresh.jpg"))

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

            if len(check_value_list) < 3:
                for i in range(0, len(check_value_list)):
                    if check_value_list[-(i + 1)] == checkvalue:
                        mask = np.array(labels, dtype=np.uint8)
                        mask[labels == label] = 255
                        mask[labels != label] = 0
                        im = im + mask
            else:
                if check_value_list[-1] == checkvalue:
                    mask = np.array(labels, dtype=np.uint8)
                    mask[labels == label] = 255
                    mask[labels != label] = 0
                    im = im + mask

                if check_value_list[-2] == checkvalue:
                    mask = np.array(labels, dtype=np.uint8)
                    mask[labels == label] = 255
                    mask[labels != label] = 0
                    im = im + mask

                if check_value_list[-3] == checkvalue:
                    mask = np.array(labels, dtype=np.uint8)
                    mask[labels == label] = 255
                    mask[labels != label] = 0
                    im = im + mask

        filename = os.path.basename(captcha_image_file)
        p = os.path.join(OUTPUT_FOLDER, filename.replace('_thresh', ''))

        cv2.imwrite(p, im)


def connedted_component_crop(input1, input2, ouput1):
    # connected component 이후 crop
    from PIL import Image
    import os
    import numpy as np

    Crop_img_size = 512
    data_number = 200

    inp = input1 + '/'
    inp2 = input2 + '/'
    oup = ouput1 + "/data/input_Catheter_"+ header.Whole_Catheter + '/'
    oup2 = ouput1 + "/data/mask_Catheter_" + header.Whole_Catheter + '/'

    filenames = os.listdir(inp)
    # filenames = [x for x in os.listdir(inp) if '_mask' in x]

    os.makedirs(oup + 'PICC/', exist_ok=True)
    os.makedirs(oup2 + 'PICC/', exist_ok=True)

    for files in filenames:
        file_form = files.split('.')
        im = Image.open(os.path.join(inp, files))
        im2 = Image.open(os.path.join(inp2, files))

        # random parameter
        width, height = im.size
        files = files.split('.')

        PICC_count_num = 0
        normal_count_num = 0

        while PICC_count_num < data_number:

            top = np.random.randint(0, height - Crop_img_size)
            left = np.random.randint(0, width - Crop_img_size)

            bottom = top + Crop_img_size
            right = left + Crop_img_size
            X_patch = im.crop((left, top, right, bottom))
            X_label = im2.crop((left, top, right, bottom))

            if np.sum(X_label) > 100:  # 1은 카테터존
                if PICC_count_num < data_number:
                    X_patch.save(os.path.join(oup + '/PICC/',
                                              files[0] + '_' + str(PICC_count_num + normal_count_num) + '_' + str(
                                                  left) + '_' + str(top) + '_' + str(right) + '_' + str(
                                                  bottom) + '.jpg'))
                    X_label.save(os.path.join(oup2 + '/PICC/',
                                              files[0] + '_' + str(PICC_count_num + normal_count_num) + '_' + str(
                                                  left) + '_' + str(top) + '_' + str(right) + '_' + str(
                                                  bottom) + '.jpg'))
                    PICC_count_num = PICC_count_num + 1


def get_JI(pred_m, gt_m):
    intersection = np.logical_and(gt_m, pred_m)

    true_sum = gt_m[:, :].sum()
    pred_sum = pred_m[:, :].sum()
    intersection_sum = intersection[:, :].sum()

    ji = (intersection_sum + 1.) / (true_sum + pred_sum - intersection_sum + 1.)

    return ji


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

    oup = inp.replace(('/'+inp.split('/')[-1]), '')

    for x in header.dir_mask_path:
        mask_path = header.dir_First_test_path.replace('/input_Catheter' + header.Data_path, x)

    image_name, dice, subin, point_rmse = correlation_code.correlation_Firt_Images(inp, mask_path, oup)

    df_image = pd.DataFrame([x for x in zip(image_name, dice, subin, point_rmse)],
                            columns=['image_name', 'dice', 'subin', 'point_rmse'])

    df_image.to_excel(oup + "/first_image_rmse_jpg.xlsx", sheet_name='Sheet1')


if __name__ == '__main__':
    for network_number in range(0, len(header.First_net_model)):
        main()