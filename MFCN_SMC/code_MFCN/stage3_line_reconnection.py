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
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from tqdm import tqdm
import networkx as nx

# model
import torch.nn as nn

# post processing
import cv2
import correlation_code
import glob

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

    # Load model
    model_dir = header.dir_checkpoint + 'FCDenseNet_Stage3_epoch41_tip.pth'

    print(model_dir)
    if os.path.isfile(model_dir):
        print('\n>> Load model - %s' % (model_dir))
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint['model_state_dict'])
        test_sampler = None
        print("  >>> Epoch : %d" % (checkpoint['epoch']))
        # print("  >>> JI Best : %.3f" % (checkpoint['ji_best']))
    else:
        print('[Err] Model does not exist in %s' % (
                    header.dir_checkpoint + test_network_name + header.filename_model))
        exit()

    # network to GPU
    net.to(device)

    dir_third_test_path = header.dir_secon_data_root + 'output_inference_segmentation_endtoend' + test_network_name + header.Whole_Catheter + '/second_output_90/' + header.test_secon_network_name + '/data/input_Catheter_' + header.Whole_Catheter

    def filter_short_structures_in_folder(input_folder, output_folder, min_area=2500):
        print(f"\n>> Pre-filtering input masks from {input_folder} to {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

        # Search recursively for common image formats
        extensions = ('*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tif')
        image_files = []
        for ext in extensions:
            # ** Add recursive search **
            image_files.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))

        print(f"Found {len(image_files)} image(s) to filter.")

        for img_path in tqdm(image_files, desc="Filtering input images"):
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Binarize
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Connected components filter
            num_labels, labels = cv2.connectedComponents(binary)
            filtered = np.zeros_like(binary, dtype=np.uint8)
            for label in range(1, num_labels):
                if np.sum(labels == label) >= min_area:
                    filtered[labels == label] = 255

            # Save to output folder preserving relative path
            rel_path = os.path.relpath(img_path, input_folder)
            out_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, filtered)

    def bridge_small_gaps(mask):
        # detect catheter tip
        tip_point = correlation_code.bounding_box_fuc(mask.copy())

        # make a protection circle around tip (so closing won't touch it)
        protected = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(protected, tip_point, 25, 1, -1)  # radius = 25 px (tune this)

        # do morphological closing to fill tiny gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # apply closing only outside tip region
        mask[protected == 0] = closed[protected == 0]
        return mask

    filtered_dir = os.path.join(header.dir_secon_data_root, "filtered_inputs")
    filter_short_structures_in_folder(dir_third_test_path, filtered_dir, min_area=2500)

    print('\n>> Load dataset -', filtered_dir)
    testset = mydataset.MyTestDataset(filtered_dir, test_sampler)
    testloader = DataLoader(testset, batch_size=header.num_batch_test, shuffle=False, num_workers=num_worker, pin_memory=True)
    print("  >>> Total # of test sampler : %d" % len(testset))

    # inference
    print('\n\n>> Evaluate Network')
    with torch.no_grad():

        # initialize
        net.train()

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
                # dir_case_id = dir_case_id.replace('\\', '/') ### if on windows system uncomment this line
                # post processing
                post_output = [post_processing(outputs_max[k][j].cpu().numpy(), original_size) for j in
                               range(1, header.num_masks)]  # exclude background
                post_output[0] = bridge_small_gaps(post_output[0])

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
                    Image.fromarray(image_original.astype('uint8')).convert('L').save(save_dir + dir_case_id + '_image.jpg')


    if connected_patch:
        oup = save_dir.replace('/Whole',
                               '/First_connected_component')
        mydataset.create_folder(oup)
        connected_component(save_dir, oup)
        print("connected component process complete")

        make_correlation_check(oup) ### comment out if no gt masks available

def connected_component(inp, oup):
    import os.path
    import cv2
    import glob
    import numpy as np
    from correlation_code import bounding_box_fuc 

    CAPTCHA_IMAGE_FOLDER = inp
    OUTPUT_FOLDER = oup
    captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*_mask.jpg"))

    for (i, captcha_image_file) in enumerate(captcha_image_files):
        image = cv2.imread(captcha_image_file, cv2.IMREAD_UNCHANGED)

        # crop margins (same as your original)
        image_zero = np.zeros(image.shape, dtype=image.dtype)
        image_zero[20:-20, 20:-20] = image[20:-20, 20:-20]
        image = image_zero

        # threshold
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # connected components
        ret, labels = cv2.connectedComponents(binary)
        sizes = []
        masks = []

        for label in range(1, ret):
            mask = np.zeros_like(binary, dtype=np.uint8)
            mask[labels == label] = 255
            sizes.append(mask.sum())
            masks.append(mask)

        if not sizes:
            continue

        # sort components by size (descending)
        idx_sorted = np.argsort(sizes)[::-1]
        largest = masks[idx_sorted[0]]

        keep_mask = largest.copy()

        if len(idx_sorted) > 1:
            second = masks[idx_sorted[1]]

            # compute tip positions
            tip1 = bounding_box_fuc(largest)
            tip2 = bounding_box_fuc(second)

            center = (image.shape[1] // 2, image.shape[0] // 2)

            # Euclidean distance to center
            d1 = np.hypot(tip1[0] - center[0], tip1[1] - center[1])
            d2 = np.hypot(tip2[0] - center[0], tip2[1] - center[1])

            keep_mask = largest.copy()

            # keep second if its tip is closer to center
            if d2 < d1:
                keep_mask = cv2.bitwise_or(largest, second)

        # save result
        filename = os.path.basename(captcha_image_file)
        p = os.path.join(OUTPUT_FOLDER, filename.replace('_mask', ''))
        cv2.imwrite(p, keep_mask)

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
