# -*- coding: utf-8 -*-
"""
Created on Fri Mar 1 2021
@author: Subin Park (subinn.park@gmail.com)
"""

import header

# common
import torch

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
import numpy as np
from glob import glob
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class UnifiedPatchTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, sampler=None):
        self.image_root = image_root
        self.sample_arr = sorted(glob(os.path.join(image_root, '**/*.jpg'), recursive=True))

        # Filter out files that are not patch images (e.g., must contain 'case')
        self.sample_arr = [p for p in self.sample_arr if 'case' in p]
        self.data_len = len(self.sample_arr)

        self.ids = []
        self.images = []
        self.masks = []

        for sample_path in self.sample_arr:
            # Track ID
            rel_id = sample_path.replace(image_root, '').replace('.jpg', '')
            self.ids.append(rel_id)

            # Track image
            self.images.append(sample_path)

            # Track corresponding mask
            mask_list = []
            for x in header.dir_mask_path:
                mask_path = image_root.replace('/input_Catheter' + header.Data_path, x) + rel_id + '.jpg'
                if os.path.isfile(mask_path):
                    mask_list.append(mask_path)

            if not mask_list:
                raise FileNotFoundError(f"No corresponding mask found for: {sample_path}")
            self.masks.append(mask_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        image_path = self.images[index]
        img = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)

        # Apply CLAHE
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        original_size = img.shape
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Load and process corresponding mask
        mask_path = self.masks[index][0]
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return {
            'input': torch.tensor(img, dtype=torch.float32),
            'masks': torch.tensor(mask, dtype=torch.float32),
            'ids': self.ids[index],
            'im_size': np.array(img.shape[1:])  # H, W
        }

    def get_id(self, index):
        return self.ids[index]

    def get_original(self, index):
        img = np.array(Image.open(self.images[index]).convert('L'), dtype=np.uint8)
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        return img


def main():

    print("\nSecond stage inference.py")


    #################################################################################################################resize_height#############
    # Semantic segmentation (inference)

    # Flag
    flag_eval_JI = True#True #False # calculate JI
    flag_save_PNG = True # preprocessR2U_Neted, mask
    original_remake = True
    third_data = True

    # GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_worker = header.num_worker
    else:
        device = torch.device("cpu")
        num_worker = 0


    # Model initialization
    net = header.net_1
    print(str(header.test_secon_network_name), "inference")


    # network to GPU
    if torch.cuda.device_count() > 1:
        print("GPU COUNT = ", str(torch.cuda.device_count()))
        net = nn.DataParallel(net)


    # Load model
    model_dir = header.dir_checkpoint + 'FCDenseNet_Stage2_epoch9.pth' # --main whole

    if os.path.isfile(model_dir):
        print(f'\n>> Load model - {model_dir}')
        checkpoint = torch.load(model_dir, map_location=device)

        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v  # strip 'module.' if present

        net.load_state_dict(new_state_dict)
        print(f"  >>> Loaded Epoch: {checkpoint['epoch']}")
    else:
        print(f'[Err] Model does not exist in {model_dir}')
        exit()

    net.to(device)
    net.train()

    test_sampler = None

    dir_Second_test_path = header.dir_secon_data_root + 'output_inference_segmentation_endtoend' + test_network_name + header.Whole_Catheter + '/First_output/random_crop/data/input_Catheter_' + header.Whole_Catheter
    # loop dataset class
    # Dataset
    print('\n>> Load dataset -', dir_Second_test_path)

    test_cxr_dir = '../output/output_inference_segmentation_endtoendFCDenseNet_Whole_RANZCR/First_output/random_crop/data/input_Catheter__Whole_RANZCR/PICC'
    testset = UnifiedPatchTestDataset(test_cxr_dir)
    testloader = DataLoader(testset, batch_size=header.num_batch_test, shuffle=False, num_workers=num_worker, pin_memory=True)

    print("  >>> Total # of test sampler : %d" % (len(testset)))

    # inference
    print('\n\n>> Evaluate Network')
    with torch.no_grad():

        # initialize
        net.train()
        ji_test = []

        for i, data in enumerate(testloader, 0):
            outputs = net(data['input'].to(device))
            outputs = outputs.detach().cpu()

            for k in range(len(data['input'])):
                prob_map = torch.sigmoid(outputs[k]).squeeze().numpy()
                pred_mask = (prob_map > 0.5).astype(np.uint8)
                pred_mask_bin = pred_mask

                original_size, dir_case_id, _ = mydataset.get_size_id(k, data['im_size'], data['ids'], header.net_label[1:])
                resized_mask = cv2.resize(pred_mask_bin, original_size, interpolation=cv2.INTER_NEAREST)

                # Save
                save_dir = header.dir_save + '/output_inference_segmentation_endtoend' + str(test_network_name) + header.Whole_Catheter + "/" + 'second_output_90/'+ str(header.test_secon_network_name)+'/patch'
                dir_case_id = dir_case_id.replace('/PICC', '').replace('/Normal', '')
                mydataset.create_folder(save_dir)

                Image.fromarray(resized_mask * 255).convert('L').save(save_dir + dir_case_id + '_mask.jpg')
                image_original = testset.get_original(i * header.num_batch_test + k)
                Image.fromarray(image_original.astype('uint8')).convert('L').save(save_dir + dir_case_id + '_image.jpg')

    if original_remake:
        inp = header.original_dir
        patch_dir = save_dir
        oup = save_dir.replace('/patch',
                                    '/whole/')
        mydataset.create_folder(oup)


        Original_file_make(inp, patch_dir, oup)
        make_correlation_check(oup) ### comment out if no gt masks available
        print("correlation finish")

    if third_data:
        inp = header.original_dir
        patch_dir = save_dir
        oup = save_dir.replace('/patch',
                                    '/data/')


        third_data_make(inp, patch_dir, oup)


def Original_file_make(inp, patch_dir, oup):
    from PIL import Image
    import os
    import numpy as np

    inp = inp
    patch_dir = patch_dir
    oup = oup

    filenames = os.listdir(inp)
    patchnames = [x for x in os.listdir(patch_dir) if '_mask' in x]

    for files in filenames:

        file_form = files.split('.')

        im = np.asarray(np.array(Image.open(os.path.join(inp, files)).convert('L'), 'uint8'))

        width, height = im.shape  # Get dimensions

        img = np.zeros((width, height), np.uint16)
        Count_img = np.ones((width, height), np.uint8)

        for patch_file in patchnames:
            if patch_file.startswith(file_form[0] + '_'):
                filevalue = patch_file.split('.')[0]
                patch = np.asarray(np.array(Image.open(os.path.join(patch_dir, patch_file)).convert('L'), 'uint8'))

                img[int(filevalue.split('_')[3]):int(filevalue.split('_')[5]),
                int(filevalue.split('_')[2]):int(filevalue.split('_')[4])] += patch
                # print(img.max())
                # uint8은 255가 max값이라 더이상 값이 커지지 않는다.

                Count_img[int(filevalue.split('_')[3]):int(filevalue.split('_')[5]),
                int(filevalue.split('_')[2]):int(filevalue.split('_')[4])] += 1

        img = img / 255.0

        img[img > 0.95] = 1
        img[img <= 0.95] = 0

        img = (img * 255).astype('uint8')

        Image.fromarray(img.astype('uint8')).convert('L').save(oup + files)


def third_data_make(inp, patch_dir, oup):
    from PIL import Image
    import os
    import numpy as np

    inp = inp
    patch_dir = patch_dir
    oup = oup

    oup1 = oup + "/input_Catheter_" + header.Whole_Catheter + '/PICC/'
    oup2 = oup + "/mask_Catheter_" + header.Whole_Catheter + '/PICC/'

    mydataset.create_folder(oup1)
    mydataset.create_folder(oup2)

    filenames = os.listdir(inp)
    patchnames = [x for x in os.listdir(patch_dir) if '_mask' in x]

    for files in filenames:

        file_form = files.split('.')

        im = np.asarray(np.array(Image.open(os.path.join(inp, files)).convert('L'), 'uint8'))

        width, height = im.shape  # Get dimensions

        img = np.zeros((width, height), np.uint16)
        Count_img = np.ones((width, height), np.uint8)

        for patch_file in patchnames:
            if patch_file.startswith(file_form[0] + '_'):
                filevalue = patch_file.split('.')[0]
                patch = np.asarray(np.array(Image.open(os.path.join(patch_dir, patch_file)).convert('L'), 'uint8'))

                img[int(filevalue.split('_')[3]):int(filevalue.split('_')[5]),
                int(filevalue.split('_')[2]):int(filevalue.split('_')[4])] += patch
                # print(img.max())
                # uint8은 255가 max값이라 더이상 값이 커지지 않는다.

                Count_img[int(filevalue.split('_')[3]):int(filevalue.split('_')[5]),
                int(filevalue.split('_')[2]):int(filevalue.split('_')[4])] += 1

        img = img / 255.0

        img[img > 0.95] = 1
        img[img <= 0.95] = 0

        img = (img * 255).astype('uint8')

        Image.fromarray(img.astype('uint8')).convert('L').save(oup1 + files)
        Image.fromarray(img.astype('uint8')).convert('L').save(oup2 + files)


def get_JI(pred_m, gt_m):

    intersection = np.logical_and(gt_m, pred_m)

    true_sum= gt_m[:,:].sum()
    pred_sum= pred_m[:,:].sum()
    intersection_sum = intersection[:,:].sum()

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
    #inp = "/media/subin/8TBDisk/code_review/11001_CXR_Catheter_data_folder/RESULT/RESULT_RMSE_whole/output_visualize_inferenceFCDenseNet_Whole_png_100_1024/"
    oup = inp.replace('/whole', '')

    # original_image = "/media/subin/8TBDisk/code_review/11001_CXR_Catheter_data_folder/data_generate/CXR_annotation_label_150_test/mask/test/"

    for x in header.dir_mask_path:
        mask_path = header.dir_First_test_path.replace('/input_Catheter' + header.Data_path, x)

    image_name, dice, subin, point_rmse = correlation_code.correlation_Images(inp, mask_path, oup)

    df_image = pd.DataFrame([x for x in zip(image_name, dice, subin, point_rmse)],
                            columns=['image_name', 'dice', 'subin', 'point_rmse'])

    df_image.to_excel(oup + "Second_image_rmse_jpg.xlsx", sheet_name='Sheet1')

if __name__=='__main__':

    #test_network_name_list = ["FCDenseNet", "U_Net", "AttU_Net"]
    test_network_name_list = ["FCDenseNet"]

    for test_network_name in test_network_name_list:
        main()
