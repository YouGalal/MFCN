
import cv2


import os

from sklearn.metrics import *

import os.path
import imutils
import math

def correlation_Images(model_inp, original_image, oup):
    filenames = [f for f in os.listdir(model_inp) if not f.startswith('.')]
    correlation = []
    dice = []
    MIOU = []
    f1 = []
    JS = []
    MCC = []
    subin = []
    point_rmse = []
    image_name = []

    for files in filenames:
        img = cv2.imread(os.path.join(model_inp, files), cv2.IMREAD_COLOR)
        img2 = cv2.imread(os.path.join(original_image,'PICC', files.replace('_mask', '')), cv2.IMREAD_COLOR)

        # img, img2 = Image_resize_Size(img, img2, 1024, 1024)
        img, img2 = Image_resize_Size(img, img2, img.shape[1], img.shape[0])

        gray_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_label = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        gray_pred, gray_label = Image_thresh(gray_pred, gray_label, 1)
        gray_pred = connected_component_one(gray_pred)
        gray_label = connected_component_one(gray_label)

        box_point_pred = bounding_box_fuc(gray_pred)
        box_point_label = bounding_box_fuc(gray_label)

        #
        output_path = oup + 'RMSE_folder/'
        os.makedirs(output_path, exist_ok=True)

        gray_rgb = cv2.cvtColor(gray_pred, cv2.COLOR_GRAY2BGR)
        label_rgb = cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)
        # print(files)
        # print('box_pred :', box_point_pred)
        # print('box_point_label :', box_point_label)

        added_image = cv2.addWeighted(gray_rgb, 0.4, label_rgb, 0.1, 0)

        cv2.circle(added_image, box_point_pred, 5, (0, 255, 0), -1)
        cv2.circle(added_image, box_point_label, 5, (0, 0, 255), -1)
        cv2.rectangle(added_image, (box_point_pred[0] - 10, box_point_pred[1] - 10),
                      (box_point_pred[0] + 10, box_point_pred[1] + 10), (255, 0, 0), 3)
        cv2.rectangle(added_image, (box_point_label[0] - 10, box_point_label[1] - 10),
                      (box_point_label[0] + 10, box_point_label[1] + 10), (0, 0, 255), 3)

        # cv2.imwrite('im.jpg', gray_rgb)
        cv2.imwrite(output_path + files, added_image)

        #

        cor = np.mean(np.multiply((img - np.mean(img)), (img2 - np.mean(img2)))) / (np.std(img) * np.std(img2))
        dice_score = dice_calc(img2, img)
        mIOU_score = get_mean_iou(img, img2)
        f1_score = get_f1_score(img, img2)
        JS_score = get_JI(img, img2)
        MCC_score = get_mcc(img, img2)
        subin_score = get_subin_code(img, img2)
        rmse_score = RMSE_calc(box_point_label, box_point_pred)

        correlation.append(cor)
        dice.append(dice_score)
        MIOU.append(mIOU_score)
        f1.append(f1_score)
        JS.append(JS_score)
        MCC.append(MCC_score)
        subin.append(subin_score)
        point_rmse.append(rmse_score)
        image_name.append(files)

    return image_name, dice, subin, point_rmse


def correlation_Firt_Images(model_inp, original_image, oup):
    filenames = [f for f in os.listdir(model_inp) if f.endswith('_mask.jpg')]
    correlation = []
    dice = []
    MIOU = []
    f1 = []
    JS = []
    MCC = []
    subin = []
    point_rmse = []
    image_name = []

    for files in filenames:
        img = cv2.imread(os.path.join(model_inp, files), cv2.IMREAD_COLOR)
        img2 = cv2.imread(os.path.join(original_image,'PICC', files.replace('_mask', '')), cv2.IMREAD_COLOR)

        # img, img2 = Image_resize_Size(img, img2, 1024, 1024)
        img, img2 = Image_resize_Size(img, img2, img.shape[1], img.shape[0])

        gray_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_label = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        gray_pred, gray_label = Image_thresh(gray_pred, gray_label, 1)
        gray_pred = connected_component_one(gray_pred)
        gray_label = connected_component_one(gray_label)

        box_point_pred = bounding_box_fuc(gray_pred)
        box_point_label = bounding_box_fuc(gray_label)

        #
        output_path = oup + '/RMSE_folder/'
        os.makedirs(output_path, exist_ok=True)

        gray_rgb = cv2.cvtColor(gray_pred, cv2.COLOR_GRAY2BGR)
        label_rgb = cv2.cvtColor(gray_label, cv2.COLOR_GRAY2BGR)
        # print(files)
        # print('box_pred :', box_point_pred)
        # print('box_point_label :', box_point_label)

        added_image = cv2.addWeighted(gray_rgb, 0.4, label_rgb, 0.1, 0)

        cv2.circle(added_image, box_point_pred, 5, (0, 255, 0), -1)
        cv2.circle(added_image, box_point_label, 5, (0, 0, 255), -1)
        cv2.rectangle(added_image, (box_point_pred[0] - 10, box_point_pred[1] - 10),
                      (box_point_pred[0] + 10, box_point_pred[1] + 10), (255, 0, 0), 3)
        cv2.rectangle(added_image, (box_point_label[0] - 10, box_point_label[1] - 10),
                      (box_point_label[0] + 10, box_point_label[1] + 10), (0, 0, 255), 3)

        # cv2.imwrite('im.jpg', gray_rgb)
        cv2.imwrite(output_path + files, added_image)

        #

        cor = np.mean(np.multiply((img - np.mean(img)), (img2 - np.mean(img2)))) / (np.std(img) * np.std(img2))
        dice_score = dice_calc(img2, img)
        mIOU_score = get_mean_iou(img, img2)
        f1_score = get_f1_score(img, img2)
        JS_score = get_JI(img, img2)
        MCC_score = get_mcc(img, img2)
        subin_score = get_subin_code(img, img2)
        rmse_score = RMSE_calc(box_point_label, box_point_pred)

        correlation.append(cor)
        dice.append(dice_score)
        MIOU.append(mIOU_score)
        f1.append(f1_score)
        JS.append(JS_score)
        MCC.append(MCC_score)
        subin.append(subin_score)
        point_rmse.append(rmse_score)
        image_name.append(files.replace('_mask', ''))

    return image_name, dice, subin, point_rmse

def bounding_box_fuc(img):
    flip_count = False

    right_index = int(img.shape[1] / 4)

    crop_img = img[:, 0:right_index]

    if np.sum(crop_img) > 100:  # 1은 카테터존

        mid_point = img.shape[1] / 2
        img = cv2.flip(img, 1)

        flip_count = True
        # print('flip')

    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv3로 update되면서 바뀐 부분

    contours = contours[1] if imutils.is_cv3() else contours[0]

    maxvalue = 0
    check_value_list = []

    # gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    if len(contours)== 0:
        box_point = tuple([0, 0])
        return box_point
    else:



        cnt = contours[0]

        most_left_point = tuple(cnt[cnt[:, :, 0].argmin()][0])[0] + 50

        for i in cnt:
            if i[0, 0] < most_left_point:
                check_value = i[0, 1]
                if check_value > maxvalue:
                    maxvalue = check_value
                    box_point_cnt = i[0]
                    box_point_right = box_point_cnt[0]

        if flip_count:
            point_x_value = mid_point - box_point_cnt[0]
            box_point_right = mid_point + point_x_value

        box_point = tuple([int(box_point_right), int(box_point_cnt[1])])

        return box_point



def connected_component_one(image):
    maxvalue = 0
    image_zero = np.zeros(image.shape, dtype=image.dtype)

    image_zero[20:-20, 20:-20] = image[20:-20, 20:-20]

    image = image_zero

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

    # gray_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    return im


def Image_resize_Size(img, img2, re_width, re_height):
    img = cv2.resize(img, (re_width, re_height))
    img2 = cv2.resize(img2, (re_width, re_height))

    # img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))

    return img, img2


def Image_thresh(img, img2, thresh_value):
    img[img >= thresh_value] = 255
    img[img < thresh_value] = 0
    img2[img2 >= thresh_value] = 255
    img2[img2 < thresh_value] = 0

    return img, img2


def dice_calc(true_mask, pred_mask, non_seg_score=1.0):
    """
        Computes the Dice coefficient.
        Args:
            true_mask : Array of arbitrary shape.
            pred_mask : Array with the same shape than true_mask.

        Returns:
            A scalar representing the Dice coefficient between the two segmentations.

    """
    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    # If both segmentations are all zero, the dice will be 1. (Developer decision)
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum


import numpy as np


def RMSE_calc(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)

    return rmse


def BinaryConfusionMatrix(prediction, groundtruth):
    """Computes scores:
    TP = True Positives
    FP = False Positives
    FN = False Negatives
    TN = True Negatives
    return: TP, FP, FN, TN"""
    prediction = np.asarray(prediction).astype(np.bool)
    groundtruth = np.asarray(groundtruth).astype(np.bool)

    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return TN, FP, FN, TP


def get_precision(prediction, groundtruth):
    _, FP, _, TP = BinaryConfusionMatrix(prediction, groundtruth)
    precision = float(TP) / (float(TP + FP) + 1e-6)
    return precision


def get_recall(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    recall = float(TP) / (float(TP + FN) + 1e-6)
    return recall


def get_accuracy(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    accuracy = float(TP + TN) / (float(TP + FP + FN + TN) + 1e-6)
    return accuracy


def get_sensitivity(prediction, groundtruth):
    return get_recall(prediction, groundtruth)


def get_specificity(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    specificity = float(TN) / (float(TN + FP) + 1e-6)
    return specificity


def get_f1_score(prediction, groundtruth):
    precision = get_precision(prediction, groundtruth)
    recall = get_recall(prediction, groundtruth)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    return f1_score


def get_dice(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    dice = 2 * float(TP) / (float(FP + 2 * TP + FN) + 1e-6)
    return dice


def get_iou1(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TP) / (float(FP + TP + FN) + 1e-6)
    return iou


def get_iou0(prediction, groundtruth):
    TN, FP, FN, TP = BinaryConfusionMatrix(prediction, groundtruth)
    iou = float(TN) / (float(FP + TN + FN) + 1e-6)
    return iou


def get_mean_iou(prediction, groundtruth):
    iou0 = get_iou0(prediction, groundtruth)
    iou1 = get_iou1(prediction, groundtruth)
    mean_iou = (iou1 + iou0) / 2
    return mean_iou


def get_JI(pred_m, gt_m):
    pred_m = np.asarray(pred_m).astype(np.bool)
    gt_m = np.asarray(gt_m).astype(np.bool)

    intersection = np.logical_and(gt_m, pred_m)

    # print("print gt_m:",gt_m)
    # print("print gt_m.size:", gt_m.shape)

    true_sum = gt_m[:, :].sum()
    pred_sum = pred_m[:, :].sum()
    intersection_sum = intersection[:, :].sum()

    ji = (intersection_sum + 1.) / (true_sum + pred_sum - intersection_sum + 1.)

    return ji


def get_mcc(prediction, groundtruth):
    from math import sqrt
    tn, fp, fn, tp = BinaryConfusionMatrix(prediction, groundtruth)
    # https://stackoverflow.com/a/56875660/992687
    x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return ((tp * tn) - (fp * fn)) / (sqrt(x) + 1e-6)


def thick_saveas(img, thick_as=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(img, kernel, iterations=thick_as)

    return dilate


def get_subin_code(prediction, groundtruth):
    from math import sqrt

    prediction_comma = thick_saveas(prediction, thick_as=3)
    groundtruth_comma = thick_saveas(groundtruth, thick_as=3)

    prediction = np.asarray(prediction).astype(np.bool)
    groundtruth = np.asarray(groundtruth).astype(np.bool)
    prediction_comma = np.asarray(prediction_comma).astype(np.bool)
    groundtruth_comma = np.asarray(groundtruth_comma).astype(np.bool)

    one = np.float(np.sum((prediction_comma == 1) & (groundtruth == 1))) / (np.float(np.sum(groundtruth == 1))+0.000000)
    two = np.float(np.sum((prediction == 1) & (groundtruth_comma == 1))) /(np.float(np.sum(prediction == 1))+0.0000000001)

    return ((one + two) / 2)