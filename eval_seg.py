import pandas as pd
import numpy as np
import nibabel as nib
import os
from medpy.metric.binary import dc, hd95
from tqdm import tqdm

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

def Dice(pred, gt):
    dice = dc(pred, gt)
    return dice


def Hd95(pred, gt):
    hd = hd95(pred, gt)
    return hd

def cal_dices(pred_path, gt_path):
    # dice1(ET): label4
    # dice2(TC): label1 + label4
    # dice3(WT): label1 + label2 + label4
    output = nib_load(pred_path)
    target = nib_load(gt_path)
    et = Dice((output == 4), (target == 4))
    tc = Dice(((output == 1) | (output == 4)), ((target == 1) | (target == 4)))
    wt = Dice((output != 0), (target != 0))

    return et, tc, wt

def cal_hds(pred_path, gt_path):
    # dice1(ET): label4
    # dice2(TC): label1 + label4
    # dice3(WT): label1 + label2 + label4
    output = nib_load(pred_path)
    target = nib_load(gt_path)
    et = Hd95((output == 4), (target == 4))
    tc = Hd95(((output == 1) | (output == 4)), ((target == 1) | (target == 4)))
    wt = Hd95((output != 0), (target != 0))

    return et, tc, wt


if __name__ == '__main__':
    root_pred = 'output/submission/TransBraTS_mgmt2024-01-29/'
    file_list = 'data/mgmt_test.csv'
    subjects = pd.read_csv(file_list, dtype=str)

    id = subjects['BraTS21ID']
    paths_pred = [(root_pred + 'BraTS2021_' + str(name) + '.nii.gz') for name in id]

    root_gt = 'H:/Brats_Dataset/BraTS2021_TrainingData/'
    paths_gt = [(root_gt + 'BraTS2021_' + str(name) + '/BraTS2021_' + str(name) + '_seg.nii.gz') for name in id]

    results = pd.DataFrame(columns=['id', 'dice_et', 'dice_tc', 'dice_wt',
                                    'hd_et', 'hd_tc', 'hd_wt'])
    for i in tqdm(range(len(paths_pred))):
        assert paths_pred[i].split('/')[-1].split('_')[1].split('.')[0]==paths_gt[i].split('/')[-1].split('_')[1]
        dice_et, dice_tc, dice_wt = cal_dices(paths_pred[i], paths_gt[i])
        if dice_et > 0.1:
            hd_et, hd_tc, hd_wt = cal_hds(paths_pred[i], paths_gt[i])
        else:
            print('error!')
            print(id, dice_et, dice_tc, dice_wt)
            continue
        id = paths_pred[i].split('/')[-1].split('.')[0]

        results = results.append({'id':id, 'dice_et':dice_et, 'dice_tc':dice_tc, 'dice_wt':dice_wt,
                                  'hd_et':hd_et, 'hd_tc':hd_tc, 'hd_wt':hd_wt}, ignore_index=True)

        print(dice_et, dice_tc, dice_wt)
        print(hd_et, hd_tc, hd_wt)
        print('--------------------')

    results.to_csv('eval_seg_res.csv')


