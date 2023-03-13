import os
import json
import numpy as np
import pickle

# kitti_dbinfos_train = '/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side/kitti_dbinfos_train.pkl'
# f = open(kitti_dbinfos_train, 'rb')
# data = pickle.load(f)
# i = 1
# print(data['Car'][i].keys())
# print('name: ', data['Car'][i]['name'])
# print('path: ', data['Car'][i]['path'])
# print('image_idx: ', data['Car'][i]['image_idx'])
# print('gt_idx: ', data['Car'][i]['gt_idx'])
# print('box3d_lidar: ', data['Car'][i]['box3d_lidar'])
# print('num_points_in_gt: ', data['Car'][i]['num_points_in_gt'])
# print('difficulty: ', data['Car'][i]['difficulty'])
# print('group_id: ', data['Car'][i]['group_id'])

kitti_infos_train = '/workspace/mnt/storage/guangcongzheng/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side/kitti_infos_train.pkl'
f = open(kitti_infos_train, 'rb')
data = pickle.load(f)
i = 0
# print(data[i].keys())
# print('image:', data[i]['image'])
# print('point_cloud:', data[i]['point_cloud'])
# print('calib:', data[i]['calib'])
# print('annos:', data[i]['annos'])

print('calib:', data[i]['calib'].keys())