import json
import cv2
import numpy as np
import vis_utils as tool

img = tool.get_rgb('/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/tools/visualize_ours/000000.jpg')
gt = tool.read_json('/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/tools/visualize_ours/lidar/000000.json')
calib_lidar2cam = tool.read_json('/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/tools/visualize_ours/lid2c/000000.json')
calib_intrinsic = tool.get_cam_calib_intrinsic('/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/tools/visualize_ours/camin/000000.json')

num_rects = 0
cam_points_3d = []
r_velo2cam, t_velo2cam = tool.get_lidar2cam(calib_lidar2cam)
for ob in gt:
    num_rects += 1
    h = np.float(ob['3d_dimensions']['h'])
    w = np.float(ob['3d_dimensions']['w'])
    L = np.float(ob['3d_dimensions']['l'])
    x = np.float(ob['3d_location']['x'])
    y = np.float(ob['3d_location']['y'])
    z = np.float(ob['3d_location']['z'])
    z = z - h / 2
    dimensions_3d = np.array([L, w, h])
    location_3d = np.array([x, y, z])
    rotation = np.array(ob['rotation'])
    corner = tool.compute_box_3d(dimensions_3d, location_3d, rotation)  # 点云坐标系下3D框8个顶点的3D坐标
    # print(corner)
    camera_8_points = np.array((r_velo2cam * np.matrix(corner).T + t_velo2cam).T)  # 相机坐标系下3D框8个顶点的3D坐标
    cam_points_3d.append(camera_8_points)
#18 8 3
cam_points_3d = np.stack(cam_points_3d)
points_2d = np.array(tool.points_cam2img(cam_points_3d, calib_intrinsic))
print(num_rects)
print(points_2d.shape)

img = tool.plot_rect3d_on_img(img, num_rects, points_2d)

cv2.imwrite('/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/tools/visualize_ours/veh_gt.jpg', img)
