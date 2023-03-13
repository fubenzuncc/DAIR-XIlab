import json
import cv2
import numpy as np
import pickle
import tools.visualize.vis_utils as tool
import tools.dataset_converter.point_cloud_i2v as transi2v


def trans_point_v2i(input_point, path_virtuallidar2world, path_novatel2world, path_lidar2novatel):
    # 车端点云坐标系 to novatel坐标系
    rotation, translation = transi2v.get_lidar2novatel(path_lidar2novatel)
    point = transi2v.trans(input_point, translation, rotation)

    # novatel坐标系 to 世界坐标系
    rotation, translation = transi2v.get_novatel2world(path_novatel2world)
    point = transi2v.trans(point, translation, rotation)

    # 世界坐标系 to 路端坐标系
    rotation, translation, delta_x, delta_y = transi2v.get_virtuallidar2world(path_virtuallidar2world)
    new_rotation = transi2v.rev_matrix(rotation)
    new_translation = -np.dot(new_rotation, translation)
    point -= np.array([delta_x, delta_y, 0]).reshape(3, 1)
    point = transi2v.trans(point, new_translation, new_rotation)

    point = point.reshape(1, 3).tolist()
    point = point[0]

    return point


def cv_show(name, pic):
    cv2.imshow(name, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


f = open(r'v2x/visual/result.pkl', 'rb')
data = pickle.load(f)

boxes_3d = np.array(data['boxes_3d'])
scores_3d = np.array(data['scores_3d'])

new_boxes_3d = []
for box in boxes_3d:
    new_box = []
    for point in box:
        path_virtuallidar2world = 'v2x/visual/infrastructure-side/virtuallidar_to_world.json'
        path_novatel2world = 'v2x/visual/vehicle-side/novatel_to_world.json'
        path_lidar2novatel = 'v2x/visual/vehicle-side/lidar_to_novatel.json'
        point = trans_point_v2i(point, path_virtuallidar2world, path_novatel2world, path_lidar2novatel)
        new_box.append(point)
    new_boxes_3d.append(new_box)

boxes_3d = np.array(new_boxes_3d)

# print(boxes_3d.shape)
# print(scores_3d)

num_rects = 0

img = tool.get_rgb('v2x/visual/infrastructure-side/img.jpg')

calib_lidar2cam = tool.read_json('v2x/visual/infrastructure-side/virtuallidar_to_camera.json')
calib_intrinsic = tool.get_cam_calib_intrinsic('v2x/visual/infrastructure-side/camera_intrinsic.json')

r_velo2cam, t_velo2cam = tool.get_lidar2cam(calib_lidar2cam)
cam_points_3d = []

for ind in range(0, len(boxes_3d)):
    if scores_3d[ind] < 0.2:
        continue
    num_rects += 1
    camera_8_points = np.array((r_velo2cam * np.matrix(boxes_3d[ind]).T + t_velo2cam).T)
    cam_points_3d.append(camera_8_points)

cam_points_3d = np.stack(cam_points_3d)
points_2d = np.array(tool.points_cam2img(cam_points_3d, calib_intrinsic))
img = tool.plot_rect3d_on_img(img, num_rects, points_2d)

print(num_rects)

cv2.imwrite('v2x/visual/road_side_coo.jpg', img)

