import json
import cv2
import numpy as np
import pickle
import tools.visualize.vis_utils as tool


def cv_show(name, pic):
    cv2.imshow(name, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


f = open(r'v2x/visual/inf.pkl', 'rb')
data = pickle.load(f)

boxes_3d = np.array(data['boxes_3d'])
scores_3d = np.array(data['scores_3d'])

print(boxes_3d.shape)
print(scores_3d)

num_rects = 0

img = tool.get_rgb('v2x/visual/vehicle-side/img.jpg')

calib_lidar2cam = tool.read_json('v2x/visual/vehicle-side/lidar_to_camera.json')
calib_intrinsic = tool.get_cam_calib_intrinsic('v2x/visual/vehicle-side/camera_intrinsic.json')

r_velo2cam, t_velo2cam = tool.get_lidar2cam(calib_lidar2cam)
cam_points_3d = []

for ind in range(0, len(boxes_3d)):
    if scores_3d[ind] < 0.5:
        continue
    num_rects += 1
    camera_8_points = np.array((r_velo2cam * np.matrix(boxes_3d[ind]).T + t_velo2cam).T)
    cam_points_3d.append(camera_8_points)

cam_points_3d = np.stack(cam_points_3d)
points_2d = np.array(tool.points_cam2img(cam_points_3d, calib_intrinsic))
img = tool.plot_rect3d_on_img(img, num_rects, points_2d)

print(num_rects)

cv2.imwrite('v2x/visual/inf.jpg', img)

