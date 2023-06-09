import mmcv
import numpy as np
# def load_points(self, pts_filename):
#     """Private function to load point clouds data.

#     Args:
#         pts_filename (str): Filename of point clouds data.

#     Returns:
#         np.ndarray: An array containing point clouds data.
#     """
#     if self.file_client is None:
#         self.file_client = mmcv.FileClient(**self.file_client_args)
#     try:
#         pts_bytes = self.file_client.get(pts_filename)
#         points = np.frombuffer(pts_bytes, dtype=np.float32)
#     except ConnectionError:
#         mmcv.check_file_exist(pts_filename)
#         if pts_filename.endswith('.npy'):
#             points = np.load(pts_filename)
#         else:
#             points = np.fromfile(pts_filename, dtype=np.float32)

#     return points

pts_filename = "/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/training/velodyne/005844.pcd"

points = np.fromfile(pts_filename, dtype=np.float32)

print(points.shape)