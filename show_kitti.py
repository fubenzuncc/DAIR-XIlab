from pypcd import pypcd
import numpy as np
pcd_file_path = "/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/velodyne/000000.pcd"
pc = pypcd.PointCloud.from_path(pcd_file_path)

np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255
