import os
import os.path as osp
from abc import ABC, abstractmethod

import torch

from v2x.dataset.dataset_utils import read_pcd, read_jpg, load_json
from v2x.v2x_utils.transformation_utils import Coord_transformation


class Frame(dict, ABC):
    def __init__(self, path, info_dict, output_dir="/tmp"):
        self.path = path
        self.output_dir = output_dir
        for key in info_dict:
            self.__setitem__(key, info_dict[key])

    @abstractmethod
    def point_cloud(self, **args):
        raise NotImplementedError

    @abstractmethod
    def image(self, **args):
        raise NotImplementedError


class VehFrame(Frame):
    def __init__(self, path, veh_dict, tmp_key="tmps", output_dir="/tmp"):
        super().__init__(path, veh_dict, output_dir)
        self.output_dir = os.path.join(output_dir, "cache_frame")
        self.id = {}
        self.id["lidar"] = veh_dict["pointcloud_path"][-10:-4]
        self.id["camera"] = veh_dict["image_path"][-10:-4]
        self.tmp_lidar = os.path.join(self.output_dir, tmp_key, "tmp_v_lidar", self.id["lidar"] + ".bin")
        self.tmp_camera = os.path.join(self.output_dir, tmp_key, "tmp_v_camera",  self.id["camera"] + ".bin")
        if not osp.exists(os.path.join(self.output_dir, tmp_key)):
            os.system("mkdir -p " + os.path.join(self.output_dir, tmp_key))
        if not osp.exists(os.path.join(self.output_dir, tmp_key, "tmp_v_lidar")):
            os.system("mkdir -p " + os.path.join(self.output_dir, tmp_key, "tmp_v_lidar"))
        if not osp.exists(os.path.join(self.output_dir, tmp_key, "tmp_v_camera")):
            os.system("mkdir -p " + os.path.join(self.output_dir, tmp_key, "tmp_v_camera"))


    def point_cloud(self, data_format="array"):
        points, _ = read_pcd(osp.join(self.path, self.get("pointcloud_path")))
        if data_format == "array":
            return points, _
        elif data_format == "file":
            if not osp.exists(self.tmp_lidar):
                points.tofile(self.tmp_lidar)
            return self.tmp_lidar
            # return osp.join(self.path, self.get("pointcloud_path"))
        elif data_format == "tensor":
            return torch.tensor(points)

    def image(self, data_format="rgb"):
        image_array = read_jpg(osp.join(self.path, self.get("image_path")))
        if data_format == "array":
            return image_array
        elif data_format == "file":
            if not osp.exists(self.tmp_camera):
                image_array.tofile(self.tmp_camera)
            return self.tmp_camera
        elif data_format == "tensor":
            return torch.tensor(image_array)


class InfFrame(Frame):
    def __init__(self, path, inf_dict, tmp_key="tmps", output_dir="/tmp"):
        super().__init__(path, inf_dict, output_dir)
        self.output_dir = os.path.join(output_dir, "cache_frame")
        self.id = {}
        self.id["lidar"] = inf_dict["pointcloud_path"][-10:-4]
        self.id["camera"] = inf_dict["image_path"][-10:-4]
        self.tmp_lidar = os.path.join(self.output_dir, tmp_key, "tmp_i_lidar", self.id["lidar"]+".bin")
        self.tmp_camera = os.path.join(self.output_dir, tmp_key, "tmp_i_camera", self.id["camera"] + ".bin")
        if not osp.exists(os.path.join(self.output_dir, tmp_key)):
            os.system("mkdir -p " + os.path.join(self.output_dir, tmp_key))
        if not osp.exists(os.path.join(self.output_dir, tmp_key, "tmp_i_lidar")):
            os.system("mkdir -p " + os.path.join(self.output_dir, tmp_key, "tmp_i_lidar"))
        if not osp.exists(os.path.join(self.output_dir, tmp_key, "tmp_i_camera")):
            os.system("mkdir -p " + os.path.join(self.output_dir, tmp_key, "tmp_i_camera"))

    def point_cloud(self, data_format="array"):
        points, _ = read_pcd(osp.join(self.path, self.get("pointcloud_path")))
        if data_format == "array":
            return points, _
        elif data_format == "file":
            if not osp.exists(self.tmp_lidar):
                points.tofile(self.tmp_lidar)
            return self.tmp_lidar
            # return osp.join(self.path, self.get("pointcloud_path"))
        elif data_format == "tensor":
            return torch.tensor(points)

    def image(self, data_format="rgb"):
        image_array = read_jpg(osp.join(self.path, self.get("image_path")))
        if data_format == "array":
            return image_array
        elif data_format == "file":
            if not osp.exists(self.tmp_camera):
                image_array.tofile(self.tmp_camera)
            return self.tmp_camera
        elif data_format == "tensor":
            return torch.tensor(image_array)

    def transform(self, from_coord="", to_coord=""):
        """
        This function serves to calculate the Transformation Matrix from 'from_coord' to 'to_coord'
        coord_list=['Infrastructure_image','Infrastructure_camera','Infrastructure_lidar',
                       'world', 'Vehicle_image','Vehicle_camera','Vehicle_lidar',
                       'Vehicle_novatel']
        Args:
            from_coord(str): element in the coord_list
            to_coord(str):  element in coord_list
        Return:
            Transformation_Matrix: Transformation Matrix from 'from_coord' to 'to_coord'
        """
        infra_name = self.id["camera"]
        trans = Coord_transformation(from_coord, to_coord, "/".join(self.path.split("/")[:-2]), infra_name, "")
        return trans


class VICFrame(Frame):
    def __init__(self, path, info_dict, veh_frame, inf_frame, time_diff, offset=None):
        # TODO: build vehicle frame and infrastructure frame
        super().__init__(path, info_dict)
        self.veh_frame = veh_frame
        self.inf_frame = inf_frame
        self.time_diff = time_diff
        self.transformation = None
        if offset is None:
            offset = load_json(osp.join(self.inf_frame.path, self.inf_frame["calib_virtuallidar_to_world_path"]))[
                "relative_error"
            ]
        self.offset = offset

    def vehicle_frame(self):
        return self.veh_frame

    def infrastructure_frame(self):
        return self.inf_frame

    def proc_transformation(self):
        # self.transformation["infrastructure_image"]["world"]
        # read vehicle to world
        # read infrastructure to novaltel
        # read novaltel to world
        # compute inv
        # compose
        pass

    def transform(self, from_coord="", to_coord=""):
        """
        This function serves to calculate the Transformation Matrix from 'from_coord' to 'to_coord'
        coord_list=['Infrastructure_image','Infrastructure_camera','Infrastructure_lidar',
                       'world', 'Vehicle_image','Vehicle_camera','Vehicle_lidar',
                       'Vehicle_novatel']
        Args:
            from_coord(str): element in the coord_list
            to_coord(str):  element in coord_list
        Return:
            Transformation_Matrix: Transformation Matrix from 'from_coord' to 'to_coord'
        """
        veh_name = self.veh_frame["image_path"][-10:-4]
        infra_name = self.inf_frame["image_path"][-10:-4]
        trans = Coord_transformation(from_coord, to_coord, self.path, infra_name, veh_name)
        return trans
