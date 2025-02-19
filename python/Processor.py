import numpy as np
import cv2
import open3d as o3d
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict


class PanoramaImageProcessor:
    def __init__(self):
        """
        初始化全景图像处理器
        """
        self.panorama_path: str | None = None
        self.panorama_image: np.ndarray | None = None
        self.logger = logging.getLogger(__name__)

    def update_image(self, panorama_path: str) -> None:
        """
        更新全景图像文件
        :param panorama_path: 全景图像路径
        :raises ValueError: 当图像文件无法读取时抛出
        """
        if not os.path.exists(panorama_path):
            raise FileNotFoundError(f"图像文件不存在: {panorama_path}")
            
        self.panorama_image = cv2.imread(panorama_path)
        if self.panorama_image is None:
            raise ValueError(f"无法读取图像文件: {panorama_path}")
        
        self.panorama_path = panorama_path
        self.logger.info(f"成功加载全景图像: {panorama_path}")

    def visualize_panorama_image(self):
        cv2.imshow("Panorama Image", self.panorama_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class PointCloudProcessor:
    def __init__(self, pcd_file_path: str, config: dict | None = None):
        """
        初始化点云处理器类。
        :param pcd_file_path: 点云文件路径
        :param config: 处理参数配置
        :raises FileNotFoundError: 当点云文件不存在时抛出
        """
        if not os.path.exists(pcd_file_path):
            raise FileNotFoundError(f"点云文件不存在: {pcd_file_path}")
            
        self.config = config or {
            'voxel_size': 0.1,
            'nb_neighbors': 20,
            'std_ratio': 2.0
        }
        self.pcd_file_path = pcd_file_path
        self.logger = logging.getLogger(__name__)
        self.pcd = self._read_and_preprocess_point_cloud()
        self.translation: np.ndarray | None = None
        self.rotation_matrix: np.ndarray | None = None
        self.point_cloud_data: np.ndarray | None = None

    def _read_and_preprocess_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        读取并预处理点云数据
        :raises ValueError: 当点云处理失败时抛出
        """
        try:
            pcd = o3d.io.read_point_cloud(self.pcd_file_path)
            if not pcd.has_points():
                self.logger.warning(f"点云文件中没有点数据: {self.pcd_file_path}")
                return pcd

            # 使用numpy进行批量处理以提高性能
            points = np.asarray(pcd.points)
            if len(points) > 1000000:  # 对大型点云进行更激进的下采样
                self.config['voxel_size'] *= 2
                self.logger.info("检测到大型点云，增加体素大小进行下采样")

            pcd = pcd.voxel_down_sample(voxel_size=self.config['voxel_size'])
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config['nb_neighbors'],
                std_ratio=self.config['std_ratio']
            )
            return pcd
        except Exception as e:
            self.logger.error(f"点云文件处理失败: {str(e)}")
            raise ValueError(f"点云文件处理失败: {str(e)}")

    @staticmethod
    def get_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        根据 roll、pitch、yaw 角度计算旋转矩阵。
        :param roll: 绕X轴旋转的角度（弧度）。
        :param pitch: 绕Y轴旋转的角度（弧度）。
        :param yaw: 绕Z轴旋转的角度（弧度）。
        :return: 旋转矩阵。
        """
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        return Rz @ Ry @ Rx  # 通过矩阵乘法计算旋转矩阵

    def update_pose(self, translation: np.ndarray, rotation_angles: np.ndarray):
        """
        更新点云的平移和旋转信息。
        :param translation: 点云坐标转换的平移向量。
        :param rotation_angles: 点云坐标转换的旋转角度，按 [roll, pitch, yaw] 顺序提供弧度值。
        """
        self.translation = translation
        self.rotation_matrix = self.get_rotation_matrix(*rotation_angles)
        self.transform_coordinates()

    def transform_coordinates(self) -> o3d.geometry.PointCloud:
        """
        对点云数据进行坐标转换。
        :return: 转换后的点云对象。
        """
        if self.translation is None or self.rotation_matrix is None:
            print("Translation or rotation matrix is not set. Skipping transformation.")
            return self.pcd

        self.point_cloud_data = np.asarray(self.pcd.points)
        # 通过平移和旋转矩阵进行坐标转换
        transformed_data = np.dot(
            (self.point_cloud_data - self.translation), self.rotation_matrix.T
        )
        self.point_cloud_data = o3d.utility.Vector3dVector(transformed_data)
        return self.pcd

    def visualize_point_cloud(self):
        """
        可视化处理后的点云数据。
        """
        o3d.visualization.draw_geometries([self.pcd])


class PointCloudPanoramaFuser:
    def __init__(self, 
                 point_cloud_processor: PointCloudProcessor,
                 panorama_image_processor: PanoramaImageProcessor,
                 output_dir: str = "output"):
        """
        初始化融合器
        :param point_cloud_processor: 点云处理器实例
        :param panorama_image_processor: 全景图像处理器实例
        :param output_dir: 输出目录
        """
        self.point_cloud_processor = point_cloud_processor
        self.panorama_image_processor = panorama_image_processor
        self.panorama_image = self.panorama_image_processor.panorama_image
        self.colors: np.ndarray | None = None
        self.color_list: list[np.ndarray] = []
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # 使用pathlib处理路径
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "colors").mkdir(exist_ok=True)
        (output_path / "greys").mkdir(exist_ok=True)

    def update(
        self,
        point_cloud_processor: PointCloudProcessor,
        panorama_image_processor: PanoramaImageProcessor,
    ):
        self.point_cloud_processor = point_cloud_processor
        self.panorama_image_processor = panorama_image_processor
        self.panorama_image = self.panorama_image_processor.panorama_image

    def cartesian_to_polar(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        将笛卡尔坐标系下的点云数据转换为极坐标系下的坐标。
        :return: 半径 r、方位角 phi、天顶角 theta的元组
        """
        point_cloud_data = np.asarray(self.point_cloud_processor.point_cloud_data)
        if point_cloud_data.ndim != 2 or point_cloud_data.shape[1] != 3:
            self.logger.error("点云数据格式错误，维度不正确")
            return None, None, None
            
        # 使用numpy的批量运算提高性能
        x, y, z = point_cloud_data.T
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # 使用clip避免数值误差
        return r, phi, theta

    def polar_to_cartesian(
        self,
        r: np.ndarray,
        phi: np.ndarray,
        theta: np.ndarray,
        radius: np.ndarray = None,
    ) -> np.ndarray:
        """
        将极坐标系下的坐标转换为笛卡尔坐标系下的点云数据。
        :param r: 半径
        :param phi: 方位角
        :param theta: 天顶角
        :param radius: 半径，若为 None 则使用 r
        :return: 笛卡尔坐标系下的点云数据
        """
        radius = r if radius is None else radius
        if radius is not None:
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            return np.column_stack((x, y, z))
        return None

    def calculate_pixel_coordinates(
        self,
        phi: np.ndarray,
        theta: np.ndarray,
        fov_w: float = 2 * np.pi,
        fov_h: float = np.pi,
    ):
        """
        计算点云在全景图像中对应像素点的坐标。
        :param phi: 方位角
        :param theta: 天顶角
        :param fov_w: 水平视场角，默认值为 2π
        :param fov_h: 垂直视场角，默认值为 π
        :return: 像素 x 坐标和像素 y 坐标
        """
        panorama_image = self.panorama_image
        panorama_width, panorama_height = (
            panorama_image.shape[1],
            panorama_image.shape[0],
        )
        x_p = ((1.295 * np.pi - phi) / fov_w) * panorama_width
        y_p = ((theta) / fov_h) * panorama_height
        return x_p, y_p

    def get_color_at_pixel(self,r: np.ndarray, x_p: np.ndarray, y_p: np.ndarray):
        """
        根据像素坐标获取全景图像中对应点的颜色。
        :param x_p: 像素 x 坐标
        :param y_p: 像素 y 坐标
        :return: 裁剪后的像素 x 坐标和像素 y 坐标
        """
        panorama_image = self.panorama_image
        w = np.clip(x_p.astype(int), 0, panorama_image.shape[1] - 1)
        h = np.clip(y_p.astype(int), 0, panorama_image.shape[0] - 1)
        self.colors = panorama_image[h, w]

        if r is not None:
            # 归一化 r
            r_min, r_max = r.min(), r.max()
            r_normalized = (r - r_min) / (r_max - r_min)*0

            # 计算权重，r 越小权重越大
            weights = 1 - r_normalized

            # 将权重应用到颜色上
            self.colors = (self.colors * weights[:, np.newaxis]).astype(np.uint8)

        self.color_list.append(self.colors)
        return h, w

    def fuse_point_cloud_with_color(self):
        """
        将计算得到的颜色附着到点云数据上。
        """
        r, phi, theta = self.cartesian_to_polar()
        if r is None or phi is None or theta is None:
            return
        x_p, y_p = self.calculate_pixel_coordinates(phi, theta)
        h, w = self.get_color_at_pixel(r,x_p, y_p)

    def calculate_average_colors(self):
        if len(self.color_list) == 0:
            return

        # 计算所有图像颜色的平均值
        average_color = np.mean(self.color_list, axis=0)

        # 将颜色值从 [0, 255] 转换为 [0, 1] 范围
        colors_normalized = average_color.astype(np.float64) / 255.0

        # 将颜色信息附着到点云上
        self.point_cloud_processor.pcd.colors = o3d.utility.Vector3dVector(
            colors_normalized
        )

    def easy_colors_fused(self):
        # 将颜色值从 [0, 255] 转换为 [0, 1] 范围
        colors_normalized = self.colors.astype(np.float64) / 255.0

        # 将颜色信息附着到点云上
        self.point_cloud_processor.pcd.colors = o3d.utility.Vector3dVector(
            colors_normalized
        )

    def _save_and_show_image(self, image, window_name, file_name):
        """
        通用的保存和显示图像的方法
        :param image: 要保存和显示的图像
        :param window_name: 显示图像的窗口名称
        :param file_name: 保存图像的文件名
        """
        try:
            # cv2.imshow(window_name, image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(file_name, image)
        except cv2.error as e:
            print(f"保存 {window_name} 时出现错误: {e}，请检查 OpenCV 配置。")

    def generate_panorama_from_r(
        self,
        r: np.ndarray,
        w: np.ndarray,
        h: np.ndarray,
        width: int = 4096,
        height: int = 2048,
    ):
        """
        根据点云的半径信息生成全景图。
        :param r: 半径
        :param w: 像素 x 坐标
        :param h: 像素 y 坐标
        :param width: 全景图像宽度，默认值为 4096
        :param height: 全景图像高度，默认值为 2048
        """
        r_min, r_max = r.min(), r.max()
        r_normalized = ((r - r_min) / (r_max - r_min) * 255).astype(np.uint8)

        r_panorama = np.zeros((height, width), dtype=np.uint8)
        r_panorama[h, w] = r_normalized

        _, filename = os.path.split(self.panorama_image_processor.panorama_path)
        file_path = os.path.join("data", "res", "greys", filename)
        self._save_and_show_image(r_panorama, "Panorama based on r values", file_path)

    def generate_color_panorama(
        self, w: np.ndarray, h: np.ndarray, width: int = 4096, height: int = 2048
    ):
        """
        根据点云的颜色信息生成彩色全景图。
        :param w: 像素 x 坐标
        :param h: 像素 y 坐标
        :param width: 全景图像宽度，默认值为 4096
        :param height: 全景图像高度，默认值为 2048
        """
        colors_panorama = np.zeros((height, width, 3), dtype=np.uint8)
        colors_panorama[h, w] = self.colors
        _, filename = os.path.split(self.panorama_image_processor.panorama_path)
        file_path = os.path.join("data", "res", "colors", filename)
        self._save_and_show_image(
            colors_panorama, "Panorama based on colors", file_path
        )

    def save_results(self, prefix: str = ""):
        """
        保存处理结果
        :param prefix: 文件名前缀
        """
        try:
            if self.point_cloud_processor.pcd is not None:
                output_path = os.path.join(
                    self.output_dir, 
                    f"{prefix}_colored_pointcloud.ply"
                )
                o3d.io.write_point_cloud(output_path, 
                                       self.point_cloud_processor.pcd)
                self.logger.info(f"已保存彩色点云到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存结果时出错: {str(e)}")


class DataPathManager:
    def __init__(self, data_dir_path: str):
        """初始化数据路径管理器"""
        if not os.path.exists(data_dir_path):
            raise FileNotFoundError(f"数据主目录不存在: {data_dir_path}")
            
        self.data_dir_path = data_dir_path
        self.logger = logging.getLogger(__name__)
        self.data_paths = self._get_data_directory_paths()

    def _get_data_directory_paths(self) -> Dict[str, str]:
        """获取数据目录结构"""
        target_dirs = ["camera_pose_data", "lidar_data", "panorama_images"]
        data_paths = {}
        
        for dir_name in target_dirs:
            dir_path = os.path.join(self.data_dir_path, dir_name)
            if not os.path.exists(dir_path):
                self.logger.warning(f"目录不存在: {dir_path}")
                continue
            data_paths[dir_name] = dir_path
            
        return data_paths

    def get_panorama_image_path(self, filename: str, group_index: int) -> str:
        """获取全景图像路径"""
        panorama_dir = self.data_paths.get("panorama_images")
        if not panorama_dir:
            raise ValueError("全景图像目录未找到")
            
        group_dir = os.path.join(panorama_dir, f"group_{group_index}")
        if not os.path.exists(group_dir):
            raise ValueError(f"组目录不存在: {group_dir}")
            
        image_path = os.path.join(group_dir, filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        return image_path

    def get_pose_data_path(self, group_index: int) -> str:
        """获取相机姿态数据路径"""
        pose_dir = self.data_paths.get("camera_pose_data")
        if not pose_dir:
            raise ValueError("相机姿态数据目录未找到")
            
        pose_file = os.path.join(pose_dir, f"group_{group_index}.txt")
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"姿态数据文件不存在: {pose_file}")
            
        return pose_file

    def get_lidar_data_path(self, filename: str, group_index: int) -> str:
        """获取激光雷达数据路径"""
        lidar_dir = self.data_paths.get("lidar_data")
        if not lidar_dir:
            raise ValueError("激光雷达数据目录未找到")
            
        group_dir = os.path.join(lidar_dir, f"group_{group_index}")
        if not os.path.exists(group_dir):
            raise ValueError(f"组目录不存在: {group_dir}")
            
        data_path = os.path.join(group_dir, filename)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"激光雷达数据文件不存在: {data_path}")
            
        return data_path
