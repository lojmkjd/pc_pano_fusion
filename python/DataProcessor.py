import numpy as np
import pandas as pd
import os
import logging
from Processor import (
    PointCloudProcessor,
    PanoramaImageProcessor,
    PointCloudPanoramaFuser,
)
from typing import Optional

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataProcessor:
    def __init__(self, data_dir_path: str):
        """
        初始化数据处理器
        
        Args:
            data_dir_path (str): 数据主目录路径
            
        Raises:
            FileNotFoundError: 当数据主目录不存在时
        """
        if not os.path.exists(data_dir_path):
            raise FileNotFoundError(f"数据主目录不存在: {data_dir_path}")
            
        self.data_dir_path = data_dir_path
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"正在初始化数据处理器，数据主目录路径为: {data_dir_path}")
        self.data_paths = self.get_data_directory_paths(data_dir_path)

    def get_data_directory_paths(self, data_dir_path: str) -> dict:
        """
        获取指定数据目录下不同类型数据的子目录路径
        
        Args:
            data_dir_path (str): 数据主目录路径
            
        Returns:
            dict: 包含不同类型数据子目录路径和文件列表的字典
            
        Raises:
            FileNotFoundError: 当数据主目录不存在时
        """
        target_dirs = ["camera_pose_data", "lidar_data", "panorama_images"]
        data_paths = {}
        
        if not os.path.exists(data_dir_path):
            raise FileNotFoundError(f"数据主目录 {data_dir_path} 不存在")
            
        logging.info(f"正在获取数据主目录 {data_dir_path} 下的子目录路径")
        
        sub_dirs = os.listdir(data_dir_path)
        for sub_dir in sub_dirs:
            if sub_dir in target_dirs:
                sub_dir_path = os.path.join(data_dir_path, sub_dir)
                if not os.path.isdir(sub_dir_path):
                    logging.warning(f"子目录 {sub_dir_path} 不是一个有效的目录")
                    continue
                    
                data_paths[sub_dir] = {
                    "path": sub_dir_path,
                    "files": sorted(os.listdir(sub_dir_path)),
                }
                logging.info(
                    f"找到子目录 {sub_dir}，路径为: {sub_dir_path}，包含 {len(data_paths[sub_dir]['files'])} 个文件"
                )
        
        if not data_paths:
            logging.warning("未找到任何有效的数据子目录")
            
        return data_paths

    def correct_header(self, header: list) -> list:
        """
        修正表头，将 "X Y" 拆分为 "X" 和 "Y"
        :param header: 原始表头
        :return: 修正后的表头
        """
        # logging.info(f"正在修正表头，原始表头为: {header}")
        if "X Y" in header:
            index = header.index("X Y")
            header = header[:index] + ["X", "Y"] + header[index + 1 :]
            # logging.info(f"表头修正后为: {header}")
        return header

    def correct_data_row(self, parts: list, header: list) -> list:
        """
        修正数据行，确保数据列数与表头一致
        :param parts: 原始数据行
        :param header: 表头
        :return: 修正后的数据行
        """
        # logging.info(f"正在修正数据行，原始数据行为: {parts}，表头为: {header}")
        if len(parts) == len(header) - 1:
            last_two = parts.pop()
            parts.extend(last_two.rsplit(" ", 1))
            # logging.info(f"数据行修正后为: {parts}")
        return parts

    def read_pos_data_file(self, pos_data_file: str) -> Optional[pd.DataFrame]:
        """
        读取相机姿态数据文件，并将其处理成DataFrame格式
        
        Args:
            pos_data_file (str): 相机姿态数据文件路径
            
        Returns:
            Optional[pd.DataFrame]: 处理后的DataFrame，如果处理失败则返回None
            
        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件格式不正确时
        """
        self.logger.info(f"正在读取相机姿态数据文件: {pos_data_file}")
        
        if not os.path.exists(pos_data_file):
            raise FileNotFoundError(f"相机姿态数据文件不存在: {pos_data_file}")
            
        _, file_extension = os.path.splitext(pos_data_file)
        if file_extension != ".txt":
            raise ValueError(f"文件 {pos_data_file} 不是 .txt 文件")

        try:
            with open(pos_data_file, "r", encoding="utf-8") as file:
                lines = file.readlines()

            if not lines:
                raise ValueError(f"文件 {pos_data_file} 为空")

            # 处理表头
            header = self.correct_header(lines[0].strip().split())

            # 使用列表推导式优化数据处理
            data = [
                self.correct_data_row(line.strip().split(), header)
                for line in lines[1:]
            ]

            df = pd.DataFrame(data, columns=header)
            self.logger.info(f"成功读取相机姿态数据文件，DataFrame 包含 {len(df)} 行")
            return df
            
        except Exception as e:
            self.logger.error(f"读取文件时发生错误: {str(e)}")
            raise

    def get_panorama_image_path(self, row: pd.Series, group_index: int) -> str:
        """
        获取全景照片的路径
        
        Args:
            row (pd.Series): DataFrame中的一行数据
            group_index (int): 数据组索引
            
        Returns:
            str: 全景照片的完整路径
            
        Raises:
            KeyError: 当数据中缺少文件名时
            IndexError: 当组索引超出范围时
            FileNotFoundError: 当图像文件不存在时
        """
        try:
            panorama_image_name = row["Filename"]
            panorama_image_dir = self.data_paths["panorama_images"]["files"][group_index - 1]
            panorama_image_path = os.path.join(
                self.data_paths["panorama_images"]["path"],
                panorama_image_dir,
                panorama_image_name,
            )
            
            if not os.path.exists(panorama_image_path):
                raise FileNotFoundError(f"全景图像文件不存在: {panorama_image_path}")
                
            self.logger.info(f"获取到全景照片路径: {panorama_image_path}")
            return panorama_image_path
            
        except KeyError:
            raise KeyError("数据中缺少'Filename'列")
        except IndexError:
            raise IndexError(f"数据组索引 {group_index} 超出范围")

    def process_single_row(
        self,
        row: pd.Series,
        group_index: int,
        point_cloud_processor: PointCloudProcessor,
        panorama_image_processor: PanoramaImageProcessor,
        fuser: PointCloudPanoramaFuser,
    ) -> None:
        """
        处理DataFrame中的单行数据
        
        Args:
            row (pd.Series): DataFrame中的一行数据
            group_index (int): 数据组索引
            point_cloud_processor (PointCloudProcessor): 点云处理器实例
            panorama_image_processor (PanoramaImageProcessor): 全景图像处理器实例
            fuser (PointCloudPanoramaFuser): 融合器实例
            
        Raises:
            KeyError: 当数据行中缺少必要的列时
            ValueError: 当数据格式不正确时
            IndexError: 当索引超出范围时
        """
        logging.info(
            f"开始处理 DataFrame 中的单行数据，行索引: {row.name}，数据组索引: {group_index}"
        )
        try:
            # 读取对应的全景照片
            panorama_image_path = self.get_panorama_image_path(row, group_index)

            # 获取全景相机的平移向量
            translation = np.array(
                [float(row["X"]), float(row["Y"]), float(row["Altitude"])]
            )
            logging.info(f"获取到全景相机的平移向量: {translation}")

            # 获取全景相机的姿态角
            rotation_angles = np.radians(
                [float(row["Roll"]), float(row["Pitch"]), float(row["Heading"])]
            )
            logging.info(f"获取到全景相机的姿态角: {rotation_angles}")

            point_cloud_processor.update_pose(translation, rotation_angles)
            panorama_image_processor.update_image(panorama_image_path)
            fuser.update(point_cloud_processor, panorama_image_processor)

            # 执行融合操作
            fuser.fuse_point_cloud_with_color()
            
            # r, phi, theta = fuser.cartesian_to_polar()
            # if r is not None and phi is not None and theta is not None:
            #     x_p, y_p = fuser.calculate_pixel_coordinates(phi, theta)
            #     h, w = fuser.get_color_at_pixel(r,x_p, y_p)
            #     fuser.generate_panorama_from_r(r, w, h)
            #     fuser.generate_color_panorama(w, h)
            # else:
            #     logging.error("点云数据格式错误，无法进行后续处理。")
        except KeyError as e:
            logging.error(f"数据行中缺少必要的列 {e}，请检查数据格式。")
        except (ValueError, IndexError) as e:
            logging.error(f"数据处理过程中出现错误：{e}，请检查数据内容。")

    def process_data(self, group_index: int = 1) -> None:
        """
        处理数据的主要接口函数
        
        Args:
            group_index (int): 数据组索引，默认为1
            
        Returns:
            None
        """
        logging.info(f"开始处理数据，数据组索引: {group_index}")
        
        if not self.data_paths:
            raise ValueError("未找到有效数据目录")

        try:
            # 预先获取所有需要的文件路径
            pos_data_file = os.path.join(
                self.data_paths["camera_pose_data"]["path"],
                self.data_paths["camera_pose_data"]["files"][group_index - 1],
            )
            pcd_file_path = os.path.join(
                self.data_paths["lidar_data"]["path"],
                self.data_paths["lidar_data"]["files"][group_index - 1],
            )
            
            # 验证文件是否存在
            if not os.path.exists(pos_data_file):
                raise FileNotFoundError(f"相机姿态数据文件不存在: {pos_data_file}")
            if not os.path.exists(pcd_file_path):
                raise FileNotFoundError(f"点云文件不存在: {pcd_file_path}")

            df = self.read_pos_data_file(pos_data_file)
            if df is None:
                raise ValueError("无法读取相机姿态数据文件")

            # 初始化处理器（移到循环外部）
            point_cloud_processor = PointCloudProcessor(pcd_file_path)
            panorama_image_processor = PanoramaImageProcessor()
            fuser = PointCloudPanoramaFuser(
                point_cloud_processor, panorama_image_processor
            )

            # 使用pandas的优化方法处理数据
            df.apply(
                lambda row: self.process_single_row(
                    row,
                    group_index,
                    point_cloud_processor,
                    panorama_image_processor,
                    fuser,
                ),
                axis=1,
            )
            
            fuser.calculate_average_colors()
            fuser.point_cloud_processor.visualize_point_cloud()
            
        except Exception as e:
            logging.error(f"处理数据时发生错误: {str(e)}")
            raise

    def save_processing_results(self, fuser: PointCloudPanoramaFuser, group_index: int) -> None:
        """
        保存处理结果
        
        Args:
            fuser (PointCloudPanoramaFuser): 点云全景图融合器实例
            group_index (int): 数据组索引
            
        Raises:
            Exception: 当保存过程中发生错误时
        """
        try:
            # 生成文件名前缀
            prefix = f"group_{group_index}"
            
            # 保存融合结果
            fuser.save_results(prefix)
            self.logger.info(f"已保存第 {group_index} 组数据的处理结果")
            
        except Exception as e:
            self.logger.error(f"保存处理结果时发生错误: {str(e)}")
            raise
