import os
import argparse
import logging
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
from pathlib import Path
import concurrent.futures

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_data_directory_paths(data_dir_path):
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

    logger.info(f"正在获取数据主目录 {data_dir_path} 下的子目录路径")

    sub_dirs = os.listdir(data_dir_path)
    for sub_dir in sub_dirs:
        if sub_dir in target_dirs:
            sub_dir_path = os.path.join(data_dir_path, sub_dir)
            if not os.path.isdir(sub_dir_path):
                logger.warning(f"子目录 {sub_dir_path} 不是一个有效的目录")
                continue

            data_paths[sub_dir] = {
                "path": sub_dir_path,
                "files": sorted(os.listdir(sub_dir_path)),
            }
            logger.info(
                f"找到子目录 {sub_dir}，路径为: {sub_dir_path}，包含 {len(data_paths[sub_dir]['files'])} 个文件"
            )

    if not data_paths:
        logger.warning("未找到任何有效的数据子目录")

    return data_paths


def correct_header(header):
    """
    修正表头，将 "X Y" 拆分为 "X" 和 "Y"
    :param header: 原始表头
    :return: 修正后的表头
    """
    if "X Y" in header:
        index = header.index("X Y")
        header = header[:index] + ["X", "Y"] + header[index + 1:]
    return header


def correct_data_row(parts, header):
    """
    修正数据行，确保数据列数与表头一致
    :param parts: 原始数据行
    :param header: 表头
    :return: 修正后的数据行
    """
    if len(parts) == len(header) - 1:
        last_two = parts.pop()
        parts.extend(last_two.rsplit(" ", 1))
    return parts


def log_and_raise(exception, message):
    logger.error(message)
    raise exception(message)


def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        log_and_raise(FileNotFoundError, f"文件不存在: {file_path}")


def validate_file_extension(file_path, expected_extension):
    _, file_extension = os.path.splitext(file_path)
    if file_extension != expected_extension:
        log_and_raise(ValueError, f"文件 {file_path} 不是 {expected_extension} 文件")


def read_pos_data_file(pos_data_file):
    """
    读取相机姿态数据文件，并将其处理成DataFrame格式
    """
    logger.info(f"正在读取相机姿态数据文件: {pos_data_file}")
    validate_file_exists(pos_data_file)
    validate_file_extension(pos_data_file, ".txt")

    try:
        with open(pos_data_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if not lines:
            log_and_raise(ValueError, f"文件 {pos_data_file} 为空")

        header = correct_header(lines[0].strip().split())
        data = [correct_data_row(line.strip().split(), header) for line in lines[1:] if line.strip()]
        df = pd.DataFrame(data, columns=header)
        logger.info(f"成功读取相机姿态数据文件，DataFrame 包含 {len(df)} 行")
        return df

    except Exception as e:
        log_and_raise(Exception, f"读取文件时发生错误: {str(e)}")


def get_panorama_image_path(row, data_paths, group_index):
    """
    获取全景照片的路径

    Args:
        row (pd.Series): DataFrame中的一行数据
        data_paths (dict): 数据路径字典
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
        panorama_image_dir = data_paths["panorama_images"]["files"][group_index - 1]
        panorama_image_path = os.path.join(
            data_paths["panorama_images"]["path"],
            panorama_image_dir,
            panorama_image_name,
        )

        if not os.path.exists(panorama_image_path):
            raise FileNotFoundError(f"全景图像文件不存在: {panorama_image_path}")

        logger.info(f"获取到全景照片路径: {panorama_image_path}")
        return panorama_image_path

    except KeyError:
        raise KeyError("数据中缺少'Filename'列")
    except IndexError:
        raise IndexError(f"数据组索引 {group_index} 超出范围")


def process_single_row(row, group_index, data_paths, pcd):
    """
    处理DataFrame中的单行数据
    """
    logger.info(f"开始处理 DataFrame 中的单行数据，行索引: {row.name}，数据组索引: {group_index}")
    try:
        panorama_image_path = get_panorama_image_path(row, data_paths, group_index)

        # 获取全景相机的平移向量和姿态角
        translation = np.array([float(row["X"]), float(row["Y"]), float(row["Altitude"])])
        rotation_angles = np.radians([float(row["Roll"]), float(row["Pitch"]), float(row["Heading"])])

        logger.info(f"获取到全景相机的平移向量: {translation}")
        logger.info(f"获取到全景相机的姿态角: {rotation_angles}")

        rotation_matrix = get_rotation_matrix(*rotation_angles)
        point_cloud_data = transform_coordinates(pcd, translation, rotation_matrix)

        panorama_image = update_image(panorama_image_path)

        r, phi, theta = cartesian_to_polar(point_cloud_data)
        if r is None or phi is None or theta is None:
            return
        x_p, y_p = calculate_pixel_coordinates(phi, theta, panorama_image)
        h, w = get_color_at_pixel(r, x_p, y_p, panorama_image)

        return (h, w, panorama_image[h, w])  # 返回像素位置和颜色

    except KeyError as e:
        logger.error(f"数据行中缺少必要的列 {e}，请检查数据格式。")
    except (ValueError, IndexError) as e:
        logger.error(f"数据处理过程中出现错误：{e}，请检查数据内容。")


def process_rows_in_parallel(df, group_index, data_paths, pcd):
    """
    并行处理DataFrame中的所有行
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_row, row, group_index, data_paths, pcd): row for index, row in df.iterrows()}
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    results.append(result[2])  # 只取颜色部分
            except Exception as e:
                logger.error(f"处理行时发生错误: {e}")
    return results


def update_image(panorama_path):
    """
    更新全景图像文件
    :param panorama_path: 全景图像路径
    :raises ValueError: 当图像文件无法读取时抛出
    """
    if not os.path.exists(panorama_path):
        raise FileNotFoundError(f"图像文件不存在: {panorama_path}")

    panorama_image = cv2.imread(panorama_path)
    if panorama_image is None:
        raise ValueError(f"无法读取图像文件: {panorama_path}")

    # 进行图像预处理：滤波和增强对比度
    panorama_image = cv2.GaussianBlur(panorama_image, (5, 5), 0)  # 高斯滤波
    panorama_image = cv2.convertScaleAbs(panorama_image, alpha=1.5, beta=0)  # 增强对比度

    logger.info(f"成功加载全景图像: {panorama_path}")
    return panorama_image


def read_and_preprocess_point_cloud(pcd_file_path):
    """
    读取并预处理点云数据
    :raises ValueError: 当点云处理失败时抛出
    """
    config = {
        'voxel_size': 0.1,
        'nb_neighbors': 20,
        'std_ratio': 2.0
    }
    try:
        pcd = o3d.io.read_point_cloud(pcd_file_path)
        if not pcd.has_points():
            logger.warning(f"点云文件中没有点数据: {pcd_file_path}")
            return pcd

        # 使用numpy进行批量处理以提高性能
        points = np.asarray(pcd.points)
        if len(points) > 1000000:  # 对大型点云进行更激进的下采样
            config['voxel_size'] *= 2
            logger.info("检测到大型点云，增加体素大小进行下采样")

        pcd = pcd.voxel_down_sample(voxel_size=config['voxel_size'])
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=config['nb_neighbors'],
            std_ratio=config['std_ratio']
        )
        return pcd
    except Exception as e:
        logger.error(f"点云文件处理失败: {str(e)}")
        raise ValueError(f"点云文件处理失败: {str(e)}")


def get_rotation_matrix(roll, pitch, yaw):
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


def transform_coordinates(pcd, translation, rotation_matrix):
    """
    对点云数据进行坐标转换。
    :return: 转换后的点云对象。
    """
    point_cloud_data = np.asarray(pcd.points)
    # 通过平移和旋转矩阵进行坐标转换
    transformed_data = np.dot(
        (point_cloud_data - translation), rotation_matrix.T
    )
    point_cloud_data = o3d.utility.Vector3dVector(transformed_data)
    return point_cloud_data


def cartesian_to_polar(point_cloud_data):
    """
    将笛卡尔坐标系下的点云数据转换为极坐标系下的坐标。
    :return: 半径 r、方位角 phi、天顶角 theta的元组
    """

    # 使用numpy的批量运算提高性能
    x, y, z = np.asarray(point_cloud_data).T
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # 使用clip避免数值误差
    return r, phi, theta


def calculate_pixel_coordinates(phi, theta, panorama_image, fov_w=2 * np.pi, fov_h=np.pi):
    """
    计算点云在全景图像中对应像素点的坐标。
    :param phi: 方位角
    :param theta: 天顶角
    :param panorama_image: 全景图像
    :param fov_w: 水平视场角，默认值为 2π
    :param fov_h: 垂直视场角，默认值为 π
    :return: 像素 x 坐标和像素 y 坐标
    """
    panorama_width, panorama_height = (
        panorama_image.shape[1],
        panorama_image.shape[0],
    )
    x_p = ((1.295 * np.pi - phi) / fov_w) * panorama_width
    y_p = ((theta) / fov_h) * panorama_height
    return x_p, y_p


def get_color_at_pixel(r, x_p, y_p, panorama_image):
    """
    根据像素坐标获取全景图像中对应点的颜色。
    :param r: 半径
    :param x_p: 像素 x 坐标
    :param y_p: 像素 y 坐标
    :param panorama_image: 全景图像
    :return: 裁剪后的像素 x 坐标和像素 y 坐标
    """
    w = np.clip(x_p.astype(int), 0, panorama_image.shape[1] - 1)
    h = np.clip(y_p.astype(int), 0, panorama_image.shape[0] - 1)
    colors = panorama_image[h, w]

    if r is not None:
        # 归一化 r
        r_min, r_max = r.min(), r.max()
        r_normalized = (r - r_min) / (r_max - r_min) * 0

        # 计算权重，r 越小权重越大
        weights = 1 - r_normalized

        # 将权重应用到颜色上
        colors = (colors * weights[:, np.newaxis]).astype(np.uint8)

    return h, w


def calculate_average_colors_from_rows(rows, pcd):
    """
    计算所有行的颜色平均值并赋值给点云
    """
    colors = []
    for row in rows:
        color_info = process_single_row(row)
        if color_info:
            colors.append(color_info[2])  # 只取颜色部分

    if colors:
        average_color = np.mean(colors, axis=0)
        colors_normalized = average_color.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)


def save_point_cloud(pcd, file_path):
    """
    保存处理后的点云数据到文件。
    :param pcd: 点云数据
    :param file_path: 保存点云的文件路径
    :raises Exception: 当保存过程中发生错误时抛出
    """
    try:
        if pcd is not None:
            o3d.io.write_point_cloud(file_path, pcd)
            logger.info(f"已保存点云到: {file_path}")
        else:
            raise ValueError("点云数据为空，无法保存。")
    except Exception as e:
        logger.error(f"保存点云时出错: {str(e)}")
        raise


def visualize_point_cloud(pcd, save_path=None):
    """
    可视化处理后的点云数据，并可选择保存图像。
    :param pcd: 点云数据
    :param save_path: 保存图像的路径，如果为 None 则不保存
    """
    o3d.visualization.draw_geometries([pcd])

    if save_path:
        # 保存点云图像
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.destroy_window()
        save_point_cloud(pcd, save_path)


def save_processing_results(pcd, group_index, output_dir="output"):
    """
    保存处理结果

    Args:
        pcd (o3d.geometry.PointCloud): 点云数据
        group_index (int): 数据组索引
        output_dir (str): 输出目录

    Raises:
        Exception: 当保存过程中发生错误时
    """
    try:
        # 生成文件名前缀
        prefix = f"group_{group_index}"

        # 使用pathlib处理路径
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "colors").mkdir(exist_ok=True)
        (output_path / "greys").mkdir(exist_ok=True)

        # 保存融合结果
        output_file_path = output_path / f"{prefix}_colored_pointcloud.ply"
        o3d.io.write_point_cloud(str(output_file_path), pcd)
        logger.info(f"已保存第 {group_index} 组数据的处理结果到: {output_file_path}")

    except Exception as e:
        logger.error(f"保存处理结果时发生错误: {str(e)}")
        raise


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='点云和全景图像数据处理工具')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='数据主目录路径 (默认: data)'
    )
    parser.add_argument(
        '--group',
        type=int,
        default=1,
        help='要处理的数据组索引 (默认: 1)'
    )
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 1. 初始化阶段
    args = parse_args()
    data_dir_path = os.path.abspath(args.data_dir)
    validate_file_exists(data_dir_path)

    logger.info(f"开始处理数据，数据目录: {data_dir_path}")
    logger.info(f"处理数据组索引: {args.group}")

    # 2. 数据路径获取阶段
    data_paths = get_data_directory_paths(data_dir_path)
    if not data_paths:
        log_and_raise(ValueError, "未找到有效数据目录")

    pos_data_file = os.path.join(data_paths["camera_pose_data"]["path"],
                                  data_paths["camera_pose_data"]["files"][args.group - 1])
    pcd_file_path = os.path.join(data_paths["lidar_data"]["path"],
                                  data_paths["lidar_data"]["files"][args.group - 1])

    validate_file_exists(pos_data_file)
    validate_file_exists(pcd_file_path)

    # 3. 数据读取与预处理阶段
    df = read_pos_data_file(pos_data_file)
    pcd = read_and_preprocess_point_cloud(pcd_file_path)

    # 4. 数据处理准备阶段
    colors = []

    # 5. 数据逐行处理阶段
    colors = process_rows_in_parallel(df, args.group, data_paths, pcd)

    # 6. 颜色处理与点云赋值阶段
    if colors:
        average_color = np.mean(colors, axis=0)
        colors_normalized = average_color.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    # 7. 结果保存与可视化阶段
    save_processing_results(pcd, args.group)
    visualize_point_cloud(pcd, "data/output/pcd/output.pcd")

    # 8. 结束阶段
    logger.info("数据处理完成")


if __name__ == "__main__":
    main()