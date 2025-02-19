import os
import argparse
import logging
from DataProcessor import DataProcessor

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    # 解析命令行参数
    args = parse_args()
    logger = logging.getLogger(__name__)
    
    try:
        # 构建并验证数据目录路径
        data_dir_path = os.path.abspath(args.data_dir)
        if not os.path.exists(data_dir_path):
            raise FileNotFoundError(f"数据目录不存在: {data_dir_path}")
            
        logger.info(f"开始处理数据，数据目录: {data_dir_path}")
        logger.info(f"处理数据组索引: {args.group}")

        # 创建数据处理器实例
        processor = DataProcessor(data_dir_path)

        # 处理数据
        processor.process_data(args.group)
        
        logger.info("数据处理完成")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()