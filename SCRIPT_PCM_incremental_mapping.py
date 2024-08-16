__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
from pathlib import Path
import pycolmap

LIST_db_files = [
    Path("E:\Parkings\L-Loop\scene-2\compo-1-OMAP\compo-1-OMAP.db"),

]


def colmap_incremental_image_folder(dp_images, dp_output, fp_db):
    """
    Incremental Mapping with COLMAP
    :param dp_images: Path to the directory containing images
    :param dp_output: Path to the output directory
    :param fp_db: Path to the COLMAP database
    :return: None
    """
    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
