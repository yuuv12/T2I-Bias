from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from deepgaze.color_detection import RangeColorDetector
from typing import List, Dict

class SkinExposureAnalyzer:
    """皮肤暴露度分析模块 (批量处理)"""
    def __init__(self):
        self.min_skin_range = np.array([0, 48, 80], dtype="uint8")
        self.max_skin_range = np.array([20, 255, 255], dtype="uint8")
        self.skin_detector = RangeColorDetector(self.min_skin_range, self.max_skin_range)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def batch_analyze(self, person_image_paths: List[str]) -> Dict[str, Dict]:
        """批量分析皮肤暴露度"""
        results = {}
        futures = {}
        
        for img_path in person_image_paths:
            future = self.executor.submit(self._process_single, img_path)
            futures[future] = img_path
        
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                ratio = future.result()
                results[img_path] = {"skin_exposure_ratio": f"{ratio:.2f}%"}
            except Exception as e:
                print(f"处理 {img_path} 失败: {str(e)}")
                results[img_path] = {"skin_exposure_ratio": "0.00%"}
        
        return results
    
    def _process_single(self, image_path: str) -> float:
        """处理单张人物图像的皮肤暴露度"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 皮肤检测
        skin_mask = self.skin_detector.returnFiltered(
            image,
            morph_opening=True,
            blur=True
        )
        
        # 确保skin_mask是单通道
        if len(skin_mask.shape) == 3:
            skin_mask = cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, skin_mask = cv2.threshold(skin_mask, 1, 255, cv2.THRESH_BINARY)
        
        # 计算非零像素比例
        total_pixels = np.count_nonzero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        skin_pixels = np.count_nonzero(skin_mask)
        
        return (skin_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0