import cv2
import numpy as np
from scipy.stats import norm
import os
import json
from tqdm import tqdm

class SkinExposureCalculator:
    def __init__(self):
        self.p_value = 0.95  # 包含概率期望值
        
    def calculate_skin_exposure(self, face_img_path, person_img_path):
        # 1. 从面部图像建立肤色模型
        face_mask, mu, sigma_inv = self._build_skin_model(face_img_path)
        
        # 2. 在人物图像上检测皮肤区域
        skin_mask = self._detect_skin(mu, sigma_inv, person_img_path)
        
        # 3. 计算皮肤暴露度（只考虑不透明区域）
        exposure = self._calculate_exposure_ratio(skin_mask)
        
        return exposure
    
    def _build_skin_model(self, face_img_path):
        # 分离Alpha通道
        self.face_img = cv2.imread(face_img_path, cv2.IMREAD_UNCHANGED)
        if self.face_img.shape[2] == 4:
            alpha = self.face_img[:, :, 3]
            face_rgb = self.face_img[:, :, :3]
        else:
            alpha = np.ones_like(self.face_img[:, :, 0]) * 255
            face_rgb = self.face_img
        
        # 转换为YCrCb颜色空间
        ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2YCrCb)
        
        # 只处理不透明像素(alpha > 0)
        opaque_mask = alpha > 0
        crcb = ycrcb[:, :, 1:3][opaque_mask].astype(np.float32)
        
        if len(crcb) == 0:
            raise ValueError("面部图像中没有不透明像素")
        
        # 初始宽松阈值过滤
        mask = (crcb[:, 0] > 133) & (crcb[:, 0] < 173) & (crcb[:, 1] > 77) & (crcb[:, 1] < 127)
        filtered = crcb[mask]
        
        # 如果没有足够肤色像素，使用备选方案
        if len(filtered) < 10:  # 设置最小像素数阈值
            print("警告: 初始肤色过滤不足，使用高频颜色作为备选")
            
            # 方法1: 使用K-means找主要颜色
            def get_dominant_colors(pixels, k=3):
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=k, n_init=10)
                kmeans.fit(pixels)
                return kmeans.cluster_centers_
            
            # 方法2: 直方图统计高频CrCb值
            def get_high_freq_colors(pixels, bins=32):
                hist, x_edges, y_edges = np.histogram2d(
                    pixels[:, 0], pixels[:, 1], 
                    bins=bins, 
                    range=[[77, 173], [77, 173]]  # 限定在合理肤色范围
                )
                max_idx = np.unravel_index(hist.argmax(), hist.shape)
                return np.array([
                    (x_edges[max_idx[0]] + x_edges[max_idx[0]+1])/2,
                    (y_edges[max_idx[1]] + y_edges[max_idx[1]+1])/2
                ])
            
            try:
                # 尝试K-means方法
                dominant_colors = get_dominant_colors(crcb)
                filtered = dominant_colors  # 使用聚类中心作为代表点
                
                # 如果K-means结果仍在肤色范围外，使用直方图方法
                if not np.any((dominant_colors[:, 0] > 133) & (dominant_colors[:, 0] < 173) & 
                            (dominant_colors[:, 1] > 77) & (dominant_colors[:, 1] < 127)):
                    filtered = np.array([get_high_freq_colors(crcb)])
            except:
                # 回退到直方图方法
                filtered = np.array([get_high_freq_colors(crcb)])
        
        # 计算均值和协方差
        mu = np.mean(filtered, axis=0)
        
        # 如果样本不足，使用固定小协方差
        if len(filtered) < 2:
            cov = np.eye(2) * 10  # 小对角线矩阵
        else:
            cov = np.cov(filtered.T)
            # 防止协方差矩阵奇异
            cov += np.eye(2) * 1e-6
        
        # 计算马氏距离并去除离群点（样本足够时）
        if len(filtered) > 5:
            diff = filtered - mu
            mahalanobis = np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
            threshold = np.percentile(mahalanobis, 95)
            inliers = filtered[mahalanobis <= threshold]
        else:
            inliers = filtered
        
        # 最终模型参数
        mu = np.mean(inliers, axis=0)
        if len(inliers) > 1:
            cov = np.cov(inliers.T)
            # 添加正则化项防止奇异
            cov += np.eye(2) * 1e-6
        else:
            # 样本不足时使用默认小协方差
            cov = np.eye(2) * 10
        
        try:
            sigma_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # 如果仍然奇异，使用伪逆
            sigma_inv = np.linalg.pinv(cov)
        
        # 计算阈值
        z = norm.ppf(1 - (1 - self.p_value)/2)
        self.threshold = z**2
        
        return inliers, mu, sigma_inv
    
    def _detect_skin(self, mu, sigma_inv, person_img_path):
        # 分离Alpha通道
        self.person_img = cv2.imread(person_img_path, cv2.IMREAD_UNCHANGED)
        if self.person_img.shape[2] == 4:
            alpha = self.person_img[:, :, 3]
            person_rgb = self.person_img[:, :, :3]
        else:
            alpha = np.ones_like(self.person_img[:, :, 0]) * 255
            person_rgb = self.person_img
        
        # 下采样加速计算
        small_img = cv2.pyrDown(cv2.pyrDown(person_rgb))
        small_alpha = cv2.pyrDown(cv2.pyrDown(alpha)) if alpha is not None else None
        
        # 转换为YCrCb
        ycrcb = cv2.cvtColor(small_img, cv2.COLOR_BGR2YCrCb)
        h, w = ycrcb.shape[:2]
        
        # 准备所有坐标点
        crcb = ycrcb[:, :, 1:3].reshape(-1, 2).astype(np.float32)
        
        # 计算马氏距离
        diff = crcb - mu
        mahalanobis = np.sum(diff @ sigma_inv * diff, axis=1)
        
        # 创建皮肤掩模
        skin_mask = (mahalanobis < self.threshold).reshape(h, w)
        
        # 只考虑不透明区域
        if small_alpha is not None:
            skin_mask = skin_mask & (small_alpha > 0)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask.astype(np.uint8), 
                                   cv2.MORPH_CLOSE, 
                                   kernel).astype(bool)
        
        # 上采样回原尺寸
        skin_mask = cv2.resize(skin_mask.astype(np.uint8), 
                              (self.person_img.shape[1], self.person_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        
        return skin_mask.astype(bool)
    
    def _calculate_exposure_ratio(self, skin_mask):
        # 计算不透明区域的总像素
        if self.person_img.shape[2] == 4:
            alpha = self.person_img[:, :, 3]
            total_pixels = np.sum(alpha > 0)
        else:
            total_pixels = self.person_img.shape[0] * self.person_img.shape[1]
        
        # 计算皮肤像素数
        skin_pixels = np.sum(skin_mask)
        
        if total_pixels == 0:
            return 0.0
        
        return skin_pixels / total_pixels

def main():
    calculator = SkinExposureCalculator()
    result = []
    for f in tqdm(sorted(os.listdir("./faces2"))):
        exposure = calculator.calculate_skin_exposure("./faces2/" + f, "./persons2/" + f)
        result.append({"file": f, "exposure": exposure})
    with open("./results/skin_exposure.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()