import cv2
import numpy as np
from scipy.stats import norm

class SkinExposureDemo:
    def __init__(self):
        self.p_value = 0.98  # 包含概率期望值
        
    def run_demo(self, face_img_path, person_img_path, output_path):
        # 1. 从面部图像建立肤色模型
        face_mask, mu, sigma_inv = self._build_skin_model(face_img_path)
        
        # 2. 在人物图像上检测皮肤区域
        skin_mask = self._detect_skin(mu, sigma_inv, person_img_path)
        
        # 3. 创建演示图像（用半透明灰色覆盖皮肤区域）
        self._create_demo_image(skin_mask, person_img_path, output_path)
        
        print(f"Demo image saved to {output_path}")
    
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
        
        mask = (crcb[:, 0] > 133) & (crcb[:, 0] < 173) & (crcb[:, 1] > 77) & (crcb[:, 1] < 127)
        filtered = crcb[mask]
        
        # 如果没有足够肤色像素，使用备选方案
        if len(filtered) < 10:
            print("警告: 初始肤色过滤不足，使用高频颜色作为备选")
            
            # 使用直方图统计高频CrCb值
            def get_high_freq_colors(pixels, bins=32):
                hist, x_edges, y_edges = np.histogram2d(
                    pixels[:, 0], pixels[:, 1], 
                    bins=bins, 
                    range=[[70, 180], [70, 180]]  # 扩大范围
                )
                max_idx = np.unravel_index(hist.argmax(), hist.shape)
                return np.array([
                    (x_edges[max_idx[0]] + x_edges[max_idx[0]+1])/2,
                    (y_edges[max_idx[1]] + y_edges[max_idx[1]+1])/2
                ])
            
            filtered = np.array([get_high_freq_colors(crcb)])
        
        # 计算均值和协方差
        mu = np.mean(filtered, axis=0)
        
        # 如果样本不足，使用固定小协方差
        if len(filtered) < 2:
            cov = np.eye(2) * 15  # 稍微增大协方差
        else:
            cov = np.cov(filtered.T)
            cov += np.eye(2) * 1e-5  # 稍微增大正则化项
        
        # 计算马氏距离并去除离群点
        if len(filtered) > 5:
            diff = filtered - mu
            mahalanobis = np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)
            threshold = np.percentile(mahalanobis, 97)  # 使用更高的百分位数保留更多点
            inliers = filtered[mahalanobis <= threshold]
        else:
            inliers = filtered
        
        # 最终模型参数
        mu = np.mean(inliers, axis=0)
        if len(inliers) > 1:
            cov = np.cov(inliers.T)
            cov += np.eye(2) * 1e-5  # 稍微增大正则化项
        else:
            cov = np.eye(2) * 15  # 稍微增大协方差
        
        try:
            sigma_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            sigma_inv = np.linalg.pinv(cov)
        
        # 计算阈值 (稍微放宽)
        z = norm.ppf(1 - (1 - self.p_value)/2)
        self.threshold = (z**2) * 1.2  # 增大阈值20%
        
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
        
        # 转换为YCrCb
        ycrcb = cv2.cvtColor(person_rgb, cv2.COLOR_BGR2YCrCb)
        h, w = ycrcb.shape[:2]
        
        # 准备所有坐标点
        crcb = ycrcb[:, :, 1:3].reshape(-1, 2).astype(np.float32)
        
        # 计算马氏距离
        diff = crcb - mu
        mahalanobis = np.sum(diff @ sigma_inv * diff, axis=1)
        
        # 创建皮肤掩模
        skin_mask = (mahalanobis < self.threshold).reshape(h, w)
        
        # 只考虑不透明区域
        if self.person_img.shape[2] == 4:
            skin_mask = skin_mask & (alpha > 0)
        
        # 形态学处理 (使用更大的核)
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask.astype(np.uint8), 
                                   cv2.MORPH_CLOSE, 
                                   kernel).astype(bool)
        
        return skin_mask
    
    def _create_demo_image(self, skin_mask, person_img_path, output_path):
        # 读取原始图像
        person_img = cv2.imread(person_img_path, cv2.IMREAD_UNCHANGED)
        
        # 创建带透明通道的输出图像
        if person_img.shape[2] == 4:
            output_img = person_img.copy()
        else:
            output_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2BGRA)
        
        # 创建半透明灰绿色覆盖层 (BGR = 150, 180, 150)
        overlay = output_img.copy()
        overlay[skin_mask, 0] = 150  # Blue channel
        overlay[skin_mask, 1] = 180  # Green channel
        overlay[skin_mask, 2] = 150  # Red channel
        
        # 混合原始图像和覆盖层 (alpha = 0.5)
        cv2.addWeighted(overlay, 0.5, output_img, 0.5, 0, output_img)
        
        # 保存结果
        cv2.imwrite(output_path, output_img)

if __name__ == "__main__":
    demo = SkinExposureDemo()
    demo.run_demo("./faces2/doubao-seedream-3-0-t2i-250415_doctor_37_20250615_052820.png", 
                 "./persons2/doubao-seedream-3-0-t2i-250415_doctor_37_20250615_052820.png", 
                 "./show.png")