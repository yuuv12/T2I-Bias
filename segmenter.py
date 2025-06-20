import os
import cv2
import numpy as np
from pathlib import Path
from typing import List
from ultralytics import YOLO
from deepface import DeepFace
import torch
from tqdm import tqdm
import traceback

tqdm.pandas(desc="Processing DataFrame")

# 确保使用GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 创建输出目录
os.makedirs("./results", exist_ok=True)

class ImageSegmenter:
    """人物分割模块 (批量处理)"""
    def __init__(self):
        try:
            self.model = YOLO("yolo11n-seg.pt").to(device)
            self.output_person = "./persons"
            self.output_face = "./faces"

            os.makedirs(self.output_person, exist_ok=True)
            os.makedirs(self.output_face, exist_ok=True)
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise

    def batch_segment(self, image_paths: List[str], return_face_location=True):
        """批量分割人物和人脸"""
        with tqdm(image_paths, desc="Processing images") as pbar:
            for img_path in pbar:
                try:
                    pbar.set_postfix(file=os.path.basename(img_path))
                    self.segment_sigle(img_path, return_face_location)
                except Exception as e:
                    print(f"\n处理图像 {img_path} 时出错: {str(e)}")
                    traceback.print_exc()  # 打印完整堆栈信息
                    continue  # 继续处理下一个图像
    
    def segment_sigle(self, image_path: str, return_face_location=True):
        faces_result = None
        faces_result = self._segment_individual_faces(image_path, return_face_location=return_face_location)
        self._segment_individual_persons(image_path, faces_result)

    def _segment_individual_persons(self, image_path: str, faces=None):
        """优化后的人物分割方法，当faces为None时直接保存合格人物，否则关联人脸并检查重叠"""
        try:
            # 输入验证
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像文件: {image_path}")

            # 获取图像总面积和最小人物面积阈值
            img_height, img_width = image.shape[:2]
            total_image_area = img_width * img_height
            min_person_area = total_image_area * 0.2  # 最小人物区域面积阈值

            # 初始化变量
            results = self.model(image, device=device, verbose=False)
            output_count = 0  # 输出计数

            # 处理每个人物检测结果
            for r in results:
                if not hasattr(r, 'masks') or r.masks is None:
                    continue
                    
                for j, mask in enumerate(r.masks.xy):
                    if int(r.boxes.cls[j].item()) != 0:  # 只处理人物类别
                        continue

                    try:
                        # 创建人物掩码和轮廓
                        mask = np.array(mask, dtype=np.int32)
                        person_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(person_mask, [mask], 255)
                        
                        contours = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours = contours[0] if len(contours) == 2 else contours[1]
                        if not contours:
                            continue
                        
                        main_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(main_contour)
                        person_bbox_area = w * h

                        # 跳过面积不足的人物
                        if person_bbox_area < min_person_area:
                            continue

                        # === 核心修改部分 ===
                        if faces is None:
                            # 无face参数时直接保存合格人物
                            transparent = np.zeros((h, w, 4), dtype=np.uint8)
                            person_roi = image[y:y+h, x:x+w]
                            mask_roi = person_mask[y:y+h, x:x+w]
                            transparent[..., :3] = person_roi
                            transparent[..., 3] = mask_roi
                            
                            stem = Path(image_path).stem
                            cv2.imwrite(f"{self.output_person}/person-{output_count}-{stem}.png", transparent)
                            output_count += 1
                            continue  # 跳过后续人脸关联逻辑

                        # === 有face参数时的完整处理流程 ===
                        person_regions = []  # 存储已保存人物区域（用于重叠检查）
                        saved_faces = set()  # 已保存的人脸索引
                        face_data = []

                        # 预处理人脸数据
                        for i, face in enumerate(faces):
                            left_eye = face["facial_area"]["left_eye"]
                            right_eye = face["facial_area"]["right_eye"]
                            face_data.append({
                                'index': i,
                                'left_eye': left_eye,
                                'right_eye': right_eye,
                                'face_img': face["face"]
                            })

                        # 关联人脸
                        associated_faces = []
                        for face in face_data:
                            left_in = cv2.pointPolygonTest(main_contour, tuple(face['left_eye']), False) >= 0
                            right_in = cv2.pointPolygonTest(main_contour, tuple(face['right_eye']), False) >= 0
                            if left_in and right_in:
                                associated_faces.append(face['index'])

                        # 跳过无人脸的人物
                        if not associated_faces:
                            continue

                        # 检查区域重叠（仅在有face时启用）
                        current_area = cv2.contourArea(main_contour)
                        should_skip = False
                        for saved in person_regions:
                            intersection_mask = cv2.bitwise_and(person_mask, saved['mask'])
                            intersection_area = cv2.countNonZero(intersection_mask)
                            if (intersection_area > 0.8 * current_area) or (intersection_area > 0.8 * saved['area']):
                                should_skip = True
                                break
                        if should_skip:
                            continue

                        # 保存人物和关联人脸
                        transparent = np.zeros((h, w, 4), dtype=np.uint8)
                        person_roi = image[y:y+h, x:x+w]
                        mask_roi = person_mask[y:y+h, x:x+w]
                        transparent[..., :3] = person_roi
                        transparent[..., 3] = mask_roi
                        
                        stem = Path(image_path).stem
                        cv2.imwrite(f"{self.output_person}/person-{output_count}-{stem}.png", transparent)

                        for face_idx in associated_faces:
                            if face_idx not in saved_faces:
                                face_img = faces[face_idx]["face"].astype(np.uint8)
                                cv2.imwrite(f"{self.output_face}/face-{output_count}-{stem}.png", face_img)
                                saved_faces.add(face_idx)

                        # 记录已保存区域
                        person_regions.append({
                            'mask': person_mask,
                            'contour': main_contour,
                            'area': current_area,
                            'bbox': (x, y, w, h)
                        })
                        output_count += 1

                    except Exception as e:
                        print(f"处理 {image_path} 中第 {j} 个人物时出错: {str(e)}")
                        continue
            
            if output_count == 0:
                print(f"图像 {image_path} 中未检测到有效人物")

        except Exception as e:
            print(f"处理人物分割 {image_path} 时出错: {str(e)}")
            raise
    def _segment_individual_faces(self, image_path: str, return_face_location=True):
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            # 人脸检测
            try:
                detections = DeepFace.extract_faces(
                    image_path, 
                    detector_backend="retinaface", 
                    normalize_face=False, 
                    color_face="bgr", 
                    enforce_detection=False
                )
            except ValueError as e:
                if "Face could not be detected" in str(e):
                    print(f"{image_path} 中未检测到人脸")
                    return
                raise
            if return_face_location:
                if not detections:
                    return
                return detections
            idx = 0
            
            if not detections:
                print(f"{image_path} 中未检测到人脸")
            for img in detections:
                try:
                    if not return_face_location:
                        output_path = f"{self.output_face}/face-{idx}-{Path(image_path).stem}.png"
                        cv2.imwrite(output_path, (img["face"]).astype(np.uint8))
                        idx += 1
                    else:
                        return 
                except Exception as e:
                    print(f"保存 {image_path} 的第 {idx} 张人脸时出错: {str(e)}")
                    continue
            if idx == 0:
                print(f"{image_path} 中未检测到人脸")
        except Exception as e:
            print(f"处理人脸检测 {image_path} 时出错: {str(e)}")
            raise  # 重新抛出异常以便batch_segment捕获

if __name__ == "__main__":
    segmenter = ImageSegmenter()
    existing = [f[7:][:-4] for f in os.listdir("./faces1") if f.endswith(".png")]
    need_process = []
    for img_path in [f[:-4] for f in os.listdir("./images") if f.endswith(".jpg")]:
        if img_path not in existing:
            need_process.append(img_path)
    need_process = ["./images/" + f + ".jpg" for f in need_process]
    segmenter.batch_segment(need_process, False)
            
        