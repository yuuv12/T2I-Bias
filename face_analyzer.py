from deepface import DeepFace
import os
from tqdm import tqdm
def analyze_face(image_path, actions=["age", "gender", "emotion", "race"]):
    """
    分析图像中的主脸
    参数:
        image_path: str - 图像文件路径
        actions: list - 要分析的特征列表
        
    返回:
        dict - 分析结果
    """
    try:
        # 仅分析主角
        result = DeepFace.analyze(
            img_path=image_path,
            actions=actions,
            detector_backend="skip",
            enforce_detection=False,
            align=True
        )
        return result[0]

    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None

def batch_analyze_face(image_paths, actions=["age", "gender", "emotion", "race"]):
    """
    批量分析图像中的主脸
    参数:
        image_paths: list - 图像文件路径列表
        actions: list - 要分析的特征列表
        
    返回:
        list - 分析结果列表
    """
    results = []
    for image_path in tqdm(image_paths):
        result = analyze_face(image_path, actions)
        results.append({
            "image_path": image_path,
            "age": result['age'],
            "gender": result['dominant_gender'],
            "emotion": result['dominant_emotion'],
            "race": result['dominant_race']
            })
    return results

def main():
    import json
    lis = ["./faces2/"+_ for _ in os.listdir("./faces2/")]
    print(lis[:3])
    result = batch_analyze_face(["./faces2/"+_ for _ in os.listdir("./faces2/")])
    with open("./results/face_anlyze.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()