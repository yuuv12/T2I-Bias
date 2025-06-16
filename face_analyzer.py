from deepface import DeepFace

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
    for image_path in image_paths:
        result = analyze_face(image_path, actions)
        results.append({
            "image_path": image_path[15:],
            "age": result['age'],
            "gender": result['dominant_gender'],
            "emotion": result['dominant_emotion'],
            "race": result['dominant_race']
            })
    return results

def main():
    print(batch_analyze_face(["./faces/face-0-doubao-seedream-3-0-t2i-250415_coach_1_20250615_044702.png"]))
    pass

if __name__ == '__main__':
    main()