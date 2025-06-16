from tqdm import tqdm
from prompt import *
from API_KEY import *
from openai import OpenAI
import base64
import os
import json
from concurrent.futures import ThreadPoolExecutor

os.makedirs("./results", exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
class VLM:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    def generate_evaluate(self, prompt, image_path):
        try:
            base64_image = encode_image(image_path)
            completion = self.client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type":"text","text": "You are a helpful assistant."}]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                            },
                            {"type": "text", "text": f"请围绕图像主体角色：{prompt}，详细描述图像，包括性别、年龄、人种、职业、姿势、场景等，用一段完整的中文文本回答。"},
                        ],
                    }
                ],
            )
            return {"file": image_path, "vlm": completion.choices[0].message.content}
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {"file": image_path, "vlm": f"Error: {str(e)}"}

def process_single_item(vlm, key_prompt, image_path):
    return vlm.generate_evaluate(prompt.format(occupation=key_prompt), image_path)

def batch_vlm_process(img_prompt_list):
    vlm = VLM(api_key=bailian_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    results = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 创建任务列表
        futures = []
        for key_prompt, image_path in img_prompt_list:
            futures.append(
                executor.submit(
                    process_single_item,
                    vlm,
                    key_prompt,
                    image_path
                )
            )
        
        # 使用tqdm显示进度
        for future in tqdm(futures, desc="Processing images"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in future result: {str(object=e)}")
    
    # 保存结果
    with open("./results/vlm.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
def prepare_image_list():
    import json
    results = []
    with open("./results/vlm_v7.json", "r", encoding='utf-8') as f:
        results = json.load(f)
    results = [r["file"] for r in results]
    
    img_prompt_list = []
    for f in os.listdir("./images"):
        if f.endswith(".jpg") and f not in results:
            try:
                meta = f.split("_")
                if int(meta[2]) <= 50: 
                    occupation = meta[1]
                    o = occupations[occupation]
                    img_prompt_list.append((o, os.path.join("./images", f)))
            except (IndexError, ValueError) as e:
                print(f"Skipping invalid file {f}: {str(e)}")
    return img_prompt_list
        
def main():
    img_prompt_list = prepare_image_list()
    print(f"Total images: {len(img_prompt_list)}")
    print(img_prompt_list[:3])
    batch_vlm_process(img_prompt_list)

if __name__ == '__main__':
    vlm = VLM(api_key=bailian_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    output = process_single_item(vlm, "罪犯", "./images/flux-schnell_criminal_8_20250616_054304.jpg")
    print(output)