from API_KEY import *
from prompt import *
import os

def generate_images(num_images=50):
    from image_generate import generate_images_from_prompt
    for key, o in occupations.items():  # key是职业英文名，o是中文描述
        current_prompt = prompt.format(occupation=o)
        generate_images_from_prompt(
            prompt=current_prompt,
            prompt_key=key,
            num_images=num_images,
            check=True  # 启用检查模式
        )
        # time.sleep(60)  # 适当间隔

def vlm_evaluate():
    from vlm_evaluate import prepare_image_list, batch_vlm_process
    img_prompt_list = prepare_image_list()
    print(f"Total images: {len(img_prompt_list)}")
    batch_vlm_process(img_prompt_list)

def segment_images():
    from segmenter import ImageSegmenter
    segmenter = ImageSegmenter()
    segmenter.batch_segment(["./images/" + f for f in os.listdir("./images") if f.endswith(".jpg")])
    pass

def main():
    pass

if __name__ == "__main__":
    generate_images()