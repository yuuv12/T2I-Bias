import os
import time
from datetime import datetime
from PIL import Image
from PIL.ImageFile import ImageFile
import requests
import jwt
import io
from abc import ABC, abstractmethod
import base64
from concurrent.futures import ThreadPoolExecutor
from prompt import *
from API_KEY import *

def generate_with_model(model_name, generator, prompt, num_images, results):
    """单个模型的生成线程函数"""
    try:
        start_time = time.time()
        print(f"[{model_name}] 开始生成{num_images}张图片...")
        
        # 特殊处理Kling模型（异步API）
        saved_files = generator.generate_and_save(
            prompt=prompt,
            num_images=num_images,
        )
        
        elapsed_time = time.time() - start_time
        results[model_name] = {
            "files": saved_files,
            "time": f"{elapsed_time:.2f}秒"
        }
        print(f"[{model_name}] 生成完成 (耗时 {elapsed_time:.2f}秒)")
        
    except Exception as e:
        print(f"[{model_name}] 生成失败: {str(e)}")
        results[model_name] = {"error": str(e)}

def generate_images_from_prompt(prompt, prompt_key=None, num_images=50, check=False):
    """并发调用所有模型生成图片"""
    os.makedirs("./images", exist_ok=True)
    
    # 初始化所有模型
    models = {
        # "豆包": DouBao(api_key=fangzhou_api_key),
        # "StableDiffusion": SD(api_key=bailian_api_key),
        "FLUX": FLUX(api_key=bailian_api_key),
        # "WANX": WAN(api_key=bailian_api_key),
        # "Kling": KLING(keling_access_key, keling_secret_key)
    }
    
    need_to_generate = {k:num_images for k in models.keys()}
    
    if check:
        print("="*50)
        print("检查本地已生成文件...")
        for model_name, model in models.items():
            # 查找该模型该prompt_key已生成的图片
            prefix = f"{model.model_name}_{prompt_key}"
            count = len([f for f in os.listdir("./images") if f.startswith(prefix)])
            need_to_generate[model_name] = 50-count
            print(f"[{model_name}] 已生成: {count}/{num_images}")
    
    # 使用线程池并发执行
    results = {}
    threads = []
    
    print("="*50)
    print(f"开始并发生成图片（提示词：'{prompt}'，每个模型生成到{num_images}张）")
    print("="*50 + "\n")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        for model_name, generator in models.items():
            thread = executor.submit(
                generate_with_model,
                model_name, generator, prompt, need_to_generate[model_name], results
            )
            threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.result()
    
    # 打印汇总结果
    print("\n" + "="*50)
    print("所有模型生成任务完成！汇总结果：")
    print("="*50)
    for model_name, result in results.items():
        if "files" in result:
            print(f"[{model_name}] 成功生成 {len(result['files'])} 张图片 | 耗时 {result['time']}")
            print(f"    保存位置: {result['files']}")
        else:
            print(f"[{model_name}] 生成失败 ❌ | 错误: {result['error']}")
    for model_name, result in results.items():
        if len(result["files"]) < need_to_generate[model_name]:
            generate_images_from_prompt(prompt, prompt_key, num_images, check)
    
    return results



class ImageGenerator(ABC):
    """图像生成器抽象基类"""
    model_name: str
    
    @abstractmethod
    def generate(self, prompt, **kwargs):
        """生成图像抽象方法"""
        pass

    @staticmethod
    def save_image(model_name, image_data, prompt, **kwargs):
        """
        保存图像到本地
        
        参数:
            image_data: dict - 包含图像数据的字典
            prompt: str - 生成图像的提示词
            
        返回:
            str - 保存的文件路径
        """
        def _process_prompt(text: str) -> str:
            text = text.replace(" ", "_")[:50].strip("一名")
            for k, v in occupations.items():
                text = text.replace(v, k)
            return text
        
        def _get_existing_files(prompt_prefix: str) -> list[str]:
            """获取已存在的文件列表"""
            return [f for f in os.listdir("./images") 
                    if f.startswith(f"{model_name}_{prompt_prefix}")]
        
        def _download_image(url: str) -> ImageFile:
            """从URL下载图像"""
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        
        def _decode_base64_image(b64_data: str) -> ImageFile:
            """解码Base64图像数据"""
            return Image.open(io.BytesIO(base64.b64decode(b64_data)))

        try:
            prompt = _process_prompt(prompt)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            existing_files = _get_existing_files(prompt)
            file_num = len(existing_files) + 1
            
            # 构建文件名
            filename = f"./images/{model_name}_{prompt}_{file_num}_{current_time}.jpg"
            
            if image_data["format"] == "url":
                image = _download_image(image_data["data"])
            else:
                image = _decode_base64_image(image_data["data"])
            
            image.save(filename, "JPEG", quality=95)
            print(f"成功保存: {filename}")
            
            return filename
            
        except Exception as e:
            print(f"保存图像时出错: {str(e)}")
            return None

    def generate_and_save(self, prompt, num_images=1, **kwargs):
        """
        批量生成并保存图像
        
        参数:
            prompt: str - 提示词
            num_images: int - 每个提示词生成的图片数量
            kwargs: dict - 其他生成参数
            
        返回:
            list - 保存的文件路径列表
        """
        saved_files = []
        
        images = self.generate(prompt, num_images=num_images, **kwargs)
        for img_data in images:
            filepath = self.save_image(self.model_name, img_data, 
                                     img_data.get("prompt", prompt))
            if filepath:
                saved_files.append(filepath)
        
        return saved_files

# Implementation Classes
class DouBao(ImageGenerator):
    """豆包图像生成器实现"""
    
    def __init__(self, api_key=None, base_url="https://ark.cn-beijing.volces.com/api/v3"):
        """
        初始化豆包图像生成器
        
        参数:
            api_key: str - 火山引擎API Key
            base_url: str - API基础URL
        """
        from volcenginesdkarkruntime import Ark
        self.client = Ark(
            base_url=base_url,
            api_key=api_key or os.environ.get("ARK_API_KEY")
        )
        self.model_name = "doubao-seedream-3-0-t2i-250415"
    
    def generate(self, prompt, num_images=1, size="1024x1024", guidance_scale=2.5, 
                 watermark=False, response_format="url", **kwargs):
        """
        生成图像
        
        参数:
            prompt: str - 生成图像的提示词
            num_images: int - 生成图像数量
            size: str - 图像尺寸
            guidance_scale: float - 指导比例
            watermark: bool - 是否添加水印
            response_format: str - 返回格式(url或b64_json)
            
        返回:
            list - 生成的图像数据列表
        """
        images = []
        
        for _ in range(num_images):
            try:
                response = self.client.images.generate(
                    model=self.model_name,
                    prompt=prompt,
                    size=size,
                    guidance_scale=guidance_scale,
                    watermark=watermark,
                    response_format=response_format
                )
                # 根据返回格式处理数据
                if response_format == "url":
                    image_data = response.data[0].url
                else:
                    image_data = response.data[0].b64_json
                images.append({
                    "data": image_data,
                    "format": response_format,
                    "prompt": prompt,
                    "model": self.model_name,
                })
            except Exception as e:
                print(f"Model:{self.model_name} 生成图像时出错 (提示词: '{prompt}'): {str(e)}")
        
        return images

class SD(ImageGenerator):
    """SD 图像生成器实现 (基于阿里云百炼 StableDiffusion 3.5)"""
    
    def __init__(self, api_key=None, model_version="large-turbo"):
        """
        初始化SD图像生成器
        
        参数:
            api_key: str - 阿里云API Key
            model_version: str - 模型版本 ('large' 或 'large-turbo')
        """
        from dashscope import ImageSynthesis
        self.client = ImageSynthesis
        self.model_name = f"stable-diffusion-3.5-{model_version}"
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    
    def generate(self, prompt, num_images=1, size="1024*1024", negative_prompt=None, response_format="url", **kwargs):
        """
        生成图像
        
        参数:
            prompt: str - 生成图像的提示词
            num_images: int - 生成图像数量
            size: str - 图像尺寸 (格式: '宽*高')
            negative_prompt: str - 负面提示词
            response_format: str - 返回格式(url或b64_json)
            
        返回:
            list - 生成的图像数据列表
        """
        images = []
        max_batch_size = 4  # 每次请求最多生成4张图
        
        try:
            remaining = num_images
            while remaining > 0:
                current_batch = min(remaining, max_batch_size)
                
                response = self.client.call(
                    model=self.model_name,
                    api_key=self.api_key,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    n=current_batch,
                    size=size
                )
                
                if response.status_code == 200:
                    for result in response.output.results:
                        images.append({
                            "data": result.url,
                            "format": "url",
                            "prompt": prompt,
                            "model": self.model_name,
                        })
                    remaining -= current_batch
                else:
                    print(f"Model:{self.model_name} 生成图像失败 (提示词: '{prompt}'): {response.message}")
                    break
        
        except Exception as e:
            print(f"Model:{self.model_name} 生成图像时出错 (提示词: '{prompt}'): {str(e)}")
        
        return images
class FLUX(ImageGenerator):
    """FLUX 图像生成器实现 (基于阿里云FLUX文生图模型)"""
    
    def __init__(self, api_key=None, model_version="schnell"):
        """
        初始化FLUX图像生成器
        
        参数:
            api_key: str - 阿里云API Key
            model_version: str - 模型版本 ('schnell', 'dev' 或 'merged')
        """
        from dashscope import ImageSynthesis
        self.client = ImageSynthesis
        self.model_name = f"flux-{model_version}"
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    
    def generate(self, prompt, num_images=1, size="1024*1024", seed=None, 
                 steps=None, guidance=3.5, offload=False, add_sampling_metadata=True,
                 response_format="url", **kwargs):
        """
        生成图像
        
        参数:
            prompt: str - 生成图像的提示词
            num_images: int - 生成图像数量
            size: str - 图像尺寸
            steps: int - 推理步数
            guidance: float - 指导度量值
            offload: bool - 是否启用CPU卸载
            add_sampling_metadata: bool - 是否在图像中嵌入元数据
            response_format: str - 返回格式(url)
            
        返回:
            list - 生成的图像数据列表
        """
        images = []
        max_batch_size = 4  # 每次请求最多生成4张图
        
        try:
            remaining = num_images
            while remaining > 0:
                current_batch = min(remaining, max_batch_size)
                
                response = self.client.call(
                    model=self.model_name,
                    prompt=prompt,
                    n=current_batch,
                    size=size,
                    steps=steps,
                    guidance=guidance,
                    offload=offload,
                    add_sampling_metadata=add_sampling_metadata,
                    api_key=self.api_key
                )
                
                if response.status_code == 200:
                    for result in response.output.results:
                        images.append({
                            "data": result.url,
                            "format": response_format,
                            "prompt": prompt,
                            "model": self.model_name,
                            "metadata": {
                                "size": size,
                                "steps": steps,
                                "guidance": guidance
                            }
                        })
                    remaining -= current_batch
                else:
                    print(f"Model:{self.model_name} 生成图像失败 (提示词: '{prompt}'): {response.message}")
                    break
        
        except Exception as e:
            print(f"Model:{self.model_name} 生成图像时出错 (提示词: '{prompt}'): {str(e)}")
        
        return images

class WAN(ImageGenerator):
    """WANX 图像生成器实现 (基于阿里云WANX文生图V2 API)"""
    
    def __init__(self, api_key=None, model_version="2.1-t2i-plus"):
        """
        初始化WANX图像生成器
        
        参数:
            api_key: str - 阿里云API Key
            model_version: str - 模型版本 ('2.1-t2i-turbo'等)
        """
        from dashscope import ImageSynthesis
        self.client = ImageSynthesis
        self.model_name = f"wanx{model_version}"
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    
    def generate(self, prompt, num_images=1, size="1024*1024", seed=None,
                 negative_prompt=None, prompt_extend=False, watermark=False,
                 response_format="url", **kwargs):
        """
        同步生成图像
        
        参数:
            prompt: str - 正向提示词
            num_images: int - 生成图像数量 (1-4)
            size: str - 图像尺寸
            negative_prompt: str - 反向提示词
            prompt_extend: bool - 是否开启prompt智能改写
            watermark: bool - 是否添加水印
            response_format: str - 返回格式(url)
            
        返回:
            list - 生成的图像数据列表
        """
        images = []
        max_batch_size = 4  # 每次请求最多生成4张图
        
        try:
            remaining = num_images
            while remaining > 0:
                current_batch = min(remaining, max_batch_size)
                
                response = self.client.call(
                    model=self.model_name,
                    api_key=self.api_key,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    n=current_batch,
                    size=size,
                    prompt_extend=prompt_extend,
                    watermark=watermark
                )
                
                if response.status_code == 200:
                    for result in response.output.results:
                        images.append({
                            "data": result.url,
                            "format": response_format,
                            "prompt": result.get('actual_prompt', prompt),
                            "model": self.model_name,
                            "metadata": {
                                "size": size,
                                "orig_prompt": result.get('orig_prompt', prompt)
                            }
                        })
                    remaining -= current_batch
                else:
                    print(f"Model:{self.model_name} 生成图像失败 (提示词: '{prompt}'): {response.message}")
                    break
        
        except Exception as e:
            print(f"Model:{self.model_name} 生成图像时出错 (提示词: '{prompt}'): {str(e)}")
        
        return images

class KLING(ImageGenerator):
    """Kling 图像生成器实现 (基于昆仑万维Kling API)"""
    
    def __init__(self, access_key=None, secret_key=None, model_version="v1"):
        """
        初始化Kling图像生成器
        
        参数:
            access_key: str - Access Key
            secret_key: str - Secret Key
            model_version: str - 模型版本 ('v1', 'v1-5' 或 'v2')
        """
        self.access_key = access_key or os.environ.get("KLING_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("KLING_SECRET_KEY")
        self.base_url = "https://api-beijing.klingai.com/v1/images/generations"
        self.model_name = f"kling-{model_version}"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": self._generate_auth_token()
        }
    
    def _generate_auth_token(self):
        """生成JWT鉴权token"""
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,  # 30分钟有效期
            "nbf": int(time.time()) - 5      # 5秒前生效
        }
        token = jwt.encode(payload, self.secret_key, headers=headers)
        return f"Bearer {token}"
    
    def generate(self, prompt, num_images=1, negative_prompt=None,
                 image=None, image_reference=None, image_fidelity=0.5, 
                 human_fidelity=0.45, resolution="1k", aspect_ratio="16:9", 
                 callback_url=None, **kwargs):
        """
        同步生成图像
        
        参数:
            prompt: str - 正向提示词(不超过2500字符)
            num_images: int - 生成图像数量 (1-9)
            negative_prompt: str - 负向提示词(不超过2500字符)
            image: str - 参考图像(Base64编码或URL)
            image_reference: str - 图片参考类型('subject'或'face')
            image_fidelity: float - 图像参考强度(0-1)
            human_fidelity: float - 面部参考强度(0-1)
            resolution: str - 清晰度('1k'或'2k')
            aspect_ratio: str - 画面纵横比
            callback_url: str - 回调通知地址
            
        返回:
            list - 生成的图像数据列表
        """
        images = []
        max_batch_size = 6
        
        try:
            remaining = num_images
            while remaining > 0:
                current_batch = min(remaining, max_batch_size)
                
                # 更新headers确保token是最新的
                self.headers["Authorization"] = self._generate_auth_token()
                
                payload = {
                    "model_name": self.model_name,
                    "prompt": prompt,
                    "n": current_batch,
                    "negative_prompt": negative_prompt,
                    "image": image,
                    "image_reference": image_reference,
                    "image_fidelity": image_fidelity,
                    "human_fidelity": human_fidelity,
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio,
                    "callback_url": callback_url
                }
                
                # 移除None值
                payload = {k: v for k, v in payload.items() if v is not None}
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                if result["code"] != 0:
                    print(f"Model:{self.model_name} 生成图像失败: {result['message']}")
                    break
                
                # 等待任务完成
                task_id = result["data"]["task_id"]
                image_urls = self._wait_for_task_completion(task_id)
                if not image_urls:
                    break
                
                for url in image_urls:
                    images.append({
                        "data": url,
                        "format": "url",
                        "prompt": prompt,
                        "model": self.model_name,
                        "metadata": {
                            "resolution": resolution,
                            "aspect_ratio": aspect_ratio
                        }
                    })
                
                remaining -= current_batch
        
        except requests.exceptions.RequestException as e:
            print(f"Model:{self.model_name} API请求失败: {str(e)}")
        except Exception as e:
            print(f"Model:{self.model_name} 生成图像时出错: {str(e)}")
        
        return images
    
    def _wait_for_task_completion(self, task_id, interval=5, timeout=300):
        """
        等待任务完成 (内部方法)
        
        参数:
            task_id: str - 任务ID
            interval: int - 检查间隔(秒)
            timeout: int - 超时时间(秒)
            
        返回:
            list - 生成的图像URL列表
        """
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                # 更新headers确保token是最新的
                self.headers["Authorization"] = self._generate_auth_token()
                
                url = f"{self.base_url}/{task_id}"
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                result = response.json()
                if result["code"] != 0:
                    print(f"Model:{self.model_name} 检查任务状态失败: {result['message']}")
                    return None
                
                status = result["data"]
                if status["task_status"] == "succeed":
                    return [img["url"] for img in status["task_result"]["images"]]
                elif status["task_status"] == "failed":
                    print(f"Model:{self.model_name} 任务失败: {status.get('task_status_msg', '未知错误')}")
                    return None
                
                time.sleep(interval)
            
            print(f"Model:{self.model_name} 等待任务超时 (任务ID: '{task_id}')")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Model:{self.model_name} 检查任务状态请求失败: {str(e)}")
            return None
        except Exception as e:
            print(f"Model:{self.model_name} 等待任务完成时出错: {str(e)}")
            return None
    
    def generate_and_save(self, prompt, num_images=1, **kwargs):
        """
        生成并保存图像
        
        参数:
            prompt: str - 提示词
            num_images: int - 生成图像数量
            kwargs: dict - 其他生成参数
            
        返回:
            list - 保存的文件路径列表
        """
        saved_files = []
        generated_images = self.generate(prompt, num_images=num_images, **kwargs)
        
        for i, img_data in enumerate(generated_images):
            filepath = self.save_image(
                model_name=self.model_name,
                image_data=img_data,
                prompt=prompt,
                prompt_key=kwargs.get("prompt_key", str(i)),
                **kwargs
            )
            if filepath:
                saved_files.append(filepath)
        
        return saved_files

if __name__ == '__main__':
    generate_images_from_prompt(prompt="一名罪犯", prompt_key="criminal", num_images=1)