import re
import psutil
import GPUtil
import json

class LanguageDetector():
    def __init__(self):
        self.chinese_pattern = re.compile("[\u4e00-\u9fa5]+")
    
    
    def is_chinese(self,context):
            return True if self.chinese_pattern.search(context) else False
     
        
    def language_detection(self,context):
        if self.is_chinese(context):
            return "zh"
        else:
            return "en"
    
    @staticmethod    
    def markdown_to_json(text):
        try:
            if '```json' in text:  # 更精确地检查Markdown中的JSON代码块
                start = text.find('```json') + len('```json\n')
                end = text.rfind('```')
                json_str = text[start:end].strip()
            else:
                json_str = text.strip()
            
            return json.loads(json_str)  # 尝试解析JSON
        except json.JSONDecodeError as e:
            print(f"解析JSON时发生错误：{e}")
            return None
            # return {
            #     "option": "",
            #     "explanation": ""
            
            # }  # 返回None或者适当的错误信息
        

class SystemInfo():
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.cpu_percent = psutil.cpu_percent()
        self.memory = psutil.virtual_memory()
        self.gpu = GPUtil.getGPUs()
        
    def get_cpu_count(self):
        return self.cpu_count
    
    def get_cpu_percent(self):
        return self.cpu_percent
    
    def get_memory(self):
        return self.memory
    
    def get_gpu(self):
        return self.gpu
    
    @staticmethod
    def get_system_and_gpu_metrics():
        # 获取CPU和内存等系统指标
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        system_metrics = {
            'cpu_percent': cpu_percent,
            'memory_used_percent': memory.percent,
            'disk_read_bytes': disk_io.read_bytes,
            'disk_write_bytes': disk_io.write_bytes,
            'net_sent_bytes': net_io.bytes_sent,
            'net_recv_bytes': net_io.bytes_recv,
        }

        # 获取GPU指标
        gpu_metrics = []
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_metrics.append({
                'gpu_id': gpu.id,
                'gpu_name': gpu.name,
                'gpu_load': gpu.load * 100,  # 转换为百分比
                'gpu_free_memory': gpu.memoryFree,
                'gpu_used_memory': gpu.memoryUsed,
                'gpu_total_memory': gpu.memoryTotal,
                'gpu_temperature': gpu.temperature,
            })

        # 将GPU指标整合到系统指标中
        system_metrics['gpu_metrics'] = gpu_metrics

        return system_metrics
    
    @staticmethod
    def calculate_metric_differences(start_metrics, end_metrics):
        differences = {}
        for key in start_metrics:
            if key == 'gpu_metrics':
                # 处理GPU指标差异
                start_gpus = start_metrics[key]
                end_gpus = end_metrics[key]
                for i, (start_gpu, end_gpu) in enumerate(zip(start_gpus, end_gpus)):
                    gpu_prefix = f'gpu_{i}_'  # 为每个GPU指标添加前缀，如gpu_0_load_change
                    differences.update({
                        gpu_prefix + 'load_change': end_gpu['gpu_load'] - start_gpu['gpu_load'],
                        gpu_prefix + 'free_memory_change': end_gpu['gpu_free_memory'] - start_gpu['gpu_free_memory'],
                        gpu_prefix + 'used_memory_change': end_gpu['gpu_used_memory'] - start_gpu['gpu_used_memory'],
                        gpu_prefix + 'temperature_change': end_gpu['gpu_temperature'] - start_gpu['gpu_temperature'],
                    })
            elif 'bytes' in key:
                # 对于磁盘和网络I/O，我们关心的是差异
                differences[key] = end_metrics[key] - start_metrics[key]
            else:
                # 对于CPU和内存，我们可能更关心最终的利用率
                differences[key] = end_metrics[key]
        return differences