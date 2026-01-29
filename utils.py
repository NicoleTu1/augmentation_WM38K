import torch
import torch.nn as nn
import numpy as np

import datetime
import time 
import sys
import psutil
import winsound # pygame
# from IPython.display import Audio, display, Javascript

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
# axes_grid1 是 plt 的附屬套件.
# make_axes_locatable 用於精準控制多個子圖 (subplot) 或 colorbar 的位置與大小. 

from model_F import F_feature_processing


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


labels_in_code = ['none', 'Random', 'Scratch', 'Near_Full', 'Loc', 
                  'L+S', 'Edge_Ring', 'ER+S', 'ER+L', 'ER+L+S', 
                  'Edge_Loc', 'EL+S', 'EL+L', 'EL+L+S', 'Donut', 
                  'D+S', 'D+L', 'D+L+S', 'D+ER', 'D+ER+S', 
                  'D+ER+L', 'D+ER+L+S', 'D+EL', 'D+EL+S', 'D+EL+L', 
                  'D+EL+L+S', 'Center', 'C+S', 'C+L', 'C+L+S', 
                  'C+ER', 'C+ER+S', 'C+ER+L', 'C+ER+L+S', 'C+EL', 
                  'C+EL+S', 'C+EL+L', 'C+EL+L+S', 'XXX']

# # 以下是取得 labels_in_code, label_to_code, code_to_label 的程式碼, 先註解起來備用.
# # # 為了後續畫 cm 的時候可以美觀一點, 將缺陷類別轉換成更簡短的名稱
# # # defect_vector_means 是缺陷類型的全名, defect_vector_means_short 是缺陷類型的簡短名稱.
# # defect_vector_means = ('Center', 'Donut', 'Edge_Loc', 'Edge_Ring', 'Loc', 'Near_Full', 'Scratch', 'Random')
# # defect_vector_means_short = ('C', 'D', 'EL', 'ER', 'L', 'NF', 'S', 'R')
# # labels_in_code = []
# # label_to_code, code_to_label = {}, {}

# # defect_types, counts = np.unique(dataset_raw['arr_1'], return_counts=True, axis=0)	# 取出原始label (0/1陣列)
# # for vector in defect_types:
# #     label = ''
# #     if sum(vector) == 0:
# #         label = 'none'
# #     elif sum(vector) == 1:
# #         label = defect_vector_means[vector.argmax()]
# #     else:
# #         for i, d in enumerate(vector):
# #             if d == 1: label += defect_vector_means_short[i] + '+'
# #         label = label.rstrip('+')  # 去掉最後的 '+'
# #     labels_in_code.append(label)

# #     label_to_code[label] = tuple(vector.tolist())   # 將 label (字串) 轉成 0/1 陣列
# #     code_to_label[tuple(vector.tolist())] = label   # 將 0/1 陣列 轉成 label (字串)
# # labels_in_code.append('XXX')    # 若有其他類型的缺陷, 用 'XXX' 來表示. 畫混淆矩陣的時候會用到. 

label_to_code = {(0, 0, 0, 0, 0, 0, 0, 0): 'none',
                 (0, 0, 0, 0, 0, 0, 0, 1): 'Random',
                 (0, 0, 0, 0, 0, 0, 1, 0): 'Scratch',
                 (0, 0, 0, 0, 0, 1, 0, 0): 'Near_Full',
                 (0, 0, 0, 0, 1, 0, 0, 0): 'Loc',
                 (0, 0, 0, 0, 1, 0, 1, 0): 'L+S',
                 (0, 0, 0, 1, 0, 0, 0, 0): 'Edge_Ring',
                 (0, 0, 0, 1, 0, 0, 1, 0): 'ER+S',
                 (0, 0, 0, 1, 1, 0, 0, 0): 'ER+L',
                 (0, 0, 0, 1, 1, 0, 1, 0): 'ER+L+S',
                 (0, 0, 1, 0, 0, 0, 0, 0): 'Edge_Loc',
                 (0, 0, 1, 0, 0, 0, 1, 0): 'EL+S',
                 (0, 0, 1, 0, 1, 0, 0, 0): 'EL+L',
                 (0, 0, 1, 0, 1, 0, 1, 0): 'EL+L+S',
                 (0, 1, 0, 0, 0, 0, 0, 0): 'Donut',
                 (0, 1, 0, 0, 0, 0, 1, 0): 'D+S',
                 (0, 1, 0, 0, 1, 0, 0, 0): 'D+L',
                 (0, 1, 0, 0, 1, 0, 1, 0): 'D+L+S',
                 (0, 1, 0, 1, 0, 0, 0, 0): 'D+ER',
                 (0, 1, 0, 1, 0, 0, 1, 0): 'D+ER+S',
                 (0, 1, 0, 1, 1, 0, 0, 0): 'D+ER+L',
                 (0, 1, 0, 1, 1, 0, 1, 0): 'D+ER+L+S',
                 (0, 1, 1, 0, 0, 0, 0, 0): 'D+EL',
                 (0, 1, 1, 0, 0, 0, 1, 0): 'D+EL+S',
                 (0, 1, 1, 0, 1, 0, 0, 0): 'D+EL+L',
                 (0, 1, 1, 0, 1, 0, 1, 0): 'D+EL+L+S',
                 (1, 0, 0, 0, 0, 0, 0, 0): 'Center',
                 (1, 0, 0, 0, 0, 0, 1, 0): 'C+S',
                 (1, 0, 0, 0, 1, 0, 0, 0): 'C+L',
                 (1, 0, 0, 0, 1, 0, 1, 0): 'C+L+S',
                 (1, 0, 0, 1, 0, 0, 0, 0): 'C+ER',
                 (1, 0, 0, 1, 0, 0, 1, 0): 'C+ER+S',
                 (1, 0, 0, 1, 1, 0, 0, 0): 'C+ER+L',
                 (1, 0, 0, 1, 1, 0, 1, 0): 'C+ER+L+S',
                 (1, 0, 1, 0, 0, 0, 0, 0): 'C+EL',
                 (1, 0, 1, 0, 0, 0, 1, 0): 'C+EL+S',
                 (1, 0, 1, 0, 1, 0, 0, 0): 'C+EL+L',
                 (1, 0, 1, 0, 1, 0, 1, 0): 'C+EL+L+S'}

code_to_label = {'none': (0, 0, 0, 0, 0, 0, 0, 0),
                 'Random': (0, 0, 0, 0, 0, 0, 0, 1),
                 'Scratch': (0, 0, 0, 0, 0, 0, 1, 0),
                 'Near_Full': (0, 0, 0, 0, 0, 1, 0, 0),
                 'Loc': (0, 0, 0, 0, 1, 0, 0, 0),
                 'L+S': (0, 0, 0, 0, 1, 0, 1, 0),
                 'Edge_Ring': (0, 0, 0, 1, 0, 0, 0, 0),
                 'ER+S': (0, 0, 0, 1, 0, 0, 1, 0),
                 'ER+L': (0, 0, 0, 1, 1, 0, 0, 0),
                 'ER+L+S': (0, 0, 0, 1, 1, 0, 1, 0),
                 'Edge_Loc': (0, 0, 1, 0, 0, 0, 0, 0),
                 'EL+S': (0, 0, 1, 0, 0, 0, 1, 0),
                 'EL+L': (0, 0, 1, 0, 1, 0, 0, 0),
                 'EL+L+S': (0, 0, 1, 0, 1, 0, 1, 0),
                 'Donut': (0, 1, 0, 0, 0, 0, 0, 0),
                 'D+S': (0, 1, 0, 0, 0, 0, 1, 0),
                 'D+L': (0, 1, 0, 0, 1, 0, 0, 0),
                 'D+L+S': (0, 1, 0, 0, 1, 0, 1, 0),
                 'D+ER': (0, 1, 0, 1, 0, 0, 0, 0),
                 'D+ER+S': (0, 1, 0, 1, 0, 0, 1, 0),
                 'D+ER+L': (0, 1, 0, 1, 1, 0, 0, 0),
                 'D+ER+L+S': (0, 1, 0, 1, 1, 0, 1, 0),
                 'D+EL': (0, 1, 1, 0, 0, 0, 0, 0),
                 'D+EL+S': (0, 1, 1, 0, 0, 0, 1, 0),
                 'D+EL+L': (0, 1, 1, 0, 1, 0, 0, 0),
                 'D+EL+L+S': (0, 1, 1, 0, 1, 0, 1, 0),
                 'Center': (1, 0, 0, 0, 0, 0, 0, 0),
                 'C+S': (1, 0, 0, 0, 0, 0, 1, 0),
                 'C+L': (1, 0, 0, 0, 1, 0, 0, 0),
                 'C+L+S': (1, 0, 0, 0, 1, 0, 1, 0),
                 'C+ER': (1, 0, 0, 1, 0, 0, 0, 0),
                 'C+ER+S': (1, 0, 0, 1, 0, 0, 1, 0),
                 'C+ER+L': (1, 0, 0, 1, 1, 0, 0, 0),
                 'C+ER+L+S': (1, 0, 0, 1, 1, 0, 1, 0),
                 'C+EL': (1, 0, 1, 0, 0, 0, 0, 0),
                 'C+EL+S': (1, 0, 1, 0, 0, 0, 1, 0),
                 'C+EL+L': (1, 0, 1, 0, 1, 0, 0, 0),
                 'C+EL+L+S': (1, 0, 1, 0, 1, 0, 1, 0)}



class Logger():
    """Logger class to log messages to both console and file."""
    def __init__(self, log_file_path=r'log_to_file.log', hostname: str=''):
        self.log_file_path = log_file_path
        self.hostname = hostname
        self.initialize()

        # hostname = socket.gethostname()
        self.audio_path_list = {
            'e2e24780b959': '2008cat.wav',    # '5070Ti'
            '2e9c0926e682': '2008cat.wav',    # '2070'
            '7d9716d33117': '2008cat.wav',    # 'I7-12700'
            'IDS-RTX5090': '2008cat.wav',  # 'IDS-RTX5090'
            'Nicole': r'C:\Nicole\Master_NTUB_11366001\Lab\Implementation\20250627-augmentation on Mixed WM38\2008cat.wav',  # 筆電
            # 再補桌機 
            }
        hostname = hostname[hostname.find('=')+1:-1]
        self.audio_path = self.audio_path_list.get(hostname, '2008cat.wav')  # 預設音效檔案
        self.play_notification_sound()

    def initialize(self):
        time_now = datetime.datetime.now()
        assert self.hostname != '', "When `is_initialize` is True, `hostname` must be provided."
        with open(self.log_file_path, 'w') as f: # 'w': write 模式, 清空舊內容
            f.write(f"[{time_now}] [NOTEBOOK] Initialized.\n")
            f.write(f"[{time_now}] [NOTEBOOK] Running on server: {self.hostname}\n")
        self.log_GPU_info()
        self.log_CPU_info()
        self.log_RAM_info()
        self.log_development_env_info()

    def log_GPU_info(self):
        message = '----- GPU Information -----\n'
        if torch.cuda.is_available():
            message += f'GPU Name: {torch.cuda.get_device_name(0)}\n'
            message += f'GPU Capability: {torch.cuda.get_device_capability(0)}\n'
            message += f'GPU Memory Allocated: {torch.cuda.memory_allocated(0)} bytes\n'
            message += f'GPU Memory Cached: {torch.cuda.memory_reserved(0)} bytes\n'
            message += f'GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory} bytes\n'
            message += f'GPU Memory Free: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)} bytes\n'
            message += f'GPU Memory Utilization: {torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100:.2f}%\n'
        else: 
            message += 'torch.cuda.is_available() == False\n'

        self.log(message)

    def log_CPU_info(self):
        message = '----- CPU Information -----\n'
        # 1. 核心數
        message += f"the number of physical cores: {psutil.cpu_count(logical=False)}\n"    # 實體核心數
        message += f"the number of logical CPUs: {psutil.cpu_count(logical=True)}\n"   # 邏輯核心數

        # 2. CPU 頻率 (單位: MHz)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            message += f"current frequency: {cpu_freq.current:.2f}MHz\n"
            message += f"max frequency: {cpu_freq.max:.2f}MHz\n"

        # 3. CPU 使用率 (每個核心)
        for i, percentage in enumerate(psutil.cpu_percent(interval=1, percpu=True)):
            message += f"core {i} usage: {percentage}%\n"

        self.log(message)

    def log_RAM_info(self):
        message = '----- RAM Information -----\n'
        virtual_mem = psutil.virtual_memory()
        message += f"Total RAM: {virtual_mem.total / (1024 ** 3):.2f} GB\n"
        message += f"Available RAM: {virtual_mem.available / (1024 ** 3):.2f} GB\n"
        message += f"Used RAM: {virtual_mem.used / (1024 ** 3):.2f} GB\n"
        message += f"RAM Usage Percentage: {virtual_mem.percent}%\n"
        self.log(message)

    def log_development_env_info(self):
        message = '----- Development Environment Information -----\n'
        message += f'Python Version: {sys.version}\n'
        message += f'PyTorch Version: {torch.__version__}\n'
        message += f'torch.device: {device}\n'
        message += f'CUDA Version: {torch.version.cuda}\n'
        message += f'cuDNN Version: {torch.backends.cudnn.version()}\n'
        self.log(message)

    def log(self, message: str='', is_print: bool=False):
        time_now = datetime.datetime.now()
        with open(self.log_file_path, 'a') as f: # 'a': append 模式, 在檔案末尾加入內容, 而不是覆蓋舊內容. 
            if message != '':
                f.write(f"[{time_now}] [NOTEBOOK] {message}\n")
                if is_print: print(message)
            else:
                f.write("----------     ----------\n")

    def play_notification_sound(self):
        # 用 winsound 播放, 只支援 wav 格式. 若要播放 mp3, 需要使用 playsound
        try:
            winsound.PlaySound(self.audio_path, winsound.SND_ASYNC)
            # 第二個參數稱為 「標記（Flags）」. 它的作用是告訴 Windows:「第一個參數到底是什麼東西？以及你要怎麼播放它？」
            # 設為 SND_FILENAME: 播放檔案
            # 設為 SND_ASYNC: 非同步播放 (程式不會卡住，會繼續執行)
        except Exception as e:
            print(f"播放音效出錯: {e}")

        # ↓↓ 用 pygame 播放. 但 pygame 太大了, 不適合只為了播放音效而匯入.
        # # try:
        # #     # 1. 初始化混音器 (只需要執行一次)
        # #     # 參數設定通常為 44.1 kHz, 16 位元, 單聲道
        # #     pygame.mixer.init(frequency=44100, size=-16, channels=1) 
            
        # #     # 2. 載入音效檔案
        # #     # Pygame 支援 WAV, MP3, OGG 等多種格式
        # #     pygame.mixer.music.load(self.audio_path)
            
        # #     # 3. 播放
        # #     pygame.mixer.music.play()
            
        # #     # 4. 關鍵：確保程式暫停足夠長的時間讓音效播放完畢
        # #     # 這裡需要根據音效長度來設定延遲
        # #     while pygame.mixer.music.get_busy():
        # #         time.sleep(0.1)

        # # except Exception as e:
        # #     print(f"使用 Pygame 播放音效時出錯: {e}")
        # # finally:
        # #     # 確保在程式結束前停止混音器
        # #     pygame.mixer.quit()



# def log_to_file(message: str=False, log_file_path=r'log_to_file.log', is_initialize: bool=False, hostname: str='', is_print: bool=False):
#     """
#     將帶有時間戳記的訊息寫入指定檔案. 

#     ### 適用情境
#     在終端機使用指令 `nohup jupyter nbconvert --to notebook --execute ...` 指令執行 ipynb 檔案時, 
#     無法直接在終端機看到 print 訊息, 可改用此函式將訊息寫入檔案.

#     :param message: 要寫入的訊息. 若無輸入, 則印出`----------     ----------`
#     :param log_file_path: 要寫入的檔案路徑. 預設為 `log_to_file.log`
#     :param is_initialize: 是否為初始化動作. 若為 True, 則會清空舊內容並寫入初始化訊息, 並且必須提供 `hostname`. 
#     :param is_print: 是否同時將訊息印出到終端機. 預設為 False.
#     """
#     time_now = datetime.datetime.now()
#     if is_initialize:        
#         assert hostname != '', "When `is_initialize` is True, `hostname` must be provided."
#         with open(log_file_path, 'w') as f: # 'w': write 模式, 清空舊內容
#             f.write(f"[{time_now}] [NOTEBOOK] Initialized.\n")
#             f.write(f"[{time_now}] [NOTEBOOK] Running on server: {hostname}\n")
#     else:
#         with open(log_file_path, 'a') as f: # 'a': append 模式, 在檔案末尾加入內容, 而不是覆蓋舊內容. 
#             if message:
#                 if is_print: print(message)
#                 f.write(f"[{time_now}] [NOTEBOOK] {message}\n")
#             else:
#                 f.write("----------     ----------\n")



def convert_seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02.0f}"



def convert_timestamps_to_strings(timestamps: tuple, format: str='%Y%m%d-%H%M%S(%z)', 
                          columns: list[str]=['start', 'duration', 'end']) -> str:
    """
    Convert timestamps to a formatted string.

    :param timestamps: A tuple of timestamps in seconds.
    :param format: The format string for strftime, default is '%Y%m%d-%H%M%S(%z)'.
    :param columns: Optional list of column names to prepend to each timestamp. ex: `[start, duration, end]`
    :type columns: list[str]
    :return times_struct: A formatted string representing the time in '%Y%M%D-%H%M%S' format. Type: list[str]
    """
    if len(timestamps) == 1:
        return time.strftime(format, time.localtime(timestamps[0]))
    else:
        times_struct = []
        for i, timestamp in enumerate(timestamps):
            time_formatted = time.strftime(format, time.localtime(timestamp))
            if not columns:
                times_struct.append(time_formatted)
            elif columns[i] == 'duration':  # 持續時間不需要顯示年月日, 只需要顯示時分秒
                time_formatted = convert_seconds_to_hms(timestamp)
                times_struct.append(f'duration: {time_formatted}')
            else:
                times_struct.append(f'{columns[i]}: {time_formatted}')
        return times_struct
    


# ##### ---------- Generating Perfect WaferMap ---------- #####
# def generate_perfect_wafermap_and_label(n_batch: int, size: tuple=(64, 64)) -> torch.Tensor:
# # def generate_perfect_wafermap_and_label(n_batch: int, size: tuple=wafer_resize_scale) -> torch.Tensor:
#     """
#     Generate a perfect wafer map and its corresponding label.   
#     The function uses vectorized operations for efficiency.  這個函式使用向量化運算來提高效率.

#     :param n_batch: Number of wafer maps to generate.
#     :param size: Size of the wafer map as a tuple (height, width). Default is (64, 64).
#     :return: A dictionary containing:

#         * '0': A tensor of shape (n_batch, 1, height, width) representing the perfect wafer maps.
#         * '1': A tensor of shape (n_batch, 8) representing the labels (all zeros).
#     """
#     assert isinstance(size, tuple) and len(size) == 2, "Size must be a tuple of (height, width)."

#     h, w = size
#     center_x, center_y = w / 2, h / 2
#     radius_sq = (min(h, w) / 2) ** 2    # radius_sq = radius squared = r ** 2

#     # 建立網格座標
#     y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # shape: (h, w)
#     # ↑ indexing='ij' 確保 y 對應到第一維 (height), x 對應到第二維 (width).
#     # indexing='xy' 是預設值, 但不適合影像處理

#     # 使用向量化運算判斷像素是否在圓形內
#     dist_sq = (x - center_x + 0.5)**2 + (y - center_y + 0.5)**2 # dist_sq = distance square     # +0.5 是為了讓圓心落在像素的正中央
#     wafermap_perfect = (dist_sq <= radius_sq).float().to(device)    # wafermap value range: 0.0 ~ 1.0
#     # 這邊不用讓 wafermap_perfect 的 range 變得跟原始資料集一樣是 (0, 1, 2), 
#     # 因為 0=未使用, 1=正常, 2=缺陷, 而 perfect wafermap 裡面沒有缺陷, 所以只需要 0 和 1 就夠了.

#     wafermap_perfect = wafermap_perfect.expand(n_batch, 1, h, w)
#     # label_perfect = torch.zeros((n_batch, 8), dtype=torch.float).to(device)
#     label_perfect = torch.zeros((n_batch, 8)).to(device)

#     result = {0: wafermap_perfect, 
#               1: label_perfect}
#     return result
# ##### ---------- Generating Perfect WaferMap ---------- #####



def get_sample_dtype_and_elementset(wafermap, label, logger: Logger) -> tuple[tuple[str, str], tuple[set, set]]:
    if type(wafermap) == np.ndarray and type(label) == np.ndarray:
        # 檢查最內層 element 的 dtype
        d = len(wafermap.shape)
        w = wafermap[0]
        for _ in range(d-1): w = w[0]   # w = 最內層 element
        dtype_wafermap = type(w)

        d = len(label.shape)
        l = label[0]
        for _ in range(d-1): l = l[0]   # l = 最內層 element
        dtype_label = type(l)

    elif type(wafermap) == torch.Tensor and type(label) == torch.Tensor:
        dtype_wafermap = wafermap.dtype
        dtype_label = label.dtype

        wafermap = wafermap.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

    else:
       raise TypeError(f'wafermap and label must be both np.ndarray or both torch.Tensor, but got {type(wafermap)} and {type(label)}')

    # 檢查 value elements
    elements_set_wafermap = set(x.item() for x in np.unique(wafermap))
    elements_set_label = set(x.item() for x in np.unique(label))

    message = '(function: get_sample_dtype_and_elementset): \n'
    message += f'  dtype:  wafermap  {dtype_wafermap} \n'
    message += f'          label     {dtype_label} \n'
    message += f'  value:  wafermap  {elements_set_wafermap} \n'
    message += f'          label     {elements_set_label} '
    logger.log(message, is_print=True)
    
    return (dtype_wafermap, dtype_label), (elements_set_wafermap, elements_set_label)



def get_starttime_from_timestamps(timestamps: tuple) -> str:
    """
    從 timestamps 中取得起始時間的字串, 格式為`%Y%m%d-%H%M%S(%z)`.

    :param timestamps: A tuple of timestamps in seconds.
    :return: A formatted string representing the start time in '%Y%m%d-%H%M%S' format.
    """
    start_time_str = time.strftime('%Y%m%d-%H%M%S(%z)', time.localtime(timestamps[0]))
    return start_time_str


def assert_sample_compliance(elements_set=(set(), set()), 
                             expected_set_wafermap={0, 0.5, 1}, expected_set_label={0, 1}, 
                             logger: Logger=None):
    """
    :param sample: (wafermap, label)
    :param expected_set_wafermap: expected set of wafermap elements.
    :param expected_set_label: expected set of label elements.
    """
    elements_set_wafermap, elements_set_label = elements_set[0], elements_set[1]
    assert elements_set_wafermap.issubset(expected_set_wafermap), \
        f'Wafermap elements {elements_set_wafermap} not compliant with expected set {expected_set_wafermap}'
    assert elements_set_label.issubset(expected_set_label), \
        f'Label elements {elements_set_label} not compliant with expected set {expected_set_label}'
    
    if logger is not None:
        logger.log('(function: assert_sample_compliance) Sample compliance check passed.', is_print=True)
    else:
        print('(function: assert_sample_compliance) Sample compliance check passed.')



def assert_nan_and_inf(tensor: torch.Tensor, tensor_name: str='tensor'):
    assert not torch.isnan(tensor).any(), f'class C: {tensor_name} is NaN! `.abs().max()`: {tensor.abs().max()}'
    assert not torch.isinf(tensor).any(), f'class C: {tensor_name} is Inf! `.abs().max()`: {tensor.abs().max()}'



def rectify_pixel_values(x:torch.Tensor):
    """
    將 tensor x 的值限制在 0 和 1 之間, 並將其標準化為 0.0, 0.5, 1.0 三個離散值.
    """
    if x.dim() in [2, 3, 4]:
        x_max = x.max()
        if x_max > 0: # 確保 x.max() 不為 0 以避免除以零的錯誤
            x = torch.round(x / x_max * 2) / 2    # 這行程式碼會對整個 x 張量進行操作, 不需要使用 for 迴圈逐個處理
        return torch.clamp(x, 0, 1)
        
    else:
        raise TypeError(f'函式只接受2維、3維或4維的tensor. x.dim()={x.dim()}, x.shape={x.shape}. ')
        


def plot_wafermap(wafermap, label, custom_title: str=None, axis: str='off'):
    """
    繪製單一晶圓圖. 

    :param wafermap: 晶圓圖影像 (Tensor, shape: (1, H, W))
    :param label: 晶圓圖標籤 (tuple)
    :param custom_title: 自訂標題 (str), 若提供則使用此標題, 否則使用預設標題格式. 
    :param axis: 是否顯示座標軸 (str), 預設為 'off'. 
    """
    # 處理 wafermap 的型態
    if isinstance(wafermap, np.ndarray):
        wafermap = torch.tensor(wafermap).detach().cpu().squeeze()
        
    # 處理 label 的型態
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().tolist()
    if isinstance(label, np.ndarray):
        label = label.tolist()
    if isinstance(label, list):
        if isinstance(label[0], float):
            label = [int(x) for x in label] # 把每個 element 都變成整數, 這樣顯示 title 時會比較短
        label = tuple(label)

    plt.figure(figsize=(3, 3))
    plt.imshow(wafermap)
    plt.axis('on') if axis == 'on' else plt.axis('off')
    label_code = label_to_code[label]
    if custom_title: 
        title = f'{custom_title}'
    else: 
        title = f'Label: {label_code}\n{label}'

    plt.title(title, fontsize=10)
    plt.tight_layout()



def plot_wafermap_by_label_from_datasetraw(label, dataset_raw):
    mask = (dataset_raw['arr_1'] == label).all(axis=1)  # axis=1 表示對每一列進行比較. axis=0 表示對每一欄進行比較.
    indices = np.where(mask)[0] # 取得符合條件的項目的索引值 
    i = np.random.choice(indices) # 隨機選擇一個索引值

    wafermap = dataset_raw['arr_0'][i]
    plt.figure(figsize=(3, 3))
    label_ = dataset_raw['arr_1'][i]
    l_ = [f'{x:2d}' for x in label_]
    l_ = ' '.join(l_)
    title = (f'{l_[1:]}\n(C  D  EL ER L  NF S  R)\n#{i}')
    plot_wafermap(wafermap, label, custom_title=title)

    return i



def plot_loss_with_lr(model_name: str, log_loss: list, log_lr: list, log_each_loss: dict,
                      epochs: int, 
                      model_config: str='(not provided)',
                      times: tuple=(-1, -1, -1)) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(18, 4))

    # --- 左側 Y 軸：loss（線性尺度） ---
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(range(epochs), log_loss, label='Loss', color='tab:blue')
    linestyle_table = { # supported linestyle: '-' is 'solid', '--' is 'dashed', '-.' is 'dashdot', ':' is 'dotted', 'None', ' ', ''
        'mse': '-.',
        'kld': ':',
        'dice': '--',
        'ssim': 'solid'
    }

    if log_each_loss: # 若有提供各個 loss 的紀錄, 則繪製出來
        for loss_label, loss_values in log_each_loss.items():
            linestyle = linestyle_table.get(loss_label, '-')  # default linestyle
            ax1.plot(range(epochs), loss_values, label=loss_label, linestyle=linestyle, color="#555555", linewidth=1)    
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # --- 右側 Y 軸：learning rate（對數尺度） ---
    ax2 = ax1.twinx()
    ax2.set_yscale('log')  # <<< 對數尺度
    ax2.set_ylabel('Learning Rate', color='tab:red')
    ax2.plot(range(epochs), log_lr, label='lr', color='tab:red', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # timestamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    times = convert_timestamps_to_strings(times, columns=['start', 'duration', 'end'])
    times_string = [f'{t}\n' for t in times]
    times_string = ''.join(times_string)

    # fig.text(0.95, 0.90, f'timestamp={timestamp}.', transform=fig.transFigure, ha='right', fontsize=10, color='gray')
    fig.text(0.95, 0.89, f'{times_string}', transform=fig.transFigure, ha='right', fontsize=10, color='gray')
    if isinstance(model_config, str):
        fig.text(0.06, 0.16, f'model config=\n{model_config}', transform=fig.transFigure, ha='left', va='bottom', fontsize=10, color='gray')
    elif isinstance(model_config, dict):
        model_config_str = '\n'.join([f'{k}: {v}' for k, v in model_config.items()])
        fig.text(0.06, 0.16, f'model config=\n{model_config_str}', transform=fig.transFigure, ha='left', va='bottom', fontsize=10, color='gray')
    fig.suptitle(f'{model_name}: Loss per Epoch')
    fig.legend()
    fig.tight_layout()

    return fig#, times_string



def plot_confusion_matrix(confusion_matrix: np.ndarray, defect_types_name: list, 
                          title: str='Confusion Matrix of Each Defect Class', 
                          times=(-1, -1, -1)) -> plt.Figure:
    """draw confusion matrix to show accuracy of each defect class"""
    fig, ax = plt.subplots(figsize=(20, 16))
    cax = ax.matshow(confusion_matrix, cmap='Blues')
    divider = make_axes_locatable(ax)   # 建立一個「分隔器」(divider), 讓使用者在 ax 所在的主繪圖區附近, 動態加入額外的座標軸 (Axes), 例如 colorbar 的座標軸. 
    cbar_ax = divider.append_axes("right", size="1%", pad=0.03)  # cbar_ax 是 colorbar 的位置. size: colorbar 的寬度, 越小越細.
    fig.colorbar(cax, cax=cbar_ax) # pad: colorbar 與主圖之間的距離. fraction: colorbar 高度相對於主繪圖區的比例 (數值越小越矮). 


    # Set ticks and labels
    ax.set_xticks(np.arange(len(defect_types_name)))
    ax.set_yticks(np.arange(len(defect_types_name)))
    ax.set_xticklabels(defect_types_name, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels(defect_types_name, fontsize=10)


    # Set axis labels
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')


    # Set title
    # balanced_accuracy = np.diag(confusion_matrix[:38]).mean()  # 每個類別的貢獻權重相同, 不論其樣本數多寡. 
    # ↑ ##### 但這樣的算法是錯的, 因為沒有除以該類別的樣本總數. #####
    # 雖然我在主程式的 validate_cls 有把 cm 都轉換成百分比, 但難保我之後不會修改 validate_cls 讓 cm 保持原始的數值形式, 
    # 所以這邊還是不要預設 cm 是百分比形式比較保險, 所以在計算上要除以各類別的樣本總數. 所以要用下面的程式碼.
    # title += f'\nBalanced Accuracy: {balanced_accuracy:.4f}'

    # 計算每一個類別的總和 (即每個類別的真實樣本總數)
    # 根據 sklearn 的 confusion_matrix, cm 的定義如下:
    # "Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class."
    # cm.shape = (n_classes, n_classes), 所以 cm[i, j] 代表真實類別為 i, 預測類別為 j 的樣本數量.
    # 所以, 對每一列求和即可得到每個類別的真實樣本總數.
    class_sums = np.sum(confusion_matrix[:38, :39], axis=1) # axis=1 表示對每一列求和, 得到每個類別的真實樣本總數.
    non_zero_indices = class_sums > 0   # 找到所有列總和大於 0 的類別索引 (不然會有除以零的問題)
    cm_diag = np.diag(confusion_matrix[:38, :39])  # 取得混淆矩陣的對角線元素 (正確分類的樣本數)
    # 篩選出有樣本的那些類別的對角線元素和總數
    valid_diag = cm_diag[non_zero_indices] # 計算每個有效類別的正確分類數 (分子)
    valid_sums = class_sums[non_zero_indices] # 計算每個有效類別的樣本總數 (分母)
    class_recall = valid_diag / valid_sums  # 計算每個有效類別的召回率 (recall)
    balanced_accuracy = np.mean(class_recall)
    title += f'\nBalanced Accuracy: {balanced_accuracy:.4f}'

    # overall_accuracy = np.trace(confusion_matrix[:38, :39]) / np.sum(confusion_matrix[:38, :39]) # 每個類別對最終結果的貢獻, 與該類別的樣本數量成正比. 
    # title += f'\nOverall Accuracy: {overall_accuracy:.4f}'

    ax.set_title(title)

    
    # Annotate each cell with the percentage value
    for i in range(len(defect_types_name)):
        for j in range(len(defect_types_name)):
            # percentage = confusion_matrix[i, j] / confusion_matrix.sum(axis=1)[i] if confusion_matrix.sum(axis=1)[i] > 0 else 0
            # font_color = 'white' if (percentage > 0.5) else 'black' # acc 較高的時候, 底色比較深, 所以字的顏色要改為白色
            # ax.text(j, i, f'{percentage:.4f}'.lstrip('0'), ha='center', va='center', color=font_color, fontsize=8)
            # # ax.text(j, i, f'{fusion_matrix[i, j]} ({percentage:.1f})', ha='center', va='center', color='black', fontsize=7)

            if confusion_matrix[i, j] > 0.8: # 0.5:
                font_color = 'white'    # acc 較高的時候, 底色比較深, 所以字的顏色要改為白色
            elif confusion_matrix[i, j] == 0.0:
                font_color = 'gray'    # acc=0 的時候, 不用很顯眼, 所以字的顏色設為灰色
            else:
                font_color = 'red'      # acc 不夠高也不為 0 時, 這些分類結果有待改善, 所以用紅色比較顯眼
            # font_color = 'white' if (confusion_matrix[i, j] > 0.5) else 'black' # acc 較高的時候, 底色比較深, 所以字的顏色要改為白色
            ax.text(j, i, f'{confusion_matrix[i, j]:.4f}'.lstrip('0'), ha='center', va='center', color=font_color, fontsize=8)


    # # Annotate each cell with the numeric value
    # for i in range(len(class_names)):
    #     for j in range(len(class_names)):
    #         ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')


    # timestamp = time.strftime('%Y%m%d-%H%M%S(%z)', time.localtime())
    times = convert_timestamps_to_strings(times, columns=['start', 'duration', 'end'])
    times_string = [t for t in times]
    times_string = '\n'.join(times_string)
    # fig.text(0.85, 0.98, f'timestamp={timestamp}.', transform=fig.transFigure, ha='right', fontsize=10, color='gray')
    fig.text(0.85, 0.95, times_string, transform=fig.transFigure, ha='right', fontsize=10, color='gray')
    plt.tight_layout()
    return fig



# 將 confusion matrix 轉換成折線圖, 對比實驗組及對照組的 accuracy
def plot_cm_as_linechart_for_each_defect_class(cm_augmented: np.ndarray, cm_baseline: np.ndarray, 
                                               defect_types_name: list, 
                                               title: str='Confusion Matrix Comparison of Each Defect Class',
                                               times: tuple=(-1, -1, -1)) -> plt.Figure:
    """
    Plot comparison of confusion matrices as line plots to show accuracy of each defect class.
    
    :param cm_augmented: Confusion matrix for augmented model (numpy array).
    :param cm_baseline: Confusion matrix for baseline model (numpy array).
    :param defect_types_name: List of defect type names (list of str).
    :param title: Title for the plot (str).
    :param times: Tuple of timestamps (start, duration, end) for annotation (tuple).
    :return: Matplotlib figure object.
    """

    # cm_augmented 和 cm_baseline 的原始 shape 都是 (39, 39). 因為只要計算 38 個缺陷類型, 所以取前 38 列, 38 欄
    cm_augmented, cm_baseline = cm_augmented[:38, :38], cm_baseline[:38, :38]
    fig, ax = plt.subplots(figsize=(18, 6))
    indices = np.arange(len(defect_types_name)) # x 軸的索引位置    
    acc_augmented = np.diag(cm_augmented[:38, :38]) / np.sum(cm_augmented[:38, :38], axis=1)  # 計算每個類別的 accuracy # shape: (38,)
    acc_baseline = np.diag(cm_baseline[:38, :38]) / np.sum(cm_baseline[:38, :38], axis=1)        # 計算每個類別的 accuracy # shape: (38,)
    ax.plot(indices, acc_augmented, marker='o', label='Augmented', color='tab:blue')    # marker='o' 代表圓形標記
    ax.plot(indices, acc_baseline, marker='x', label='Baseline', color='tab:orange')    # 三角形='^', 十字形='x', 方形='s'
    
    ax.set_xticks(indices)
    ax.set_xticklabels(defect_types_name, rotation=45, ha='left', fontsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)

    ax.set_title(title)
    times = convert_timestamps_to_strings(times, columns=['start', 'duration', 'end'])
    times_string = [f'{t}\n' for t in times]
    times_string = ''.join(times_string)
    ax.text(0.95, 0.95, f'{times_string}', 
            transform=fig.transFigure, ha='right', fontsize=10, color='gray')   # transform=fig.transFigure 代表相對於整個 figure 的位置
    ax.legend()
    plt.tight_layout()

    return fig



# 畫出資料集中各個故障型別的數量分佈圖 (堆疊直條圖: 原始資料集+驗證資料集)
def plot_defect_type_distribution(dataset: tuple, validation: tuple=None, legend: tuple=('Training Set', 'Validation Set'), title: str=''):
    """
    Plot the distribution of defect types in the dataset.   
    Args:
        dataset (tuple): A tuple containing training defect types and their counts.
        validation (tuple, optional): A tuple containing validation defect types and their counts. Defaults to None.
        title (str, optional): Title for the plot. Defaults to ''.
    """
    fig, ax = plt.subplots(figsize=(16, 4))

    train_types, train_counts = dataset

    if isinstance(train_types[0], np.ndarray) or isinstance(train_types[0], tuple):
        train_defect_type_labels = [label_to_code[tuple(x)] for x in train_types]
    elif isinstance(train_types[0], str):
        train_defect_type_labels = train_types
    else: 
        raise TypeError('types[0] should be tuple/list, np.ndarray or str')

    if validation is None:
        ax.bar(train_defect_type_labels, train_counts, color='lightgray')
        for i, v in enumerate(train_counts):
            ax.text(i, v, str(v), color='black', ha='center', fontsize=8.5)  # 畫出數值標籤
            # ax.text(i, v, str(v), color='navy', ha='left', rotation=45)  # 畫出數值標籤
        
    else:
        total_types, total_counts = dataset
        val_types, val_counts = validation
        train_counts = np.ndarray([], dtype=int)
        for ttl, v in zip(total_counts, val_counts):
            train_count = ttl - v
            train_counts = np.append(train_counts, train_count)
        train_counts = train_counts[1:]

        if isinstance(val_types[0], np.ndarray) or isinstance(val_types[0], tuple):
            val_defect_type_labels = [label_to_code[tuple(x)] for x in val_types]
        elif isinstance(val_types[0], str):
            val_defect_type_labels = val_types
        else: 
            raise TypeError('types[0] should be tuple/list, np.ndarray or str')

        ax.bar(train_defect_type_labels, total_counts, color='lightgray') 
        ax.bar(val_defect_type_labels, val_counts, color='gray', alpha=0.7) # alpha=1 代表不透明
        for i, (ttl, t, v) in enumerate(zip(total_counts, train_counts, val_counts)):
                ax.text(i, ttl, str(ttl), color='gray', ha='center', va='bottom', fontsize=8.5)  # 畫出數值標籤 (整個資料集)
                ax.text(i, (ttl-v)//2+v, str(t), color='black', ha='center', va='center', fontsize=8.5)  # 畫出數值標籤 (訓練集)
                ax.text(i, v//2, str(v), color='black', ha='center', va='center', fontsize=8.5)  # 畫出數值標籤 (驗證集)

        # for v in np.unique(total_counts):
        #     ax.axhline(v, color='gray', linestyle='--', linewidth=0.5, alpha=0.5) # 畫出輔助線 # 若在數值標籤的迴圈一起畫, 會因為有很多重疊而變成像是直線一樣
    
        plt.legend(legend)

    ys = [500, 1000, 1500, 2000]
    for y in ys:
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.25) # 畫出輔助線

    ax.set_yticks(np.arange(0, 2251, 250))
    ax.set_xlabel('Defect Type')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Defect Types {title}')
    plt.xticks(rotation=45, ha='right') # x軸標籤旋轉45度
    plt.tight_layout()
    # plt.show()
    return fig


from model_C import C_ColorizerNetwork
def plot_colored_wafermaps(timestamp: str, ref_image, ref_label, colored_wafermaps, 
                           colVAE: C_ColorizerNetwork,
                           module_F: F_feature_processing=None,
                        #    colVAE_loss_function: str=None, 
                           n_row=2, n_col=4):
    """
    繪製生成的彩色晶圓圖. 

    :param ref_image: 參考影像 (shape: (1, 1, H, W))
    :param colored_wafermaps: 生成的彩色晶圓圖 (shape: (n_sample, 1, H, W))
    :param colVAE: C_ColorizerNetwork 實例
    :param colVAE_loss_function: colVAE 使用的損失函數描述字串
    :param n_row: 繪製的列數
    :param n_col: 繪製的欄數
    """
    fig_width, fig_height = (15, 6) if n_col == 5 else (10, 5.5)
    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_width, fig_height)) # row2, col4


    if ref_image.size(0) == 1 and ref_image.dim() == 3:
        for i in range(n_row * n_col):
            if i == 0:
                ax = axes[i // n_col, i % n_col]
                ax.imshow(ref_image.detach().cpu().squeeze())
                # ax.axis('off')
            else:
                ax = axes[i // n_col, i % n_col]
                ax.imshow(colored_wafermaps[i-1].detach().cpu().squeeze())
                ax.axis('off')
        note = f'1 ref_image + {n_col * n_row - 1} colored_wafermaps'

    else:
        axes = axes.flatten()
        if len(ref_label) == len(ref_image):
            zipped = zip(ref_image, ref_label, colored_wafermaps)
        else:
            zipped = zip(ref_image, colored_wafermaps)

        for i, packed in enumerate(zipped):
            if len(packed) == 3:
                ref_image, ref_label, colorized_image = packed
                ref_label = tuple(ref_label.detach().cpu().squeeze().numpy().tolist())
                ref_label = label_to_code[ref_label]
            else:
                ref_image, colorized_image = packed
                ref_label = ('')
            # ref_image.shape = colorized_image.shape = (1, H, W)
            
            if i * 2 >= n_row * n_col:
                break
            else: 
                ax = axes[i * 2]
                ax.imshow(ref_image.detach().cpu().squeeze())
                ax.axis('off')
                ax.set_title(f'#{i}+n ref_label={ref_label}', fontsize=8, loc='left')

                ax = axes[i * 2 + 1]
                ax.imshow(colorized_image.detach().cpu().squeeze())
                ax.axis('off')
        note = f'(ref_wafermap + colored_wafermaps) * {n_col * n_row // 2}'

            
    # 加上註記文字
    x, y = (0.95, 0.89) if n_col == 5 else (0.95, 0.95)
    fig.text(x, y, f'{timestamp}\n{note}', transform=fig.transFigure, ha='right', fontsize=10, color='gray')

    fig.tight_layout()

    # 加上標題
    fontsize=14 if n_col == 5 else 12
    title = f'Generated Colored Wafer Maps \nloss_function = {colVAE.config['loss_function']}'
    title += f'\nC.encoder_input_channels={colVAE.encoder_input_channels}, z_dim={colVAE.z_dim}, qk_channel={module_F.qk_channel if module_F is not None else "N/A"}'
    fig.suptitle(title, fontsize=fontsize)
    n_crlf_in_title = title.count('\n')
    top = 0.89 if n_col <= 5 else 0.86
    top -= 0.03 * (n_crlf_in_title - 1) 
    fig.subplots_adjust(top=top)  # 調整標題與子圖之間的距離, top越大, 距離越近

    return fig



def save_modelparams(model: nn.Module, optimizer: nn.Module, 
                     log_loss: list, 
                     timestamps: tuple|list, 
                     pathname: str='record of experiment/', 
                     filename: str=None):
    """
    :param nn.Module model: The model to save.
    :param nn.Module optimizer: The optimizer to save.
    :param list log_loss: The loss value to save.
    :param tuple|list timestamps: The timestamps for the saved file. Default: None.
    :param str pathname: The directory path to save the file.
    :param str filename: The filename to save the model parameters. ex: `R_ref` or `R_gray`. Default: `type(model).__name__`
    """
    assert isinstance(model, nn.Module), f'model must be an instance of nn.Module, but got {type(model)}.'

    # 只儲存模型參數
    # torch.save(R_gray.state_dict(), f'/record of experiment/{timestamp}_Rgray_weights.pth') # lab server
    
    # 儲存模型參數 + optimizer
    start_time_str = get_starttime_from_timestamps(timestamps)
    obj = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
        'loss': log_loss,
    }
    filename = model.config['name'] if filename is None else filename
    filename = f'{start_time_str}_{filename}_weights.pth'
    torch.save(obj, pathname + filename)  
    print(f'Saved.\n{pathname}{filename}')


# # 以下是舊的 save_modelparams 函式, 已被上面的取代, 留著以備不時之需.
# def save_modelparams(model: nn.Module, optimizer: nn.Module, epochs: int, model_config: str, 
#                      log_loss: list, 
#                      loss_function: str=None,
#                      timestamp: str=None, 
#                      pathname: str='record of experiment/', 
#                      filename: str=None):
#     """
#     :param nn.Module model: The model to save.
#     :param nn.Module optimizer: The optimizer to save.
#     :param int epochs: The number of epochs trained.
#     :param str model_config: The configuration of the model.
#     :param list log_loss: The loss value to save.
#     :param str timestamp: The timestamp for the saved file. recommend format is '%Y%m%d-%H%M%S(%z)'. Default: None.
#     :param str pathname: The directory path to save the file.
#     :param str filename: The filename to save the model parameters. ex: `R_ref` or `R_gray`. Default: `type(model).__name__`
#     """
#     assert isinstance(model, nn.Module), f'model must be an instance of nn.Module, but got {type(model)}.'

#     # 只儲存模型參數
#     # torch.save(R_gray.state_dict(), f'/record of experiment/{timestamp}_Rgray_weights.pth') # lab server
    
#     # 儲存模型參數 + optimizer
#     # timestamp = time.strftime('%Y%m%d-%H%M%S(%z)', time.localtime(timestamp))
#     obj = {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'epochs': epochs,
#         'model_config': model_config,
#         'loss': log_loss,
#         'loss_function': loss_function # 給 colVAE 用的
#     }
#     # filename = type(model).__name__ if filename is None else filename
#     filename = model.name if filename is None else filename
#     filename = f'{timestamp}_{filename}_weights.pth'
#     torch.save(obj, pathname + filename)  
#     print(f'Saved.\n{pathname}{filename}')



# def save_text_loss_of_experiment(timestamp: str, times, model_name: str, model_config: str, log_loss: list, log_lr: list, 
#                                  filename=None, 
#                                  is_each_loss=False): #, each_loss_label=['mse', 'kl', 'dice', 'ssim']):
def save_text_loss_of_experiment(timestamps: tuple|list, model: torch.nn.Module, log_loss: list, log_lr: list, 
                                 filename=None, 
                                 is_each_loss=False): #, each_loss_label=['mse', 'kl', 'dice', 'ssim']):
    """ Save the training loss and learning rate logs to a text file.

    The filename is `{timestamp}_{model_name}_loss.txt`.

    ~~:param timestamp: Timestamp of the experiment.~~

    :param timestamps: Tuple of (start_time, duration, end_time).
    :param model: The model instance.
    :param log_loss: List of loss values logged during training.
    :param log_lr: List of learning rates logged during training.
    :param filename: Optional custom filename (without extension). Default is None, which results in `{timestamp}_{model_name}_loss.txt`.
    :param is_each_loss: If True, log_loss is assumed to be a list of lists, where each inner list contains multiple loss components per epoch.
    :param each_loss_label: List of labels for each loss component when is_each_loss is True.
    :return None: Print the path to the directory, and the filename of the saved record.
    """
    
    # timestamp = time.strftime('%Y%m%d-%H%M%S(%z)', time.localtime(timestamps[0]))
    start_time_str = get_starttime_from_timestamps(timestamps)

    if filename:
        filename = f'{start_time_str}_{model.config['name']}_{filename}.txt'
    else:
        filename = f'{start_time_str}_{model.config['name']}_loss.txt'

    times_str = convert_timestamps_to_strings(timestamps, columns=['start', 'duration', 'end'])

    with open(f'record of experiment/{filename}', 'w') as f:
        f.write(f'timestamp=\n{times_str}\n\n')

        if hasattr(model, 'config'):
            f.write(f'model config=\n')
            f.write('\n'.join(f'{k}: {v}' for k, v in model.config.items()))
            f.write('\n\n')
        elif hasattr(model, 'model_config'): # 雖然 model 的設定都存在 model.config (字典), 但舊的 model 是存在 model.model_config (字串).
            f.write(f'model config=\n{model.model_config}\n\n')
        
        if is_each_loss:    
            for key, value in log_loss.items():
                f.write(f'-----  Loss: {key:^10}  -----\n') # 最長的loss名稱是 conceptual / contextual, 所以這裡寬度給10讓它置中
                f.write(f'{"Epoch":>5s} | {"Loss":<14s} | {"Learning Rate":<11s}\n')
                f.write('-' * 30 + '\n')
                for i, (l, lr) in enumerate(zip(value, log_lr)):
                    f.write(f'{i+1:5d} | {l:.10f} | {lr:.7f}\n') 
                f.write('\n')
        else:
            # f.write(f'model name={model_name}\n')
            f.write(f'{"Epoch":>5s} | {"Loss":<14s} | {"Learning Rate":<11s}\n')
            f.write('-' * 30 + '\n')
            for i, (loss, lr) in enumerate(zip(log_loss, log_lr)):
                f.write(f'{i+1:5d} | {loss:.10f} | {lr:.7f}\n')
    
    # print(homepath, filename, sep='')
    print(filename, sep='')



# def save_text_cm_of_experiment(timestamps: tuple, model_name: str, model_config: str|dict, cm: np.ndarray):
def save_text_cm_of_experiment(timestamps: tuple, cm: np.ndarray, model: torch.nn.Module):
    """ 
    Save the confusion matrix and experiment details to a text file.    

    The filename is `{timestamp}_{model_name}_cm.txt`.

    :param timestamps: Tuple of (start_time, duration, end_time) in seconds.
    :param model: The model instance.
    :param cm: Confusion matrix as a numpy array.
    :return None: Print the path to the directory, and the filename of the saved record
    """
    start_time_str = get_starttime_from_timestamps(timestamps)
    filename = f'{start_time_str}_{model.config["name"]}_cm.txt'

    times_str = convert_timestamps_to_strings(timestamps, columns=['start', 'duration', 'end'])

    with open(f'record of experiment/{filename}', 'w') as f:
        f.write(f'timestamp=\n{times_str}\n\n')
        # f.write(f'{times_str}\n\n')

        # f.write(f'model name={model_name}\n')
        if hasattr(model, 'config'):    # 舊的模型沒有 .config 屬性, 所以要檢查
            f.write(f'model config=\n')
            f.write('\n'.join(f'{k}: {v}' for k, v in model.config.items()))
            f.write('\n\n')
        elif hasattr(model, 'model_config'): # 雖然 model.config 都是 dict, 但舊的 model.model_config 還是 str.
            f.write(f'model config=\n{model.model_config}\n\n')        
        # if isinstance(model_config, dict):
        #     f.write(f'model config=\n')
        #     f.write('\n'.join(f'{k}: {v}' for k, v in model_config.items()))
        #     f.write('\n\n')
        # elif isinstance(model_config, str): # 雖然 model.config 都是 dict, 但舊的 model.model_config 還是 str.
        #     f.write(f'model config=\n{model_config}\n\n')

        # f.write(f'Confusion Matrix:\n{cm}') # 不可以這樣寫, 因為只會印出每列的前/後幾個元素, 中間會以 ... 省略號表示. (但可以改成下面 np.savetext 的寫法?)
        # np.savetxt(f'record of experiment/{filename}', cm, fmt='%.4f', delimiter=' ', newline='\\n', header='', footer='', comments='', encoding=None, append=True)
        f.write('Confusion Matrix:\n')
        for line in cm:
            for element in line:
                f.write(f'{element:.4f} ')
            f.write('\n')
    
    # print(homepath, filename, sep='')
    print(f'Text saved. \n{filename}')



def save_fig_of_experiment(figtype: str, fig: plt.Figure, timestamps: tuple, model_name: str):
    """ Save the figure of the experiment to a file. 
    
    The filename is `{timestamp}_{model_name}_{figtype}.png`.

    :param figtype: Type of the figure. Must be either 'loss', 'cm', 'generated_wafermaps', 'distOfLoss', or 'distOfLoss(truncated)'.
    :param fig: The figure to save.
    ~~:param timestamp: Timestamp of the experiment.~~
    :param timestamps: Tuple of (start_time, duration, end_time) in seconds.
    :param model_name: Name of the model. The attribute of the model, `model.name`.
    :return None: Print the path to the directory and the filename of the saved figure.
    """
    start_time_str = get_starttime_from_timestamps(timestamps)
    # timestamp = time.strftime('%Y%m%d-%H%M%S(%z)', time.localtime(timestamps[0]))

    assert figtype in ['loss', 'cm', 'generated_wafermap', 'generated_wafermaps', 'distOfLoss', 'distOfLoss(truncated)'], f'figtype must be "loss", "cm", "generated_wafermaps", "distOfLoss", or "distOfLoss(truncated)", but got {figtype}.'
    filename = f'{start_time_str}_{model_name}_{figtype}.png'
    fig.savefig(f'record of experiment/{filename}', dpi=300, bbox_inches='tight') #bbox_inches='tight' 讓圖表不會有多餘的空白邊界.

    # print(homepath, filename, sep='')
    print(f'Fig saved. \n{filename}')



##### ---------------- 計算 loss component 的 statistic ---------------- #####
def get_statistic_of_each_loss_component(path=None, filenames=None, n_TargetBatch=None, is_ALL_target_batch=False) -> dict:
    """
    for `C`olorizerNetwork training. try to normalized each loss component to z_score, so we need to calculate the mean and std.
    
    If no parameters are passed, a dictionary containing the value([mean, std, median, mad, iqr, min, max]) for key('mse', 'kld', 'dice', 'ssim').

    :param str path: `r'record of experiment'`
    :param list filenames: list of filenames to read. Default: None, which means using hard-coded filenames. 
    """
    from collections import defaultdict

    if (path is None) and (filenames is None) and (n_TargetBatch is None) and (is_ALL_target_batch == False):
        n_TargetBatch = 0 if n_TargetBatch is None else n_TargetBatch
        # 因為沒有輸入檔案路徑, 所以直接回傳預先算好的值
        result = { 
            # # data ref: '20251009-052619(+0000)_C_ColorizerNetwork_log_each_loss.txt' ~ '20251009-052840(+0000)_C_ColorizerNetwork_log_each_loss.txt'
            # # n_sample=30, n_TargetBatch=0
            'mse' : {'mean': 504.127624512, 'std': 7.031450272, 'median': 503.109222412, 'mad': 3.766998291, 'iqr': 7.932678223, 'min': 494.511291504, 'max': 527.642700195}, 
            'kld' : {'mean': 1.383654356, 'std': 0.077658668, 'median': 1.377963781, 'mad': 0.051566482, 'iqr': 0.102445722, 'min': 1.238697648, 'max': 1.535209060}, 
            'dice' : {'mean': 0.503958821, 'std': 0.017638486, 'median': 0.503927350, 'mad': 0.013158202, 'iqr': 0.025584638, 'min': 0.470315039, 'max': 0.539154053}, 
            'ssim' : {'mean': 0.781747103, 'std': 0.006256873, 'median': 0.780617774, 'mad': 0.003374517, 'iqr': 0.007283986, 'min': 0.770378947, 'max': 0.797101498}, 

            # # data ref: 20251008-060658(+0000)_C_ColorizerNetwork_log_each_loss.txt' ~ '20251008-061105(+0000)_C_ColorizerNetwork_log_each_loss.txt'
            # # n_sample=30, n_TargetBatch=0
            # 'mse':  {'mean': 6.226030827, 'std': 0.010899451, 'median': 6.223666668, 'mad': 0.008213043, 'iqr': 0.016858578, 'min': 6.210076332, 'max': 6.248386383},
            # 'kld':  {'mean': 0.853521764, 'std': 0.027294548, 'median': 0.852841437, 'mad': 0.010493696, 'iqr': 0.021151543, 'min': 0.794904292, 'max': 0.939512968},
            # 'dice': {'mean': 0.496889293, 'std': 0.021516964, 'median': 0.496405900, 'mad': 0.019920170, 'iqr': 0.040448993, 'min': 0.464943647, 'max': 0.530123770},
            # 'ssim': {'mean': 0.779832780, 'std': 0.006977445, 'median': 0.779060245, 'mad': 0.002950549, 'iqr': 0.007351339, 'min': 0.766385078, 'max': 0.794241190}

            # # data ref: 20251001-081240(+0000)_C_ColorizerNetwork_log_each_loss.txt' ~ '20251001-085824(+0000)_C_ColorizerNetwork_log_each_loss.txt'
            # # n_sample=20, n_TargetBatch=0
            # # !!!!! 這組數據是錯的, 因為當時把每個 loss 的 goal 設為 -1 才跑 training !!!!!!
            # 'mse': (5.222774982452393, 0.010073106735944748), # mean, std
            # 'kld': (0.13235540688037872, 0.02738793194293976),
            # 'dice': (0.5075749158859253, 0.018361791968345642),
            # 'ssim': (0.21983885765075684, 0.006255291868001223)
            }
        return result
    else:
        from collections import defaultdict
        import os

        # 讀取檔案, 用以計算 (mean, std) of component loss
        path = r'record of experiment' if path is None else path
        filenames = [
                '20251009-052619(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052638(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052643(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052647(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052651(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052654(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052659(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052703(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052707(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052712(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052717(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052720(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052724(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052729(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052733(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052738(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052741(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052745(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052750(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052755(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052801(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052805(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052809(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052813(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052817(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052822(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052827(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052832(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052836(+0000)_C_ColorizerNetwork_log_each_loss.txt', 
                '20251009-052840(+0000)_C_ColorizerNetwork_log_each_loss.txt',                 
                ] if filenames is None else filenames
            
        loss_value_at_TargetBatch = defaultdict(list)  

        if not is_ALL_target_batch:
            if isinstance(n_TargetBatch, int): target_text = f'loss for batch #{n_TargetBatch:3d}='
            
        for filename in filenames:
            file_path = os.path.join(path, filename)
            
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    if is_ALL_target_batch:
                        # 計算所有 batch 的 loss
                        pos = line.find('loss for batch #')
                        if pos != -1:
                            loss_label = line[:pos].strip()
                            loss_value_at_TargetBatch[loss_label].append(float(line[pos+20:-1]))
                    
                    else: # 如果只計算指定的batch
                        # assert isinstance(n_TargetBatch, int) and n_TargetBatch >= 0, \
                        #     f'except n_targetBatch is int and >= 0, but got {type(n_TargetBatch)} in value {n_TargetBatch}.'
                        
                        if line.find(target_text) != -1:
                            pos = line.find(target_text)
                            loss_label = line[:pos-1].strip()

                            pos = line.find('=')
                            loss_value_at_TargetBatch[loss_label].append(float(line[pos+1:-1]))

        length = len(loss_value_at_TargetBatch[loss_label])
        result = defaultdict(dict)

        print(f'len(losses)={length}')
        for loss_label in loss_value_at_TargetBatch.keys():
            values = loss_value_at_TargetBatch[loss_label]
            value = torch.tensor(values, dtype=torch.float, device=device)
            # print(value.shape)    # torch.Size([20])
            mean = torch.mean(value)
            assert mean != 0, f'mean sould not be zero. but got mean={mean} for {loss_label}'

            std = torch.std(value, unbiased=True)
            assert std != 0, f'std sould not be zero. but got std={std} for {loss_label}'

            median = torch.median(value)
            assert median != 0, f'median sould not be zero. but got median={median} for {loss_label}'

            mad = torch.median(torch.abs(value - median))  # median absolute deviation
            assert mad != 0, f'mad sould not be zero. but got mad={mad} for {loss_label}'

            min = torch.min(value)
            max = torch.max(value)
            assert min <= max, f'min should be less than or equal to max, but got min={min}, max={max} for {loss_label}'

            iqr = torch.quantile(value, 0.75) - torch.quantile(value, 0.25)  # interquartile range
            assert iqr != 0, f'iqr sould not be zero. but got iqr={iqr} for {loss_label}'

            # print(f"{loss_label}: 'mean': {mean:.9f}, 'std': {std:.9f}, 'median': {median:.9f}, 'mad': {mad:.9f}, 'iqr': {iqr:.9f}, 'min': {min:.9f}, 'max': {max:.9f}")
            result[loss_label] = {
                'mean': mean,
                'std': std,
                'median': median,
                'mad': mad,
                'iqr': iqr,
                'min': min,
                'max': max,
            }
        print(result)
        print(type(result['mse']['mean']))
        
        return result

        # for loss_label, values in loss_value_at_TargetBatch.items():
        #     value = torch.tensor(values, dtype=torch.float, device=device)
        #     # print(value.shape)    # torch.Size([20])
        #     mean = torch.mean(value)
        #     std = torch.std(value, unbiased=True)
        #     median = torch.median(value)
        #     mad = torch.median(torch.abs(value - median))  # median absolute deviation
        #     min = torch.min(value)
        #     max = torch.max(value)
        #     iqr = torch.quantile(value, 0.75) - torch.quantile(value, 0.25)  # interquartile range
        #     assert std != 0, f'std sould not be zero. but got std={std} for {loss_label}'
        #     assert mad != 0, f'mad sould not be zero. but got mad={mad} for {loss_label}'

        #     print(f"{loss_label}: 'mean': {mean:.9f}, 'std': {std:.9f}, 'median': {median:.9f}, 'mad': {mad:.9f}, 'iqr': {iqr:.9f}, 'min': {min:.9f}, 'max': {max:.9f}")
##### ---------------------------------------------------------------- #####

