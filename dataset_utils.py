import torch
from torchvision import transforms
import numpy as np

from utils import device, rectify_pixel_values



batch_size = 128 # 64
wafer_resize_scale = (64, 64)


dataset_config = {
    'batch_size': batch_size, 
    'wafer_resize_scale': wafer_resize_scale
}


# transform = transforms.Compose([
#     transforms.ToPILImage(),  # 將三維張量 (C, H, W) 或二維張量 (H, W) 轉換為 PIL Image (shape=(H, W)). 會將像素值映射到 [0, 1].
#     # ↑ 為什麼需要 ToPILImage? 因為下面的 Resize 跟 ToTensor 要求 input 必須是 PIL 影像. 
#     # (有許多影像處理的 operation 是專門為 PIL 影像設計的, 不能在 tensor 上執行, 所以要先轉換成 PIL 影像)
#     transforms.Resize(wafer_resize_scale, interpolation=transforms.InterpolationMode.NEAREST), # NEAREST: 最近鄰插值, 不會改變像素值
#     transforms.ToTensor()
# ])



class WafermapDataset(torch.utils.data.Dataset):
    """
    * 輸入的晶圓圖 value 應該是 {0, 1, 2}, 經過 transform 後會變成 {0.0, 0.5, 1.0}.
    * 輸入的完美晶圓圖的 value 是 {0, 1}, 經過 transform 後會變成 {0.0, 0.5}.
    """
    def __init__(self, dataset_raw, transform=None, columns: tuple=[0, 1]):
        super(WafermapDataset, self).__init__()
        if columns:
            assert all(col in dataset_raw.keys() for col in columns), \
                f'columns {columns} not in dataset_raw.keys(): {dataset_raw.keys()}'
            
            # 根據 dataset_raw 的來源, 決定如何改變資料型態
            if isinstance(dataset_raw[columns[0]], np.ndarray) and isinstance(dataset_raw[columns[1]], np.ndarray): # 資料來源是資料集 (是 ndarray)
                self.wafermap = torch.tensor(dataset_raw[columns[0]], dtype=torch.float)
                self.label = torch.tensor(dataset_raw[columns[1]], dtype=torch.float)
            else:   # 其餘的資料來源都是自產的 perfect dataset (是 tensor)
                self.wafermap = dataset_raw[columns[0]].float()
                self.label = dataset_raw[columns[1]].float()
        self.transform = transform
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        # __getitem__ 函式應回傳 CPU 上的張量, 而將張量移動到 GPU (透過 .to(device)) 的工作應交給訓練迴圈來處理 (在訓練迴圈中, 拿到 batch 之後再搬移). 
        # 如果在這裡就將資料移至 GPU, 當 DataLoader.num_workers > 0 時, 每個 worker 都會嘗試在自己的程序中初始化 GPU, 這會導致錯誤或效能問題. 
        wafermap = self.wafermap[index]
        label = self.label[index]
            
        wafermap = wafermap / 2   # 若不做這一行, wafermap 的 element 會是 {0, 1, 2}, 在經過 transform 的 ToPILImage 後, 原本的 {0, 1, 2} 會變成 {0, 0.9961, 1}.

        if self.transform: 
            # transform 一定要在 value / 2 之後進行
            # 因為 transform 裡面的 ToTensor 在轉換數值時, 
            # 即便原始數值已經介於 [0.0, 1.0], 仍然會做某種映射, 使得輸出值有所改變 (似乎是這樣), 例如 0.5 轉換後會變成 0.498....
            # 另外, transforms.ToTensor() 會自己將 (H, W) 轉換為 (1, H, W), 所以在這裡不用手動多加一個維度.
            wafermap = self.transform(wafermap)
            wafermap = rectify_pixel_values(wafermap)

        return (wafermap, label)



class MyDataLoader(torch.utils.data.DataLoader):    # 為了能自訂 dataloader 的名稱, 只好自己寫一個 DataLoader, 加入 name 屬性.
    def __init__(self, dataloader_name: str, dataset, batch_size=batch_size, shuffle=None, config: dict=dataset_config, 
                #  num_workers=4, pin_memory=True
                # 因為訓練都在 GPU 上進行, 所以預設 num_workers=4, pin_memory=True. 但是這樣設定的話, 
                # (1) 有時候 num_workers 會出問題 
                # (2) 在做資料型態檢查的時候, 雖然只要檢查某個 batch 或某個 sample, 但 DataLoader 還是會啟動所有的 workers 去讀取整個 dataset, 這樣會花比較多時間.
                 ):   
        super(MyDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                        #    num_workers=num_workers, pin_memory=pin_memory, 
                                           )
        # persistent_workers=True 讓 worker 在 epoch 之間不關閉，減少重新啟動的開銷
        self.name = dataloader_name
        self.config = {
            'name': dataloader_name,
        }
        self.config.update(config)

    def __iter__(self):
        return super(MyDataLoader, self).__iter__()



def generate_perfect_wafermap_and_label(n_batch: int, size: tuple=(64, 64)) -> torch.Tensor:
# def generate_perfect_wafermap_and_label(n_batch: int, size: tuple=wafer_resize_scale) -> torch.Tensor:
    """
    Generate a perfect wafer map and its corresponding label.   
    The function uses vectorized operations for efficiency.  這個函式使用向量化運算來提高效率.

    :param n_batch: Number of wafer maps to generate.
    :param size: Size of the wafer map as a tuple (height, width). Default is (64, 64).
    :return: A dictionary containing:

        * '0': A tensor of shape (n_batch, 1, height, width) representing the perfect wafer maps.
        * '1': A tensor of shape (n_batch, 8) representing the labels (all zeros).
    """
    assert isinstance(size, tuple) and len(size) == 2, "Size must be a tuple of (height, width)."

    h, w = size
    center_x, center_y = w / 2, h / 2
    radius_sq = (min(h, w) / 2) ** 2    # radius_sq = radius squared = r ** 2

    # 建立網格座標
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')  # shape: (h, w)
    # ↑ indexing='ij' 確保 y 對應到第一維 (height), x 對應到第二維 (width).
    # indexing='xy' 是預設值, 但不適合影像處理

    # 使用向量化運算判斷像素是否在圓形內
    dist_sq = (x - center_x + 0.5)**2 + (y - center_y + 0.5)**2 # dist_sq = distance square     # +0.5 是為了讓圓心落在像素的正中央
    wafermap_perfect = (dist_sq <= radius_sq).float() # .to(device) 因為使用 pin_memory=True, 所以 DataLoader 會自動幫忙把資料搬到 GPU 上, 不需要在這裡手動搬移.
    # ↑ wafermap value range: 0.0 ~ 1.0
    # 這邊不用讓 wafermap_perfect 的 range 變得跟原始資料集一樣是 (0, 1, 2), 
    # 因為 0=未使用, 1=正常, 2=缺陷, 而 perfect wafermap 裡面沒有缺陷, 所以只需要 0 和 1 就夠了.

    wafermap_perfect = wafermap_perfect.expand(n_batch, 1, h, w)
    # label_perfect = torch.zeros((n_batch, 8), dtype=torch.float).to(device)
    label_perfect = torch.zeros((n_batch, 8)) # .to(device) 因為使用 pin_memory=True, 所以 DataLoader 會自動幫忙把資料搬到 GPU 上, 不需要在這裡手動搬移.

    result = {0: wafermap_perfect, 
              1: label_perfect}
    return result


def generate_perfect_loader(batch_size=64, wafer_resize_scale=wafer_resize_scale):
    """
    :param batch_size: 批次大小
    :param wafer_resize_scale: 晶圓圖的尺寸 (寬和高)
    :return: 包含完美晶圓圖的 DataLoader
    """
    # 生成完美晶圓圖數據集
    perfect_set = generate_perfect_wafermap_and_label(n_batch=batch_size, size=wafer_resize_scale)
    perfect_set = WafermapDataset(perfect_set, transform=None) 
    # ↑ 不用 transform, 因為 generate_perfect_wafermap_and_label 回傳的資料就已經是 tensor, 
    # 而且 size 也是 wafer_resize_scale
    perfect_loader = MyDataLoader('perfect_loader', perfect_set, batch_size=batch_size)
    return perfect_loader