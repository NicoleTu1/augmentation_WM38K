import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # 用於 GPU 加速運算 # 舊版本是 from torch.cuda.amp import GradScaler, autocast

import time
from collections import defaultdict
from loss_functions import loss_function_colorizeVAE

from utils import device, assert_nan_and_inf, Logger, rectify_pixel_values
from dataset_utils import dataset_config, generate_perfect_wafermap_and_label



# from model_R import vgg_block # 這句話會造成 循環匯入(circular import). 因為 C 要匯入 R, 而 R 也要匯入 utils, utils 又匯入 C.
def vgg_block(in_channels, out_channels): # 從 model_R.py 複製過來的
    """
    `nn.Sequential(Conv2d, BatchNorm2d, ReLU, MaxPool2d)`
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2), 
    )


class C_ColorizerNetwork(nn.Module):
    """
    The parameter for building instance is as follows:

    :param model_name: Name of the model (optional).
    :param image_size: Size of the input images, e.g., (64, 64).
    :param R_info: Dictionary containing information about the R network, e.g., {'Rref_path': '...', 'Rgray_path': '...'}.

    The parameters for the `forward()` function are as follows:

    :param gray_image: 灰階影像
    :param Wab: 色彩權重
    :param Attenmap: 注意力圖
    :param ref_image: 參考影像
    :param ~~X1ab_t_minus_1~~ref_image: 影片著色文獻中, "前一幀的著色結果". 晶圓圖因沒有前一幀的著色結果, 所以用 `ref_image` 取代
    :return: 彩色晶圓圖 (即影片著色文獻中, "當前幀的預測 ab 通道 `Xab_t`")
    """

    def __init__(self, model_name, image_size: tuple, R_config: dict, 
                 R_info: str='', 
                 z_dim=256, 
                 encoder_input_channels=4):
        """
        :param model_name: Name of the model (optional).
        :param image_size: Size of the input images, e.g., (64, 64).
        :param R_info: Dictionary containing information about the R network, e.g., {'Rref_path': '...', 'Rgray_path': '...'}.
        :param encoder_input_channels: Default=4. If do ablation for Attention Mechanism, set to 2.
        """
        super(C_ColorizerNetwork, self).__init__()

        # 為了方便後續將所有 input 調整成同樣的尺寸, 這裡先建立一個變數, 儲存 upsample 的目標尺寸.
        self.upsample_target_size = image_size

        # 編碼器部分
        # 預期輸入通道數: gray_image(1) + Wab(1) + Attenmap(1) + ref_image(1) = 4 通道
        # encoder_input_channels = 1 + 1 + 1 + 1
        # 為了做消融實驗, 驗證移除 Wab(1) + Attenmap(1) 後, 資料增強的效果如何, 所以要在建立實例的時候就定義 encoder_input_channels.
        # 因此一般情況下, encoder_input_channels=4, 但若要做消融實驗則 encoder_input_channels=2.
        assert encoder_input_channels in (2, 4), f'`encoder_input_channels` must be 4 (for Attention Mechanism) or 2 (for no Attention Mechanism).'
        self.encoder_input_channels = encoder_input_channels
        ##### ↑這個部分應該是有修改空間的, 
        # 在影像著色的文獻中, 需要使用前一幀的著色結果 (X1ab_t_minus_1) 來輔助當前幀的著色,
        # 所以輸入通道數為 9 通道, 即 gray_image(1) + Wab(2) + Attenmap(1) + ref_image(3) + X1ab_t_minus_1(2) = 9 通道.
        # 但是在晶圓圖的著色中, 不需要前一幀的著色結果, 目前的程式已經將這個部分拿掉.
        # 未來也許可嘗試使用另一張 `ref_image` 取代前一幀著色結果...?這樣就可以學到兩張 ref_image...? 
        # 而且, 文獻中的 Wab 是 2 通道, 代表 ab 色彩空間的兩個維度,
        # 但晶圓圖是 1 通道, 代表灰階值的加權. 這部分也許可以在消融實驗比較, 看拿掉wab對結果的影響為何.
                
        # 計算編碼器最終輸出的展平尺寸，用於連接全連接層
        self.encoder_output_spatial_size = self.upsample_target_size[0] // 2 ** 3 # 冪次 = 編碼器中 MaxPool2d 的次數
        self.encoder_output_channels = 256 # 編碼器輸出的通道數
        latent_spatial_dim = self.encoder_output_channels * self.encoder_output_spatial_size ** 2 # 即 (C * H * W) = (256 * encoder_output_spatial_size ** 2)

        self.encoder = nn.Sequential(
            vgg_block(encoder_input_channels, 64),  # (B, 4, 64, 64) -> (B, 64, 32, 32)
            vgg_block(64, 128),  # (B, 64, 32, 32) -> (B, 128, 16, 16)
            vgg_block(128, self.encoder_output_channels),  # (B, 128, 16, 16) -> (B, C, 8, 8)
            # vgg_block(256, 512),  # (B, 256, 8, 8) -> (B, 512, 4, 4)
        )

        
        # 潛在空間維度 (z_dim)
        self.z_dim = z_dim
        self.fc_mu = nn.Linear(latent_spatial_dim, self.z_dim)
        self.fc_logvar = nn.Linear(latent_spatial_dim, self.z_dim)
        # mu 跟 logvar 不可以加入 relu 或 sigmoid 或任何活化函式, 因為 mu 和 logvar 需要是實數, 不能被限制在某個範圍內.
        # Gemini: mu 跟 logvar 不適合加入 BatchNorm1d.
            # BatchNorm1d 在訓練時會根據 小批次 (minibatch) 的平均值和變異數來正規化資料.  
            # 如果你的批次大小很小, 或者資料分佈不均勻, BatchNorm1d 的均值和變異數就會變得不穩定, 這可能導致梯度在反向傳播時爆炸, 進而產生 NaN. 
            # 與 VAE 架構不搭: mu 和 logvar 在 VAE 中代表潛在空間的高斯分佈參數. 理論上, 它們的值應該是沒有上下限的, 可以為任意實數. 
            # 雖然 BatchNorm1d 不會像 Sigmoid 那樣限制值的範圍, 但它會將輸出值正規化到一個平均值為 0、變異數為 1 的分佈. 
            # 這會干擾模型學習潛在空間分佈的過程，特別是當變異數 logvar 應該學習負值時 (當變異數小於 1 時)。
        

        # 解碼器部分
        # 解碼器的輸入是 latent_variant `z`
        
        # 這一塊是拿來提取"調整尺寸後的ref_image"的特徵, 稱為 `conditioned_features`.
        # 他會先跟 `z` 拼接, 然後解碼, 以提供額外的上下文資訊. 
        self.conditioned_features_channels = 64
        self.conditioning_features_extractor = nn.Sequential(
            vgg_block(1, 32),  # (B, 1, 64, 64) -> (B, 32, 32, 32)
            vgg_block(32, 64),  # (B, 32, 32, 32) -> (B, 64, 16, 16)
            vgg_block(64, self.conditioned_features_channels),  # (B, 64, 16, 16) -> (B, C, 8, 8)
            # vgg_block(64, 64),  # (B, 64, 16, 16) -> (B, 64, 8, 8)
        )
        
        # 將潛在空間的 `z` 投影到與 `conditioned_features` 相同的空間, 以利後續拼接.
        self.z_projected_channels = 64
        # latent_to_spatial_initial 的輸出尺寸 = z_projected_channels * H * W
        self.latent_to_spatial_initial = nn.Linear(self.z_dim, self.z_projected_channels * self.encoder_output_spatial_size ** 2) # 目標通道 * H * W
        # ↑ initial 在這裡的意思是將潛在空間的特徵映射回初始的空間特徵圖, 初始的空間特徵圖是指解碼器開始處理的特徵圖(??). 

        # 反向卷積, 以恢復空間解析度
        # 初始輸入通道為 (z_projected 通道數 + conditioned_features 通道數) = (64 + 64) = 128
        self.decoder_input_channels = (self.z_projected_channels) + (self.conditioned_features_channels)
        self.decoder = nn.Sequential(   
            nn.ConvTranspose2d(self.decoder_input_channels, 64, kernel_size=4, stride=2, padding=1), # -> 16x16 
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), # -> 64x64
            nn.Sigmoid()    # 將輸出限制在0-1之間
        )

        self.config = {
            'name': model_name if model_name else __class__.__name__, 
            'upsample_target_size': image_size,
            'loss_function': None,    # will be updated during training phase
            'R_config': R_config, 
            'z_dim': self.z_dim,
            'encoder_input_channels': self.encoder_input_channels,
            'encoder_output_channels': self.encoder_output_channels,
            'conditioned_features_channels': self.conditioned_features_channels,
            'z_projected_channels': self.z_projected_channels,
            # 'encoder_output_spatial_size': self.encoder_output_spatial_size,
            # 'decoder_input_channels': self.decoder_input_channels,
            'train_loader': None,   # will be defined during training phase
            'validation_loader': None,   # will be defined during inference phase
        }
        
        if R_info:
            print('R_info will be deprecated in class C_ColorizerNetwork. The replacement is R_config(dict).')
            self.config['R_info'] = R_info

    def update_config(self, key: str, value):
        """
        Update the configuration dictionary with a new key-value pair.

        :param key: The key to update in the configuration dictionary.
        :param value: The new value to set for the specified key.
        :return: The updated configuration dictionary.
        """
        self.config[key] = value
        return self.config

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, gray_image, ref_image, Wab, Attenmap, X1ab_t_minus_1=None):
        
        # 先將所有的 input 調整為相同的尺寸 (即 upsample_target_size)
        ##### 以下插值函式若改為使用 `mode='bilinear', align_corners=False` 會不會使得生成影像的多樣性更高? 
        gray_image_resized = F.interpolate(gray_image, size=self.upsample_target_size, mode='nearest')
        ref_image_resized = F.interpolate(ref_image, size=self.upsample_target_size, mode='nearest')
        if self.encoder_input_channels == 4:
            Wab_resized = F.interpolate(Wab, size=self.upsample_target_size, mode='nearest')
            Attenmap_resized = F.interpolate(Attenmap, size=self.upsample_target_size, mode='nearest')
            # 拼接所有調整後的輸入，作為編碼器的輸入
            encoder_input = torch.cat([gray_image_resized, Wab_resized, Attenmap_resized, ref_image_resized], dim=1)
            # shape = (B, 4, 32, 32)
        else:
            encoder_input = torch.cat([gray_image_resized, ref_image_resized], dim=1)
        # print(f'encoder_input.abs().max(): {encoder_input.abs().max()}')
        
        
        # 編碼器前向傳播, 然後展平以便傳入全連接層
        encoded = self.encoder(encoder_input)   # shape = (B, 256, encoder_output_spatial_size, encoder_output_spatial_size)
        encoded_flat = encoded.view(encoded.size(0), -1)    # shape = (B, 256 * encoder_output_spatial_size * encoder_output_spatial_size)


        # 取得 mu 和 logvar
        mu = self.fc_mu(encoded_flat)   
        logvar = self.fc_logvar(encoded_flat)
        
        # 重參數化, 採樣潛在變數 z
        z = self.reparameterize(mu, logvar)


        # 解碼器前向傳播
        # 將潛在空間的變數 `z` 投影回空間特徵圖 (將 `z` 投影到與 `conditioned_features` 相同的空間)
        z_projected = self.latent_to_spatial_initial(z) # shape: (B, 64 * encoder_output_spatial_size * encoder_output_spatial_size)
        z_projected = z_projected.view(z_projected.size(0), self.z_projected_channels, self.encoder_output_spatial_size, self.encoder_output_spatial_size) 

        # 從 `ref_image_resized` 提取條件特徵，用於解碼器的輸入 [參考原文的 Figure 4]  
        # 這邊再提取一次 ref_image_resized 的特徵, 是為了了給解碼器使用. 若不提取, 則解碼器無法使用 ref_image 的資訊, 那麼解碼器就無法生成有意義的彩色晶圓圖.
        # ref_image_resized shape: (B, 1, 64, 64)
        conditioned_features = self.conditioning_features_extractor(ref_image_resized)   # (B, 64, 8, 8)

        # 拼接潛在空間特徵和條件特徵，作為解碼器的輸入
        # assert z_projected.shape[3] == conditioned_features.shape[2], \
        #     f'Expected `z_projected.shape[-1]` matches `conditioned_features.shape[-1]`. But\n' \
        #     f'z_projected.shape: {z_projected.shape}.\n'    \
        #     f'conditioned_features shape: {conditioned_features.shape}.'
        decoder_input = torch.cat([z_projected, conditioned_features], dim=1) # (B, 128, 8, 8)

        # 透過解碼器生成 ab 通道 (因為是 wafer map 所以改成 1 通道)
        x_recon = self.decoder(decoder_input) # 輸出 (B, 1, 32, 32)

        return x_recon, mu, logvar # 回傳 x_recon 和 VAE 的 mu/logvar，以便計算 KL 散度損失



def train_colVAE(model, optimizer, scheduler, 
                 epochs, augment_loader, 
                 model_F, loss_weights: dict[list], 
                 window_size: int, 
                 scaler: GradScaler, logger: Logger,
                #  is_AttentionMechanism: bool=True, # 直接用 colVAE.encoder_input_channels 來判斷. 4=使用注意力機制, 2=不使用注意力機制
                 ):
                #  normalization_method='none', statistic_for_normalization=None,
                #  goal_percent=1, 
                #  KL_annealer: KL_Annealer = None,):
    model.train()
    
    w_sum = sum([w[0] for w in loss_weights.values() if w[0] > 0])
    assert 0.98 <= w_sum <= 1.2, f'Sum of loss weights must be 1.0. But got {w_sum}.'

    log_loss, log_lr = [], []
    log_each_loss = defaultdict(list) # 每個 key 的 value 是一個 list, 用來存放每個 epoch 的損失值.
    time_start = time.time()

    gray_images = generate_perfect_wafermap_and_label(dataset_config['batch_size'], dataset_config['wafer_resize_scale'])[0].to(device)
    for epoch in range(epochs):
        train_loss = 0.0
        each_loss_sum = defaultdict(float) # 用來累加每個 batch 的損失值, 最後再除以 batch 數量, 得到每個 epoch 的平均損失值.

        for ref_image, failureType in augment_loader:
            optimizer.zero_grad()
            ref_image = ref_image.to(device)
            gray_image = gray_images[:ref_image.size(0), :, :, :]  # shape=(B, 1, H, W)

            # current_kld_annealer_weight = KL_annealer.get_weight()

            if model.encoder_input_channels == 4: # 使用注意力機制
                model_F.eval()
                with torch.no_grad():   # model_F 在 eval 模式下不需要梯度, 可以包在 torch.no_grad() 內, 可以避免儲存梯度計算圖, 讓訓練更高效. 
                    wab, attenmap = model_F(gray_image, ref_image)
                with autocast(device_type=device.type):
                    colored_image, mu, logvar = model(gray_image, ref_image, wab, attenmap) 
            else: # model.encoder_input_channels == 2
                # 因為不使用注意力機制, 所以不叫 model_F 做事
                with autocast(device_type=device.type):
                    colored_image, mu, logvar = model(gray_image, ref_image, None, None)

            with autocast(device_type=device.type):
                loss, each_loss_dict, each_loss_note = loss_function_colorizeVAE(colored_image, ref_image, mu, logvar, loss_weights, 
                                                                                                    window_size=window_size
                                                                                                    )
                                                                                                    # goal_percent, 
                                                                                                    # normalization_method=normalization_method,
                                                                                                    # statistic=statistic_for_normalization, 
                                                                                                    # KL_annealer_weight=current_kld_annealer_weight)
                # each_loss_dict 已經除以 batch_size 了
                # loss 要除以 batch_size 嗎? 不用, 因為 loss_function_colorizeVAE 裡面的每個 loss 都已經除以 batch_size 了
                # each_loss_dict 的內容是: {'mse': mse_loss, 'kld': kld_loss, 'dice': dice_loss, 'ssim': ssim_loss}

            # KL_annealer.step()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ##### 梯度剪裁: 可以嘗試不同的 max_norm 值, 例如 1.0 / 5.0 / 10.0

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach().item()  # 累加每個 batch 的 loss 值
            
            # 累加 each_loss_dict 裡面的每個 loss 到 each_loss_sum
            for loss_label in each_loss_dict.keys():
                each_loss_sum[loss_label] += each_loss_dict[loss_label].detach().item()    # 累加每個 batch 的 loss 值, 最後再計算平均值.
                # each_loss_sum[loss_label].append(each_loss_dict[loss_label].detach())   # 儲存每個 batch 的 loss 值, 以便觀察每個 epoch 的 loss 變化趨勢. 也可以用來計算平均值 (但會比上面那行慢一點).

        l = len(augment_loader)
        loss_avg = train_loss / l  # 為什麼這邊要除以 batch 數量? 因為每個訓練步驟都處理一個 batch 的資料. 這行程式碼計算了整個 epoch 的平均損失.
        scheduler.step(loss_avg) 
        log_loss.append(loss_avg)
        # log_loss.append(loss_avg.item())

        for loss_label in loss_weights.keys():
            s = each_loss_sum[loss_label] / l   # 計算每個 epoch 的平均損失值 (若 each_loss_sum[loss_label] 是 float)
            log_each_loss[loss_label].append(s) # (若 each_loss_sum[loss_label] 是 float)
            # s = sum(each_loss_sum[loss_label]) / l    # 計算每個 epoch 的平均損失值 (若 each_loss_sum[loss_label] 是 list)
            # log_each_loss[loss_label].append(s.item())    # (若 each_loss_sum[loss_label] 是 list)
            # log_each_loss[loss_label] = each_loss_sum[loss_label]
        log_lr.append(scheduler.get_last_lr()[0])
        if (epoch + 1) % 10 == 0: 
            message = f'Epoch [{epoch+1:3d}/{epochs:3d}], loss: {loss_avg:.9f}, lr: {scheduler.get_last_lr()[0]:.9f}'
            logger.log(message, is_print=True)
    
    loss_equation_str = ''
    for loss_label, (loss_weight, adjustment) in loss_weights.items():
        # 計算方式: loss_weight * (|loss_label - adjustment|) 
        loss_equation_str += f'{loss_weight:.4f} * (|{loss_label}-{adjustment}|) + '
        # 計算方式: loss_weight * max(0, loss_label - adjustment) + adjustment
        # loss_equation_str += f'[{loss_weight:.4f} * max(0, {loss_label}-{adjustment}) + a] + '
    loss_equation_str = loss_equation_str.rstrip(' + ')
    if window_size != 11:
        loss_equation_str += f'\nSSIM_window_size={window_size}'
    # if KL_annealer is not None:
    #     loss_equation_str += f'\n  KL_Annealer(total_steps={KL_annealer.total_steps}, max_weight={KL_annealer.max_weight})' 

    model.update_config('loss_function', loss_equation_str) 
    
    time_end = time.time()
    time_duration = time_end - time_start
    timestamps = (time_start, time_duration, time_end)

    # dataloader_name = augment_loader.name if hasattr(augment_loader, 'name') else None
    model.update_config('train_loader', augment_loader.config)

    return log_loss, log_lr, timestamps, log_each_loss, each_loss_note
    # return log_loss, log_lr, timestamps, dataloader_name, each_loss_sum, each_loss_note  # 用於統計各 loss 的統計量. 此行會回傳 each_loss_sum, 它記錄每個batch的loss, 不是每個epoch的平均loss.
















@torch.no_grad()
def generate_colorized_wafermap(colVAE, module_F, 
                                dataset=None,
                                augment_loader=None, 
                                n_sample: int=5, 
                                # is_no_AttentionMechanism=False, 
                                ) -> list:
    """
    使用訓練好的 colVAE 生成彩色晶圓圖。

    :param colVAE: 訓練好的 Color Network C 模型
    :param module_F: Feature Processing Network F 模型
    :param augment_loader: 用於生成參考影像的資料載入器
    :param n_sample: 生成的彩色晶圓圖數量
    :return colored_wafermaps: 生成的彩色晶圓圖 (list)
    :return colVAE.loss_function: colVAE 使用的損失函數描述字串
    """
    colVAE.eval()

    if augment_loader is None:  # 如果沒有提供 augment_loader, 就使用 dataset 裡面的第 100 張影像作為參考影像
        colored_wafermaps = torch.tensor([]).to(device)  # 儲存生成的彩色晶圓圖
        ref_image = dataset[100][0].unsqueeze(0)  # 取得第 100 張影像作為參考影像
        ref_image = ref_image.to(device)
        ref_image = ref_image.repeat(n_sample, 1, 1, 1)

        gray_image = generate_perfect_wafermap_and_label(n_sample, colVAE.upsample_target_size)[0].to(device)  # shape=(n_sample, 1, H, W), value=0/0.5

        if colVAE.encoder_input_channels == 4: # 有注意力機制
            module_F.eval()
            Wab, Attenmap = module_F(gray_image, ref_image) # 使用 F 計算 Wab 和 Attenmap
            colored_image, _, _ = colVAE(gray_image, ref_image, Wab, Attenmap)  # 使用 Color Network C 生成彩色晶圓圖
        else: # 沒有注意力機制
            colored_image, _, _ = colVAE(gray_image, ref_image, None, None)  # 使用 Color Network C 生成彩色晶圓圖

        colored_wafermaps = rectify_pixel_values(colored_image)
        return colored_wafermaps, ref_image[0]
            
    else:   # 如果有提供 augment_loader, 就從中取出 augment_loader.batch_size 張影像作為參考影像
        # colored_wafermaps = torch.tensor([]).to(device)  # 儲存生成的彩色晶圓圖 # 改成 list 應該會執行的比較快?
        # colored_labels = torch.tensor([], dtype=torch.int).to(device)  # 儲存生成的彩色晶圓圖 # 改成 list 應該會執行的比較快?
        colored_wafermaps = []  # 儲存生成的彩色晶圓圖 
        colored_labels = []  # 儲存生成的彩色晶圓圖

        for ref_wafer, ref_label in augment_loader:
            ref_image = ref_wafer.to(device)
            ref_label = ref_label.to(device)
            batch_size = ref_image.size(0)

            gray_image = generate_perfect_wafermap_and_label(batch_size, colVAE.upsample_target_size)[0].to(device)  # shape=(B, 1, H, W), value=0/0.5

            if colVAE.encoder_input_channels == 4: # 有注意力機制
                module_F.eval()
                Wab, Attenmap = module_F(gray_image, ref_image) # 使用 F 計算 Wab 和 Attenmap
                colored_image, _, _ = colVAE(gray_image, ref_image, Wab, Attenmap)  # 使用 Color Network C 生成彩色晶圓圖
            else: # 沒有注意力機制
                colored_image, _, _ = colVAE(gray_image, ref_image, None, None)  # 使用 Color Network C 生成彩色晶圓圖

            # colored_wafermaps = torch.cat([colored_wafermaps, colored_image], dim=0)
            # colored_labels = torch.cat([colored_labels, ref_label], dim=0)
            colored_wafermaps.append(colored_image)
            colored_labels.append(ref_label)

        # colored_wafermaps = rectify_pixel_values(colored_wafermaps)
        colored_wafermaps = torch.cat(colored_wafermaps, dim=0)
        colored_wafermaps = rectify_pixel_values(colored_wafermaps)
        colored_labels = torch.cat(colored_labels, dim=0)
        
        return colored_wafermaps, colored_labels
    

