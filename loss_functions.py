import torch
import torch.nn.functional as F

def dice_loss_wafer(pred, target, eps=1e-6, adjustment=0):
    """
    針對晶圓圖優化的 Dice Loss. 實際衡量的是：
    * 空間重疊程度：不只是數量, 更重要的是位置的一致性
    * 形狀相似性：確保缺陷區域的幾何結構和分布模式相似
    * 區域一致性：比單純的像素分類更關注連續區域的正確性
    Dice 的值介於 0 到 1 之間, 1 表示完美重疊, 0 表示完全不重疊. 
    """
    # pred = pred.to(device)
    # target = target.to(device)

    # 將預測值和目標值都壓縮到 [0,1] 範圍
    # pred = torch.sigmoid(pred)
    # target = target.clamp(0, 1) # 雖然 clamp 是不可微分的 operation, 但在這裡並不會導致反向傳播出錯. 因為: 
    # 反向傳播的目的是計算損失函數對模型參數的梯度. 梯度會從損失值反向傳播到模型的輸出 pred, 然後再傳播到模型內部的每一層. 
    # 然而, 梯度不會傳播到 target, 因為 target 是真實標籤（ground truth）, 反向傳播過程中不需要去更新它. 

    
    # 計算交集和聯集
    intersection = (pred * target).sum(dim=(2, 3))  
    # assert_nan_and_inf(intersection, f'class dice_loss_wafer: intersection')

    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # dim=(2,3) 的意思是, 對 (H,W) 維度求和
    # assert_nan_and_inf(union, f'class dice_loss_wafer: union')

    # calculate the Dice value
    # print('intersection:', intersection, '\n', 'union:', union)
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1 - dice).mean()



def gaussian_window(size, sigma=1.5):
    """
    :param size: 窗口大小
    :param sigma: 高斯分佈的標準差, 也就是機率密度函數的寬度.
        1.5 是一個常見的選擇, 可以根據需要調整. 0.5 會使得高斯分佈更陡峭, 2.0 會使得高斯分佈更平緩.
        這個參數影響 SSIM 計算中局部區域的權重分佈.
        這個值不宜過小, 否則會導致窗口內的像素權重過於集中, 失去平滑效果;
        也不宜過大, 否則會導致窗口內的像素權重過於分散, 無法有效捕捉局部結構資訊.
    :return: 高斯窗口
    """
    # coords = torch.arange(size, dtype=torch.float)    # coords 是 coordinates (座標)
    coords = torch.arange(size)    # coords 是 coordinates (座標)
    
    # 將座標中心化. 例如, size=11, 則 coords = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    coords -= size // 2 
    
    # 計算高斯分佈 (根據高斯分佈的公式)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))    # gaussian formula for 1D # shape=(size,)
    # 分子: (2 * sigma ** 2) 是高斯分佈的(標準差的平方乘以 2), 這個值控制高斯分佈的寬度.
    # 分母: coords ** 2 是將每個座標平方, 這樣可以確保距離中心點越遠的座標, 對應的值越小. 
    # torch.exp 用於計算 e 的冪次, e 約等於 2.71828, torch.exp 的輸出值域是 (0, +∞). 
    # g 的值域是 (0, 1], 因為當 coords 越接近 0 時, g 越接近 1; 當 coords 越遠離 0 時, g 越接近 0.

    g /= g.sum()    # normalize to make sum(g)=1
    # g 是窗口內的權重, 將 g 正規化, 可以確保窗口內所有像素的權重總和為 1.
    # return g.unsqueeze(0).unsqueeze(0)  # shape=(1, 1, size)

    # Create 2D Gaussian kernel using outer product
    window_1d_x = g.unsqueeze(0).unsqueeze(0)   # shape=(1, 1, size)
    window_1d_y = g.unsqueeze(0).unsqueeze(0).transpose(1, 2)
    window_2d = torch.matmul(window_1d_y, window_1d_x)
    # window_2d /= window_2d.sum()  # normalize to make sum(window_2d)=1  # 這個步驟其實是多餘的, 因為前面已經正規化過了.
    return window_2d  # shape=(1, 1, size, size)


def ssim_wafer(pred, target, window_size=11, epsilon=1e-7, reduction='mean', adjustment=0):
    """
    Structural Similarity Index Measure, SSIM
    針對晶圓圖的 SSIM 損失

    :param pred: 繪製的晶圓圖
    :param target: 參考的晶圓圖
    :param window_size: 用於計算局部統計量的高斯窗口大小. 必須是奇數, 以確保窗口有一個中心像素.
        一般來說, window_size 越大, SSIM 越能捕捉到更大範圍的結構資訊, 但計算量也會增加.
        通常選擇 11 或 7 作為窗口大小, 這是經過實驗驗證的常見選擇.
        但具體選擇多少, 仍需根據應用情境和影像特性來調整.
        例如, 對於高解析度圖像, 可以考慮使用較大的窗口; 對於低解析度圖像, 則可以使用較小的窗口.
    :param reduction: 指定如何聚合損失值 ('mean', 'sum', 'none').
    :return: SSIM 損失值
    """
    
    # pred = pred.to(device)
    # target = target.to(device)
    
    # 建立高斯窗口 (就是卷積核)
    # assert pred.size(1) == target.size(1), f'pred and target must have the same number of channels. But got pred.size(1): {pred.size(1)}, target.size(1): {target.size(1)}'
    window = gaussian_window(window_size).to(pred.device)   # shape=(1, 1, window_size)
    window = window.expand(pred.size(1), 1, window_size, window_size)   
    # ↑ window 就是卷積核, shape=(C, 1, window_size, window_size)
    
    # 計算平均數
    # ↓ 利用 F.conv2d 的功能, 以矩陣運算的方式, 在多個獨立的通道上同時、高效地計算局部 mu. 
    # 這比使用迴圈（for loop）來逐點計算要快得多, 也更符合 GPU 的平行運算特性. 
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.size(1))
    # groups: 將輸入和輸出通道劃分為組. 
    # 當 groups 的值等於輸入通道數 `pred.size(1)` 時, PyTorch 會為每個輸入通道分配一個獨立的卷積核. 
    # 這代表每個輸入通道的卷積運算都是獨立進行的, 它們之間不會互相影響. 
    # 這種特殊的卷積被稱為深度可分離卷積（depthwise separable convolution）中的深度卷積（depthwise convolution）. 
   
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 計算變異數和共變異數 (一樣透過 F.conv2d 來計算, 以達到高效計算的目的)
    # 這裡的邏輯是基於：sigma_x^2 = E[x^2] - E[x]^2
    # sigma1_sq = 平均值的平方的平均 減去 平均值的平方
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=pred.size(1)) - mu1_mu2

    # assert_nan_and_inf(sigma1_sq, f'ssim_wafer: sigma1_sq')
    # assert_nan_and_inf(sigma2_sq, f'ssim_wafer: sigma2_sq')
    # assert_nan_and_inf(sigma12, f'ssim_wafer: sigma12')
    sigma1_sq = sigma1_sq.clamp(min=1e-7)
    sigma2_sq = sigma2_sq.clamp(min=1e-7)
    sigma12 = sigma12.clamp(min=-1e-7)

    # 計算 SSIM
    # SSIM 的完整公式包含三個部分: 亮度相似性、對比度相似性、結構相似性. 算式簡化合併後為:
    # ssim_loss = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + epsilon)
    # 其中有常數 C1 和 C2 用於穩定計算, 避免分母為零.
    # 但是晶圓圖沒有顏色資訊, 所以我們只計算結構 (structure) 相似性分量: 
    # S(x, y) = (sigma_xy + C3) / (sigma_x * sigma_y + C3)  , 其中 C3 = C2 / 2. 
    # 常數項 C 是為了防止分母為零,  並且應與輸入數據的動態範圍 L 有關: C2 = (k2 * L) ** 2 
    # Gemini: 標準 SSIM 常數計算, 通常 k_2=0.03 (待求證). 而 L 在晶圓圖中是 1 (因為輸入圖像已經被正規化到 [0, 1] 範圍內).
    C3 = (0.03 * 1) ** 2 / 2  
    # print('(ssim_wafer): ')
    # print(f'{sigma1_sq.min()}, {sigma1_sq.max()}')
    # print(f'{sigma2_sq.min()}, {sigma2_sq.max()}')
    # print(f'{sigma12.min()}, {sigma12.max()}')
    # print(f'分子={(sigma12 + C3).min()}, {(sigma12 + C3).max()}')
    # print(f'分母={(torch.sqrt(sigma1_sq * sigma2_sq) + C3).min()}, {(torch.sqrt(sigma1_sq * sigma2_sq) + C3).max()}')
    ssim_loss = (sigma12 + C3) / (torch.sqrt(sigma1_sq * sigma2_sq) + C3) # range=[-1, 1]
    ssim_loss = ssim_loss.clamp(min=-1.0, max=1.0)  # SSIM 的值域理論上是 [-1, 1], 但有時候計算會超出這個範圍, 所以進行 clamp.

    return 1 - ssim_loss.mean()   # SSIM range=[0,1], 越靠近 0 越不相似, 越靠近 1 越相似, 所以回傳的損失設為 (1 - SSIM)

    # if reduction == 'mean':
    #     # assert_nan_and_inf(ssim_loss, f'ssim_wafer: ssim_loss')
    #     # assert -1.0 <= ssim_loss.mean().item() <=1.0, f'SSIM mean value out of range [-1, 1]! SSIM.mean={ssim_loss.mean().item()}'
    #     return 1 - ssim_loss.mean()   # SSIM range=[0,1], 越靠近 0 越不相似, 越靠近 1 越相似, 所以回傳的損失設為 (1 - SSIM)
    # elif reduction == 'sum':
    #     print('(ssim_wafer) the `reduction` method is sum.')
    #     return 1 - ssim_loss.sum()
    # else:
    #     print('(ssim_wafer) the `reduction` method is none.')
    #     return 1 - ssim_loss
    


def kld_loss_wafer(mu, logvar, batch_size, adjustment=0):
    """
    計算 KL 散度損失 (Kullback-Leibler Divergence Loss) 用於變分自編碼器 (VAE).
    KL 散度衡量兩個機率分佈之間的差異。在 VAE 中, 我們希望潛在變數的分佈接近標準正態分佈 N(0, I).
    這樣可以確保潛在空間的連續性和可解釋性, 並促進生成樣本的多樣性.

    :param mu: 潛在變數的均值
    :param logvar: 潛在變數的對數方差
    :return: KL 散度損失值
    """
    # 原始公式 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    # assert_nan_and_inf(mu, f'kld_loss_wafer: mu')
    # assert_nan_and_inf(logvar, f'kld_loss_wafer: logvar')
    # assert_nan_and_inf(logvar.exp(), f'kld_loss_wafer: logvar.exp()')

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) # / batch_size
    # dim=1 的意思是對每個樣本的所有潛在維度求和, 因為 mu/logvar 的 shape 是 (B, latent_dim)
    # assert_nan_and_inf(kld_loss, f'kld_loss_wafer: kld_loss')

    return kld_loss.mean()  # 回傳 batch 中所有樣本的平均 KLD loss



def mse_loss_wafer(x_recon, x, batch_size, adjustment=0):
    """
    計算均方誤差損失 (Mean Squared Error Loss) 用於晶圓圖著色任務.
    MSE 損失衡量預測值與真實值之間的平方差異, 這有助於模型學習在像素層面上重建彩色晶圓圖.
    在晶圓圖著色任務中, MSE 損失有助於確保生成的彩色晶圓圖在整體亮度和顏色分佈上與真實圖像相似.

    :param x_recon: 重建後的彩色晶圓圖
    :param x: 真實的彩色晶圓圖
    :return: MSE 損失值
    """
    # mse_loss = F.mse_loss(x_recon, x, reduction='mean') # 計算 "每個 pixel 的平均誤差" (per pixel loss)
    mse_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size # 計算 "每張圖的平均誤差" (per image loss)

    return mse_loss



##### Normalize Loss Values #####
# # 以下函式被應用在 dictionary mapping, 所以所有函數的引數必須相同
# # all functions below are used in a dictionary mapping, so all functions must have the same arguments

# def _compute_zscore(loss_label, values, statistic, each_loss_note):
#     mean = statistic[loss_label]['mean']
#     # assert isinstance(mean, torch.Tensor), f'Expect mean is a torch tensor. But got {type(mean)} for {loss_label}.'
#     std = statistic[loss_label]['std']
#     # assert std > 0, f'std must be greater than 0 for z-score normalization. But got std={std} for {loss_label}.'
#     zscore = (values - mean) / std
#     # each_loss_note[loss_label] += f', compute z_score by (mean={mean:.3f}, std={std:.3f})'
#     return zscore

# def _compute_robust_zscore(loss_label, values, statistic, each_loss_note):
#     median = statistic[loss_label]['median']
#     mad = statistic[loss_label]['mad'] # median absolute deviation
#     # assert mad > 0, f'mad must be greater than 0 for robust z-score normalization. But got mad={mad} for {loss_label}.'
#     robust_zscore = (values - median) / mad
#     # each_loss_note[loss_label] += f', compute robust z_score by (median={median:.3f}, mad={mad:.3f})'
#     return robust_zscore

# def _compute_minmax_scaling(loss_label, values, statistic, each_loss_note):
#     vmin = statistic[loss_label]['min']
#     vmax = statistic[loss_label]['max']
#     # assert vmax > vmin, f'vmax must be greater than vmin for min-max normalization. But got vmin={vmin}, vmax={vmax} for {loss_label}.'
#     minmax = (values - vmin) / (vmax - vmin)
#     # each_loss_note[loss_label] += f', compute min-max by (vmin={vmin:.3f}, vmax={vmax:.3f})'
#     return minmax

# def _compute_log1p(loss_label, values, statistic, each_loss_note):
#     log_values = torch.log1p(values)  # log(1 + x)
#     # each_loss_note[loss_label] += f', compute log1p'
#     return log_values


# def normalize_loss(loss_label: str, loss_value: torch.Tensor, 
#                    each_loss_note: dict, 
#                    statistic: dict, method: str) -> torch.Tensor: 
#     # statistic: dict. 
#     # key = loss_label('mse', 'kld', 'dice', 'ssim')
#     # value = tuple(mean, std, median, mad)

#     if method == 'none': return loss_value

#     # Dictionary Mapping
#     normalization_map = {
#         'zscore': _compute_zscore,
#         'robust_zscore': _compute_robust_zscore,
#         'minmax': _compute_minmax_scaling,
#         'log1p': _compute_log1p,
#     }

#     assert method in normalization_map, f'Unsupported normalization method: {method}. Supported methods are: {list(normalization_map.keys())}'
    
#     compute_function = normalization_map[method]
#     loss_value = compute_function(loss_label, loss_value, statistic, each_loss_note)
#     loss_value = torch.abs(loss_value)  # Take absolute value to ensure non-negativity

#     return loss_value



from collections import defaultdict

def loss_function_colorizeVAE(x_recon, x, mu, logvar, 
                              loss_weights: dict[list[float, float]], 
                              window_size: int=11
                              ) -> tuple:
                            #   goal_percent=1, 
                            #   statistic=None, normalization_method=['none'], 
                            #   KL_annealer_weight: float=None) -> tuple:
    """
    :param x_recon: 重建的晶圓圖, shape=(B, 1, H, W)
    :param x: 目標晶圓圖(參考圖), shape=(B, 1, H, W) 
    :param mu: VAE 編碼器輸出的平均值
    :param logvar: VAE 編碼器輸出的對數變異數
    :param dict loss_weights: 損失權重. default=`dict: {'mse': 0.35, 'kld': 0.1, 'dice': 0.35, 'ssim': 0.2, 'contextual': 0.1}`

    :return: 總損失值 (weighted sum), 損失計算式字串, 各部分損失值的 tuple (mse_loss, kld_loss, dice_loss, ssim_loss) (original value, not weighted)
    """
    # x_recon = x_recon.to(device)
    # x = x.to(device)
    # mu = mu.to(device)
    # logvar = logvar.to(device)
    
    # assert_nan_and_inf(x_recon, f'loss_function_colorizeVAE: x_recon')
    # assert_nan_and_inf(x, f'loss_function_colorizeVAE: x')
    # assert_nan_and_inf(mu, f'loss_function_colorizeVAE: mu')
    # assert_nan_and_inf(logvar, f'loss_function_colorizeVAE: logvar')

    # normalize weights. make sum of weights = 1 → 在外面做就好了, 不然每次呼叫這個函式都要做 normalization 很浪費效能
    # loss_weight_sum = sum([weight[0] for weight in loss_weights.values() if weight[0] > 0])
    # if loss_weight_sum > 0: # 為了避免除以 0, 所以先檢查 sum 是否大於 0
    #     for loss_label in loss_weights.keys():
    #         if loss_weights[loss_label][0] > 0: loss_weights[loss_label][0] /= loss_weight_sum

    batch_size = x_recon.size(0)
    each_loss = {}
    # each_loss_note = defaultdict(str)  # 用於記錄每個損失的計算細節
    
    # def assert_nan_inf_for_loss_component(each_loss, loss_label, x_recon, x):
    #     assert not torch.isnan(each_loss[loss_label]).any(), f"{loss_label} is NaN! Check inputs. x_recon.max(): {x_recon.max():.9f}, x_recon.min(): {x_recon.min():.9f}, x.max(): {x.max():.9f}, x.min(): {x.min():.9f}"
    #     assert not torch.isinf(each_loss[loss_label]).any(), f"{loss_label} is Inf! Check inputs. x_recon.max(): {x_recon.max():.9f}, x_recon.min(): {x_recon.min():.9f}, x.max(): {x.max():.9f}, x.min(): {x.min():.9f}"


    # print(f'(loss function) statistic: {statistic}') ##### debug
    for loss_label, (weight, adjustment) in loss_weights.items():
        if weight == 0: continue
        # 算 loss_value
        if (loss_label == 'mse'):     # MSE (保持像素級一致性)
            loss_value = mse_loss_wafer(x_recon, x, batch_size) # mse 的值域是 [0, +∞), 因為 F.mse_loss 的輸出是非負值.
        elif (loss_label == 'kld'):   # KLD 損失 (保持潛在空間的正則性)
            loss_value = kld_loss_wafer(mu, logvar, batch_size) # kld 的值域是 [0, +∞), 因為 logvar.exp() >= 0, 所以整個式子的值不會是負值.
            # if KL_annealer_weight is not None:
            #     loss_value = KL_annealer_weight * loss_value 
        elif (loss_label == 'dice'):  # Dice Loss (保持形狀一致性, 適合晶圓圖的缺陷區域)
            loss_value = dice_loss_wafer(x_recon, x)
        elif (loss_label == 'ssim'):  # SSIM Loss (保持結構相似性)
            loss_value = ssim_wafer(x_recon, x, window_size=window_size, reduction='mean')  
        # each_loss[loss_label] = loss_value  # 儲存原始的 loss value, 尚未經過 normalization 和 adjustment

        # # 處理正規化. 若 normalization_method == 'none', 則 normalize_loss 會直接回傳原始的 loss_value
        # if isinstance(normalization_method, str):   # 只做一個正規化方法
        #     loss_value = normalize_loss(loss_label, loss_value, each_loss_note, statistic, method=normalization_method)
        # elif isinstance(normalization_method, list):# 做多個正規化方法
        #     for method in normalization_method:
        #         loss_value = normalize_loss(loss_label, loss_value, each_loss_note, statistic, method=method)
        # else:
        #     raise ValueError(f'normalization_method must be str or list. But got {type(normalization_method)}')
        # each_loss[loss_label] = loss_value      # 儲存經過 normalization 後的 loss value

        # 處理 adjustment
        if adjustment != 0:
            # each_loss_note[loss_label] = f'{loss_label} adjustment={adjustment}'
            loss_value = torch.abs(loss_value - adjustment)  # Take absolute value to ensure non-negativity
            
        each_loss[loss_label] = loss_value      # 儲存經過 normalization 和 adjustment 後的 loss value

        # assert_nan_inf_for_loss_component(each_loss, loss_label, x_recon, x)


    # 總損失
    total_loss = 0
    # loss_equation_str = ''
    for loss_label, (weight, adjustment) in loss_weights.items():
        if weight != 0:
            # loss_value_weighted = goal_percent * (weight * each_loss[loss_label])
            loss_value_weighted = weight * each_loss[loss_label]
            total_loss += loss_value_weighted
            # if normalization_method != ['none']: # 如果有提供 normalization_method, 就是經過標準化
            #     normalization_process_str = ''
            #     for method in reversed(normalization_method) if isinstance(normalization_method, list) else [normalization_method]:  # reversed 是因為 loss 計算式是從內到外的
            #         normalization_process_str += f'{method}('
            #     left_parentheses = ')' * len(normalization_method)
            #     loss_equation_str += f'{loss_weights[loss_label][0]:.3f} * {normalization_process_str}(|{loss_label}-{loss_weights[loss_label][1]}|){left_parentheses} + '
            # else: # 沒做標準化
            #     loss_equation_str += f'{loss_weights[loss_label][0]:.3f} * (|{loss_label}-{loss_weights[loss_label][1]}|) + '            
            # loss_equation_str += f'{loss_weights[loss_label][0]:.3f} * (|{loss_label}-{loss_weights[loss_label][1]}|) + '            

    # loss_equation_str = loss_equation_str.rstrip(' + ')
    # if goal_percent != 1: loss_equation_str = f'{goal_percent} * [{loss_equation_str}]'


    # print(f'(loss function) total_loss.item(): {total_loss.item():.9f}')  ##### debug
    # assert_nan_and_inf(total_loss, f'loss_function_colorizeVAE: total_loss')

    return total_loss, '', each_loss, ''
    # return total_loss, loss_equation_str, each_loss, each_loss_note