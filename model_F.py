import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureStitchingModule(nn.Module): 
    """
    拼接 f1, f2, f3, f4, 再轉換成 `(B, out_channels, out_size[0], out_size[1])` 並輸出.

    stitch 後的輸出, 會作為 A 在 forward 的輸入.

    ## `forward(self, features)`
    - `features`: tuple of (f1, f2, f3, f4), where each feature map has shape `(B, R.n_features, H_i, W_i)`. Default R.n_features={64, 128, 256, 512}.
    - return `stitched_features`: Tensor of shape `(B, out_channels, out_size[0], out_size[1])` containing the stitched features.

    :param in_channels: List of input channels for each feature map. Default is `[64, 128, 256, 512]`. 

        Or you can input an integer to specify the minimum number of channels.
        If the input is an integer, it will be processed as `[in_channels * 2**i for i in range(4)]`.

        用於指定每個輸入特徵圖的通道數, 作為"各個改變通道數的卷積層"的輸入.
    :param out_channels: Number of output channels after stitching. Default is 64.
    :param out_size: Size of the output feature map after stitching. Default is (32, 32).
    """

    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=64, out_size=(32, 32)):
        super(FeatureStitchingModule, self).__init__()

        # 先處理 in_channels 的輸入, 確保它是正確的類型和格式.
        assert isinstance(in_channels, int) or isinstance(in_channels, list), \
            f'in_channels must be an int or a list, but got {type(in_channels)}'
        if isinstance(in_channels, list):
            assert len(in_channels) == 4, f'The length of in_channels should be 4, but got {len(in_channels)}'
        else:
            assert in_channels > 0, f'in_channels must be a positive integer, but got {in_channels}'
            in_channels = [in_channels * 2**i for i in range(4)]


        # 以下的四個 process_fi 模塊, 分別處理 R 輸出的不同尺寸的特徵圖, 並將其轉換為相同的輸出通道數和大小.
        # pooling layer 不使用 MaxPool2d, 是因為每個模塊的 input shape 不同, 
        # 而使用 AdaptiveMaxPool2d 可確保輸出大小固定為 out_size.
        
        # input shape: (B, R.n_features//8, 32, 32)
        self.process_f1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=1), # 用於改變通道數
            nn.AdaptiveMaxPool2d(out_size), # 用於改變輸出大小
            nn.ReLU()
        )

        # input shape: (B, R.n_features//4, 16, 16)
        self.process_f2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=1), 
            nn.AdaptiveMaxPool2d(out_size), 
            nn.ReLU()
        )

        # input shape: (B, R.n_features//2, 8, 8)
        self.process_f3 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, kernel_size=1), 
            nn.Upsample(out_size, mode='nearest'),  # 用於改變輸出大小
            nn.ReLU()
        )

        # input shape: (B, R.n_features, 4, 4)
        self.process_f4 = nn.Sequential(
            nn.Conv2d(in_channels[3], out_channels, kernel_size=1), 
            nn.Upsample(out_size, mode='nearest'),
            nn.ReLU()
        )

        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, features): 
        """
        ## `forward(self, features)`
        - `features`: tuple of (f1, f2, f3, f4), where each feature map has shape `(B, R.n_features, H_i, W_i)`. Default R.n_features={64, 128, 256, 512}.
        - return `stitched_features`: Tensor of shape `(B, out_channels, out_size[0], out_size[1])` containing the stitched features.
        """
        # assert len(features) == 4, f'features must be a tuple of 4 elements, but got {len(features)} elements.'
        f1, f2, f3, f4 = features
        f1 = self.process_f1(f1)
        f2 = self.process_f2(f2)
        f3 = self.process_f3(f3)
        f4 = self.process_f4(f4)

        stitched_features = torch.cat((f1, f2, f3, f4), dim=1)  # concatenate along channel dimension
        stitched_features = self.final_conv(stitched_features)
        # assert not torch.isnan(stitched_features).any(), 'stitched_features contains NaN values after final_conv.'

        return stitched_features # shape: (B, out_channels, out_size[0], out_size[1])
    


class A_FeatureAssociation(nn.Module):
    """
    ## `forward(self, gray_prime, ref_prime, ref_wafermap)`
    - gray_prime: output of `stitching_module(gray_features)`. shape=`(B, stitch_out_channels, 16, 16)`
    - ref_prime: output of `stitching_module(ref_features)`. shape=`(B, stitch_out_channels, 16, 16)`
    - ref_wafermap: reference wafermap, shape=`(B, 1, 64, 64)`

    :param stitch_out_channels: 用於指定拼接後的輸出特徵圖的通道數.
    :param qk_channels: 指定 q 和 k 的通道數.
    :param tau: 控制注意力分佈的溫度參數. Default=1.0.
    :param learnable_tau: 指定 tau 是否為可學習的參數. Default=True.
    :param epsilon: 數值穩定性的微小常數. Default=1e-12.
    """
    def __init__(self, stitch_out_channels=64, qk_channels=4, tau=1.0, learnable_tau=True, epsilon=1e-12):
        super(A_FeatureAssociation, self).__init__()
        
        self.stitch_out_channels = stitch_out_channels
        self.qk_channel = qk_channels
        # self.tau = nn.Parameter(torch.tensor(tau).to(device)) if learnable_tau else tau  # 如果 learnable_tau 為 True, 則 tau 是可學習的參數.
        self.tau = nn.Parameter(torch.tensor(tau)) if learnable_tau else tau  # 這裡是上一行移除 .to(device), 因為要使用 scalar 加速運算. 它會自動把參數放到 model 所在的 device.
        self.epsilon = epsilon

        # 用於將 stitching 輸出的特徵圖 gray_prime 和 ref_prime 的通道數轉換為 q 和 k 的通道數.
        self.theta = nn.Conv2d(stitch_out_channels, qk_channels, kernel_size=1)

    def get_A_info(self):
        A_info = {
            # 'stitch_out_channels': self.stitch_out_channels,
            'qk_channel': self.qk_channel,
            # 'tau': self.tau.item() if isinstance(self.tau, torch.Tensor) else self.tau,
            # 'learnable_tau': isinstance(self.tau, torch.Tensor),
            # 'epsilon': self.epsilon,
        }
        return A_info

    def forward(self, gray_prime, ref_prime, ref_image):
        """
        :param gray_prime: output of `stitching_module(gray_features)`. shape=`(B, stitch_out_channels, 32, 32)` 
        :param ref_prime: output of `stitching_module(ref_features)`. shape=`(B, stitch_out_channels, 32, 32)` 
        :param ref_image: reference wafermap, shape=`(B, 1, 64, 64)`
        :return: attenmap: attention map, shape=`(B, 1, 64, 64)`
        :return: wab: weighted average of reference wafermap, shape=`(B, 1, 64, 64)`
        """
        
        assert gray_prime.shape[1:] == ref_prime.shape[1:], \
            f"Expected gray_prime and ref_prime to have the same shape, but got {gray_prime.shape} and {ref_prime.shape}"
            # gray_prime 和 ref_prime 的 shape 都是 (B, stitch_out_channels, H, W)
        # assert not torch.isnan(gray_prime).any(), f'gray_prime contains NaN values. `.abs().max()`: {gray_prime.abs().max()}'
        # assert not torch.isnan(ref_prime).any(), f'ref_prime contains NaN values. `.abs().max()`: {ref_prime.abs().max()}'
        
        B, C, H, W = gray_prime.shape   
        # assert (H, W) == (32, 32), f"Expected gray_prime and ref_prime to have shape (B, {self.stitch_out_channels}, 32, 32), but got {gray_prime.shape}"
        C_qk = self.qk_channel

        # calculate phi_gray and phi_ref
        ## mapping (B, C, H, W) -> (B, C_qk, H, W)
        phi_gray = self.theta(gray_prime)
        phi_ref = self.theta(ref_prime)

        phi_gray = phi_gray - phi_gray.mean(dim=1, keepdim=True)  # normalize across channel dimension
        phi_ref = phi_ref - phi_ref.mean(dim=1, keepdim=True)

        phi_gray = F.normalize(phi_gray, p=2, dim=1, eps=self.epsilon)  # `p=2` means L2 normalization
        phi_ref = F.normalize(phi_ref, p=2, dim=1, eps=self.epsilon)
        # L2 正規化有兩個目的: 
        ## 目的1: 使所有特徵向量的長度都變成 1
        # 這意味著特徵向量的長度不再影響它們的相似性或距離計算. 在影像辨識或特徵學習中, 這很重要. 
        # 例如, 如果一張晶圓圖的特徵向量是因為其亮度較高而具有較大的長度, 我們不希望模型因此就認為它是一個「更重要」的特徵. 
        # L2 正規化確保了模型只關注特徵的內容和模式, 而不是其絕對的幅度. 
        ## 目的2: 穩定餘弦相似性 (Stabilize Cosine Similarity)
        # 在許多深度學習任務中（尤其是聚類或檢索）, 我們經常用餘弦相似性 (Cosine Similarity) 來衡量兩個特徵向量 u 和 v 的相似度：
        # cos(u, v) = (u · v) / ||u||_2 · ||v||_2  ( u·v 是內積, ||u||_2 和 ||v||_2 是向量的 L2 範數(長度).
        # 透過 L2 正規化, 我們將每個向量的長度標準化為 1 (即 ||u||_2 = 1 和 ||v||_2 = 1),
        # 這時候 Cosine Similarity 的計算就變成: Cos(u_norm, v_norm) = u_norm · v_norm
        # 所以內積 u·v 就直接等同於餘弦相似度, 因為分母變成 1.
        # 這樣做的好處是計算更簡單, 並且避免了除以非常小的範數 (這可能導致數值不穩定性). 


        # assert not torch.isnan(phi_gray).any(), f'phi_gray contains NaN values. `.abs().max()`: {phi_gray.abs().max()}'
        # assert not torch.isnan(phi_ref).any(), f'phi_ref contains NaN values. `.abs().max()`: {phi_ref.abs().max()}'

        # calculate f_matrix 
        ## the naming `f_matrix` is from `Feature similarity Matrix`? 
        ## or from `the Matrix that calculate from F_feature_processing`?
        phi_gray_flat = phi_gray.view(B, C_qk, H * W)
        phi_ref_flat = phi_ref.view(B, C_qk, H * W)
        f_matrix = torch.bmm(phi_gray_flat.transpose(1, 2), phi_ref_flat) # output shape=(B, H*W, H*W)
        # ↑ f_matrix 的內容是 "gray 的每個 pixel 與 ref 的每個 pixel 之間的相似度"
        # 如果將計算方式改為 `torch.bmm(phi_gray_flat, phi_ref_flat.transpose(1, 2))`,
        # 則 output shape=(C_qk, C_qk), 代表的意義是 "phi_gray 通道與 phi_ref 通道之間的相似程度", 就不是每個 pixel 之間的相似度了.

        # assert not torch.isnan(f_matrix).any(), f'f_matrix contains NaN values. `.abs().max()`: {f_matrix.abs().max()}'

        # calculate wab 
        ## wab is meant weighted color. The "ab" in wab is the "ab" in lab color space.
        ## wab 提供了從 ref_image 提取的、經過加權處理的顏色資訊, 用於指導 gray_image 的著色方式
        ## wab 相比 attenmap, wab 較側重於顏色資訊本身, 即從 ref_image 中提取並加權後用於色彩化的具體顏色值
        # 這邊是在計算一個加權平均特徵 (Weighted Average Feature), 也就是在執行一個注意力機制 (Attention Mechanism).
        # 利用先前計算的相似度矩陣 f_matrix 來決定 ref_image 中的哪些像素應該對最終輸出做出貢獻, 進而產生一個新的、經過注意力調整的特徵圖 wab.
        attention_weights = F.softmax(f_matrix / self.tau, dim=2)   # shape=(B, H*W, H*W)  # dim=2 是對參考像素維度進行 softmax. 
        ref_image = F.interpolate(ref_image, size=(H, W), mode='nearest')
        # assert ref_image.shape == (B, 1, H, W), f"Expected ref_image to have shape (B, 1, {H}, {W}), but got {ref_image.shape}"
        ref_wafermap_flat = ref_image.view(B, 1, H * W)
        # 根據 attention_weights 對 ref_image 的顏色 (ref_flat) 進行加權求和
        # attention_weights: (B, H*W, H*W)
        # ref_wafermap_flat: (B, 1, H*W). 需要轉置為 (B, H*W_ref, 1)
        wab_flat = torch.bmm(attention_weights, ref_wafermap_flat.transpose(1, 2))  # shape=(B, H*W, 1)
        wab = wab_flat.view(B, 1, H, W)

        # calculate attenmap
        # attenmap 提供關於 gray 和 ref 之間, 語義對應和特徵相似性的資訊
        # attenmap 相比 wab, 更側重於特徵層面的對應關係和相似性, 它表示模型在圖像不同區域之間分配的注意力或重要性, 以此來指導色彩的傳播和對齊
        attenmap_flat, _ = torch.max(attention_weights, dim=2)  # shape=(B, H*W)
        attenmap = attenmap_flat.view(B, 1, H, W)
        # ↑這樣的寫法會得到「每個位置最大注意力的值」, 讓某個位置只受單一最重要參考點影響
        # attention_weights.shape: (B, P, P), P=H*W. 矩陣中每一列的 i 代表 image_gray 的第 i 個像素, 對應到 image_ref 的 P 個像素的權重分佈. 
        # dim=2 代表的是, 對於 image_gray 中的每一個像素 i（對應矩陣的一行）, 它會從 P 個權重中找出最大值. 
        # attenmap 是單通道的, 稱之為注意力熱圖. 這個熱圖是模型可解釋性 (Explainability) 的工具. 它顯示了輸入圖片的哪些部分是模型在執行注意力機制時, 認為最重要或最獨特的. #####
        # 這樣的寫法能確保對每個目標位置的注意力權重總和被保留, 得到的值是「對所有參考位置的整體注意力量」

        # assert not torch.isnan(wab).any(), f'wab contains NaN values. `.abs().max()`: {wab.abs().max()}'
        # assert not torch.isnan(attenmap).any(), f'attenmap contains NaN values. `.abs().max()`: {attenmap.abs().max()}'

        return wab, attenmap # both shape in (B, 1, H, W)

        ### 筆記 ###    
        # f_matrix: 輸入圖上的 i 點與參考圖上的 j 點的特徵有多相似. 
        # attention_weights: 將這個相似度轉化為機率權重. 
        #
        # attenmap: 
        # 一個單通道的注意力熱圖, 代表輸入圖上每個像素的重要性或獨特性.
        # range = [0, 1], 其中 1 代表該位置對應到參考圖的某個位置有非常高的注意力權重,
        #
        # wab: 
        # 代表從參考圖中提取的顏色資訊.
        # 一個新的特徵圖, 其每個像素的值都是參考圖所有像素的加權平均, 權重由輸入圖的特徵決定. 
        # 可以被視為從參考圖中提取的、針對當前輸入圖的「精煉資訊」, 
        # 通常會被加到模型的解碼器 (Decoder) 中, 以幫助更精確地重建晶圓圖. 
        # range = {0, 0.5, 1}, 就是原始影像的像素值.



class F_feature_processing(nn.Module):
    """
    F contains the R_module and A_module.
    """

    # def __init__(self, R_gray, R_ref, stitch_out_channels=64, stitch_out_size=(32, 32), qk_channel=4):
    def __init__(self, R_ref, stitch_out_channels=64, stitch_out_size=(32, 32), qk_channel=4):
        
        super(F_feature_processing, self).__init__()
        self.name = 'F'
        self.stitch_out_channels = stitch_out_channels
        self.stitch_out_size = stitch_out_size
        self.qk_channel = qk_channel
        # self.R_gray = R_gray
        self.R_ref = R_ref
        self.stitch_in_channels = [R_ref.output_channels // 8, R_ref.output_channels // 4, R_ref.output_channels // 2, R_ref.output_channels]
        self.stitcher = FeatureStitchingModule(in_channels=self.stitch_in_channels, out_channels=self.stitch_out_channels, out_size=self.stitch_out_size)#.to(device)
        self.A = A_FeatureAssociation(stitch_out_channels=self.stitch_out_channels, qk_channels=self.qk_channel, learnable_tau=False)#.to(device)

    def get_R_info(self):
        R_info = {
            # 'R_gray': (self.R_gray.name), 
            'R_ref': f'name={self.R_ref.name}, output_channels={self.R_ref.output_channels}',
        }
        return R_info
    
    def get_F_info(self):
        F_info = {
            'stitch_out_channels': self.stitch_out_channels,
            'stitch_out_size': self.stitch_out_size,
            'qk_channel': self.qk_channel,
            # 'R_info': self.get_R_info(),
            # 'A_info': self.A.get_A_info(),
        }
        return F_info

    def forward(self, gray_image, ref_image):   # shape=(B, 1, 64, 64)
        # gray_image, ref_image = gray_image.to(device), ref_image.to(device)
        # _, gray_features = self.R_gray(gray_image)  # feature maps are in tuple (f1, f2, f3, f4), which are from layer 1~4. 
        _, gray_features = self.R_ref(gray_image)  # feature maps are in tuple (f1, f2, f3, f4), which are from layer 1~4. 
        _, ref_features = self.R_ref(ref_image) # feature maps are in tuple (f1, f2, f3, f4), which are from layer 1~4. 
        
        gray_prime = self.stitcher(gray_features)   # shape=(B, self.stitch_out_channels, (out_size))
        ref_prime = self.stitcher(ref_features) # shape=(B, self.stitch_out_channels, (out_size))

        wab, attenmap = self.A(gray_prime, ref_prime, ref_image)

        # print(f'\nclass F forward: \nwab.abs().max(): {wab.abs().max()}, wab.abs().min(): {wab.abs().min()}') # 這裡的 max/min 會忽略 NaN, 所以 print 出來的值不會是 NaN. 但是可以藉此觀察 Wab/Attenmap 是否過大.
        # print(f'attenmap.abs().max(): {attenmap.abs().max()}, attenmap.abs().min(): {attenmap.abs().min()}')
        # assert not torch.isnan(wab).any(), f'wab contains NaN values. `.abs().max()`: {wab.abs().max()}'
        # assert not torch.isnan(attenmap).any(), f'attenmap contains NaN values. `.abs().max()`: {attenmap.abs().max()}'

        return wab, attenmap
        
