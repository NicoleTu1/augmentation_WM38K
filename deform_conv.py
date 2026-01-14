# reference: Deformable_Convolutional_Networks_for_Efficient_Mixed-Type_Wafer_Defect_Pattern_Recognition

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torchvision.ops import DeformConv2d

import time
from sklearn.metrics import confusion_matrix

from utils import Logger, label_to_code, labels_in_code, plot_confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 由於 torchvision.ops.DeformConv2d 需要兩個輸入 (input_features 和 offset_map), 
# 而標準的 nn.Sequential 只能接受單一輸入並將其傳遞給下一個層, 因此不能直接在 nn.Sequential 中使用 DeformConv2d
# 解決方案：使用 nn.Module 自訂 Block 的 class

class DeformableConv(nn.Module):
    """
    將 DeformConv2d 封裝成一個接受單一輸入張量的模塊, 
    使其可以在 nn.Sequential 或類似的串聯結構中使用. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=True):
        super().__init__()

        # 1. 偏移量學習層 (標準 Conv2D)
        # 它的輸出就是 DCN 所需的 offsets 張量
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size ,  # 每個取樣點有 x 和 y 兩個偏移量
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding,
            groups=groups, 
            bias=bias,  # 偏移量卷積通常使用 bias
        )
        # 權重初始化 (可選, 但常見於 DCN 實作)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # 2. 變形與卷積層 (Deformable Convolution Layer)
        # DCN 會接收輸入 (x) 和偏移量 (offset)
        # ConvOffset2D_train 只變形不改變通道數, 所以一般會將 out_channels 設為 in_channels.
        # 但這邊會將 ConvOffset2D_train 直接接上標準 Conv2D, 所以 out_channels 可以設為最終需要的通道數.
        self.dcn = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            bias=bias,
        )

    def forward(self, x):
        offset_map = self.offset_conv(x)    # 學習偏移量. shape: [B, 2*K*K, H, W]
        output = self.dcn(x, offset_map)    # 變形取樣並進行卷積
        
        return output



def dc_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, name=None):
    # 在 trian_mutil_label.txt 中, block 包含: 
    # * 偏移卷積層 (ConvOffset2D_train): 應用於輸入特徵圖. 
    # * 標準 2D 卷積層 (Conv2D): 使用 (3, 3) 核心尺寸. 
    # * 批次正規化層 (BatchNormalization). 
    # * ReLU 啟用函數 (Activation('relu')). 
    return nn.Sequential(
        DeformableConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),   # (ConvOffset2D_train + Conv2D)
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )



class Classifier_ValidateAugmentation(nn.Module):
    def __init__(self, name='cls', in_channels=1, num_classes=8, conv_num=32):
        super(Classifier_ValidateAugmentation, self).__init__()
        self.name = name
        self.model_config = f'in_channels={in_channels}, num_classes={num_classes}, conv_num={conv_num}'

        # Keras: conv_num = 32, fullnet_num = 128
        C1 = conv_num          # 32
        C2 = conv_num * 2      # 64
        C3 = conv_num * 4      # 128
        C4 = conv_num * 8      # 256
        C5 = conv_num * 4      # 128 (注意: Conv5 的輸出通道數減少了)
        
        # ------------------- 卷積 -------------------
        # Keras 實作的輸入形狀為 (B, H, W, 1). PyTorch 為 (B, 1, H, W). 
        self.dc_conv = nn.Sequential(
            dc_block(in_channels, C1, stride=2, name='Conv_1'),             # Conv 1 Layer: Input -> DC(filters=1) -> Conv(32, stride=2) -> BN -> ReLU
            dc_block(C1, C2, stride=2, name='Conv_2'),                      # Conv 2 Layer: DC(filters=32) -> Conv(64, stride=2) -> BN -> ReLU
            dc_block(C2, C3, stride=2, name='Conv_3'),                      # Conv 3 Layer: DC(filters=64) -> Conv(128, stride=2) -> BN -> ReLU
            dc_block(C3, C4, stride=1, name='Conv_4'), # 這一層 stride=1    # Conv 4 Layer: DC(filters=128) -> Conv(256, stride=1) -> BN -> ReLU
            dc_block(C4, C5, stride=2, name='Conv_5'),                      # Conv 5 Layer: DC(filters=256) -> Conv(128, stride=2) -> BN -> ReLU
            nn.AdaptiveAvgPool2d((1, 1)),    # tensorflow.keras 的 GlobalAveragePooling2D
        )

        # ------------------- 分類 -------------------        
        self.fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(C5, num_classes), # C5=128 (Conv5 輸出通道數)
            # nn.Sigmoid()              # Sigmoid 輸出 (多標籤分類) # 因為搭配
        )

    def forward(self, x):
        # x_shape 範例: (B, 1, 52, 52) (針對 MixedWM38 資料集, 通常會統一尺寸, 如 52x52)
        x = x.to(device)
        x = self.dc_conv(x)
        outputs = self.fc(x)
        
        return outputs



def train_cls(model, train_loader, criterion, optimizer, scheduler=None, epochs=50, 
              scaler: GradScaler=None, 
              logger: Logger=None):
    model.train()
    log_loss, log_lr = [], []
    time_start = time.time()

    for epoch in range(epochs):
        running_loss = 0
        for wafermap, label in train_loader:
            label = label.float().to(device)
            wafermap = wafermap.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type):

                predict= model(wafermap)
                # predict = torch.sigmoid(predict) # 這行不需要, 因為 criterion 是 BCEWithLogitsLoss, 它會「內部」對 input 進行 Sigmoid 轉換.
                # ↑ 只有在 inference 時才需要加上 Sigmoid, 因為 inference 時不會使用 criterion, 所以需要手動將 logits 轉換為 probabilities.
                
                # assert predict.device == label.device, f'predict.device={predict.device}, label.device={label.device}' ##### debug
                loss = criterion(predict, label)
            
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) ####
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach().item()

        if scheduler: scheduler.step(running_loss)
        loss_avg = running_loss / len(train_loader)
        log_loss.append(loss_avg)
        log_lr.append(optimizer.param_groups[0]['lr'])
        if (epoch + 1) % 10 == 0: 
            message = f'Epoch [{epoch+1:3d}/{epochs:3d}] loss: {loss_avg:.14f}, lr: {optimizer.param_groups[0]["lr"]:.9f}'
            print(message)
            logger.log(message)
            
    time_end = time.time()
    time_duration = time_end - time_start
    times = (time_start, time_duration, time_end)

    dataloader_name = train_loader.name if hasattr(train_loader, 'name') else None

    return log_loss, log_lr, times, dataloader_name



@torch.no_grad()
def validate_cls(model, validation_loader, defect_types, model_config: str):
    """
    :param model: the trained model
    :param validation_loader: DataLoader for the validation dataset
    :param defect_types: list of defect types for labeling
    :return confusion matrix figure:
    :return times: tuple of (start_time, duration, end_time)
    :return confusion matrix data: type `np.ndarray`, shape=(num_classes, num_classes)
    :return dataloader_name: name of the dataloader used for validation, if available.
    """
    model.eval()
    all_preds = []
    all_labels = []
    time_start = time.time()

    for wafermap, label in validation_loader:
        wafermap = wafermap.to(device)
        # label = label.to(device)
        label_ = [label_to_code[tuple(x)] for x in label.cpu().numpy()] # 把 ground truth label 轉換成代碼, 用於計算 confusion matrix

        predict = model(wafermap)    
        predict = torch.sigmoid(predict)  # R 的預測值 (predict) 沒有經過 activation function, 所以這邊才會做 torch.sigmoid. 
        predict = predict.cpu().numpy() > 0.5   # Convert logits to binary predictions
        # ↑ predict 是一個 numpy array, 元素型別為 bool, shape 與 predict 相同（通常是 batch_size x num_classes）.
        # 例如：如果 batch_size=64, num_classes=8, 則 predict.shape=(64, 8), 每個元素都是 True/False.
        predict_ = [label_to_code[tuple(x)] if tuple(x) in label_to_code else 'XXX' for x in predict] 
        # ↑ 把預測 label 轉換成代碼, 用於計算 confusion matrix.
        # 若預測的 label 不在 label_to_index 中, 則設為 'XXX' (未出現在 ground truth 中的預測組合)
        
        all_preds.append(predict_)
        all_labels.append(label_)

    all_labels = sum(all_labels, [])
    all_preds = sum(all_preds, [])

    time_end = time.time()
    time_duration = time_end - time_start
    times = (time_start, time_duration, time_end)

    cm = confusion_matrix(all_labels, all_preds, labels=labels_in_code)    
    # 把正確的資料筆數從數量改成百分比
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    # cm = cm * 100  # 如果要顯示百分比, 可以將這一行取消註解, 但這樣會使得 confusion matrix 的值變成百分比, 而不是小數.

    # 設定標題
    assert validation_loader.name, '`validation_loader.name` must be set. Please set the `name` of the `validation_loader` before calling `inference_Rref`.'
    model_config += f'\ndataloader(inference) = {validation_loader.name}'

    # macro_avg_acc = np.diag(cm[:38]).mean() # 38 是因為 cm 裡面有一個多出來的類別 'XXX', 所以要取前 38 個類別的對角線值來計算 macro average accuracy.
    # model_config += f'\nmacro average accuracy = {macro_avg_acc:.4f}'
    
    title = f'Confusion Matrix of {model.name}\n--------------------\n{model_config}'

    fig = plot_confusion_matrix(cm, defect_types, title=title, times=times)
    
    return fig, times, cm, model_config




# class cls_test(nn.Module):
#     def __init__(self, name='cls', in_channels=1, num_classes=8, conv_num=128):
#         super(cls_test, self).__init__()
#         self.name = name
#         self.model_config = f'in_channels={in_channels}, num_classes={num_classes}, conv_num={conv_num}'

#         # Keras: conv_num = 32, fullnet_num = 128
#         C1 = conv_num          # 32
#         C2 = conv_num * 2      # 64
#         C3 = conv_num * 4      # 128
#         C4 = conv_num * 8      # 256
#         C5 = conv_num * 4      # 128 (注意: Conv5 的輸出通道數減少了)
        
#         # ------------------- 卷積 -------------------
#         # Keras 實作的輸入形狀為 (B, H, W, 1). PyTorch 為 (B, 1, H, W). 
#         self.dc_conv = nn.Sequential(
#             dc_block(in_channels, C1, stride=2, name='Conv_1'),             # Conv 1 Layer: Input -> DC(filters=1) -> Conv(32, stride=2) -> BN -> ReLU
#             dc_block(C1, C2, stride=2, name='Conv_2'),                      # Conv 2 Layer: DC(filters=32) -> Conv(64, stride=2) -> BN -> ReLU
#             dc_block(C2, C3, stride=2, name='Conv_3'),                      # Conv 3 Layer: DC(filters=64) -> Conv(128, stride=2) -> BN -> ReLU
#             dc_block(C3, C4, stride=1, name='Conv_4'), # 這一層 stride=1    # Conv 4 Layer: DC(filters=128) -> Conv(256, stride=1) -> BN -> ReLU
#             dc_block(C4, C5, stride=2, name='Conv_5'),                      # Conv 5 Layer: DC(filters=256) -> Conv(128, stride=2) -> BN -> ReLU
#             nn.AdaptiveAvgPool2d((1, 1)),    # tensorflow.keras 的 GlobalAveragePooling2D
#         )

#         # ------------------- 分類 -------------------        
#         self.fc = nn.Sequential(
#             nn.Flatten(), 
#             nn.Linear(C5, num_classes), # C5=128 (Conv5 輸出通道數)
#             # nn.Sigmoid()              # Sigmoid 輸出 (多標籤分類) # 因為搭配
#         )

#     def forward(self, x):
#         # x_shape 範例: (B, 1, 52, 52) (針對 MixedWM38 資料集, 通常會統一尺寸, 如 52x52)
#         x = x.to(device)
#         x = self.dc_conv(x)
#         outputs = self.fc(x)
        
#         return outputs

