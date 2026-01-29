import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # 用於 GPU 加速運算 # 舊版本是 from torch.cuda.amp import GradScaler, autocast
from torchvision.models.vision_transformer import VisionTransformer

from sklearn.metrics import confusion_matrix
import time

from utils import device, Logger, label_to_code, labels_in_code, plot_confusion_matrix


def train_R(model, train_loader, criterion, optimizer, scheduler, epochs, 
            scaler: GradScaler, 
            logger: Logger):
    """
    :param model: The model to be trained.
    :param train_loader: The DataLoader for training data.
    :param criterion: The loss function.
    :param optimizer: The optimizer.
    :param scheduler: The learning rate scheduler (optional).
    :param epochs: The number of training epochs.

    :return log_loss: list of average loss per epoch
    :return log_lr: list of learning rate per epoch
    :return timestamps: tuple of (start_time, duration, end_time)

    ~~:return dataloader_name: name of the dataloader used for training, if available.~~
    """
    model.train()
    log_loss, log_lr = [], []
    time_start = time.time()

    for epoch in range(epochs):
        running_loss = 0
        for wafermap, label in train_loader:
            optimizer.zero_grad()
            wafermap = wafermap.to(device)
            label = label.float().to(device)  # BCEWithLogitsLoss expects float labels
            
            with autocast(device_type=device.type):  # 用於 GPU 加速運算 (混合精度訓練)
                predict, _ = model(wafermap)
                # predict = torch.sigmoid(predict) # 這行不需要, 因為 criterion 是 BCEWithLogitsLoss, 它會「內部」對 input 進行 Sigmoid 轉換.
                # ↑ 只有在 inference 時才需要加上 Sigmoid, 因為 inference 時不會使用 criterion, 所以需要手動將 logits 轉換為 probabilities.
                loss = criterion(predict, label)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()
            
            running_loss += loss.item()

        if scheduler: scheduler.step(running_loss)
        loss_avg = running_loss / len(train_loader)
        log_loss.append(loss_avg)
        log_lr.append(optimizer.param_groups[0]['lr'])
        if (epoch + 1) % 10 == 0: 
            message = f'Epoch [{epoch+1:3d}/{epochs:3d}] loss: {loss_avg:.9f}, lr: {optimizer.param_groups[0]["lr"]:.9f}'
            logger.log(message, is_print=True)
            
    time_end = time.time()
    time_duration = time_end - time_start
    timestamps = (time_start, time_duration, time_end)

    # dataloader_name = train_loader.name if hasattr(train_loader, 'name') else None
    # dataloader_config = train_loader.config if hasattr(train_loader, 'config') else None
    # dataloader_info = f'name={dataloader_name}, config={dataloader_config}'
    # return log_loss, log_lr, timestamps, dataloader_info

    model.config['train_loader'] = train_loader.config
    # dataloader_config = f'{train_loader.config}'
    # return log_loss, log_lr, timestamps, dataloader_config
    return log_loss, log_lr, timestamps




@torch.no_grad()
def validate_R(model, validation_loader, defect_types):
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

        predict, _ = model(wafermap)    
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
    timestamps = (time_start, time_duration, time_end)

    cm = confusion_matrix(all_labels, all_preds, labels=labels_in_code)    
    # 把正確的資料筆數從數量改成百分比
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    # cm = cm * 100  # 如果要顯示百分比, 可以將這一行取消註解, 但這樣會使得 confusion matrix 的值變成百分比, 而不是小數.

    assert validation_loader.name, '`validation_loader.name` must be set. Please set the `name` of the `validation_loader` before calling `inference_Rref`.'
    
    if hasattr(model, 'config'):
        model.config['validation_loader'] = validation_loader.config
        model_config_str = '\n'.join([f'{k}={v}' for k, v in model.config.items()])
        title = f'Confusion Matrix of {model.name}\n--------------------\n{model_config_str}'
    elif hasattr(model, 'model_config'):
        model.model_config += f'\nvalidation_loader = {validation_loader.config["name"]}'
        title = f'Confusion Matrix of {model.name}\n--------------------\n{model.model_config}'

    # if isinstance(model_config, dict):
    #     model_config['dataloader(inference)'] = validation_loader.name
    #     model_config_str = '\n'.join([f'{k}={v}' for k, v in model_config.items()])
    #     title = f'Confusion Matrix of {model.name}\n--------------------\n{model_config_str}'
    # elif isinstance(model_config, str):
    #     model_config += f'\ndataloader(inference) = {validation_loader.name}'
    #     title = f'Confusion Matrix of {model.name}\n--------------------\n{model_config}'

    fig = plot_confusion_matrix(cm, defect_types, title=title, times=timestamps)
    
    return fig, timestamps, cm



##### ----- R1 ----- #####
class R_Vanilla(nn.Module):
    """
    `forward(self, x)`
    - x: shape=(B, 1, 64, 64), B=batch
    """
    def __init__(self, model_name=None, n_features=64, dilation=1, groups=1):
        super(R_Vanilla, self).__init__()
        self.name = model_name if model_name else type(self).__name__
        self.n_features = n_features
        self.model_config = f'n_features={n_features}, dilation={dilation}, groups={groups}'

        def conv_block(in_channels, out_channels, groups=1):
            return nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation, groups=groups),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU()
            )
        
        # input: resized wafermap scale = (64, 64)
        assert groups <= n_features // 8, f'groups must be less than or equal to {n_features // 8}(`n_features // 8`)'
        self.conv1 = conv_block(1, n_features // 8, groups=1) # 64 -> 32
        self.conv2 = conv_block(n_features // 8, n_features // 4, groups=groups)   # 32 -> 16
        self.conv3 = conv_block(n_features // 4, n_features // 2, groups=groups)   # 16 -> 8
        self.conv4 = conv_block(n_features // 2, n_features, groups=groups)    # 8 -> 4

        self.conv_out_size = n_features * 4**2

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, n_features),
            nn.ReLU(), 
            nn.Linear(n_features, 8)    # 最後不接 activation function, 因為 criterion 是 BCEWithLogitsLoss, 它內部會對 input 做 Sigmoid 轉換.
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        predict = f4.view(f4.size(0), -1)
        predict = self.fc(predict)  # shape=(B, 8) 
        return predict, (f1, f2, f3, f4)
        # 訓練時, 拿 predict 來用.
        # 推論時, 拿 f1, f2, f3, f4 來輸出 attenmap, wab.


##### ----- R2 ----- #####
# from torchvision.models import vit_b_16
class R_ViT(VisionTransformer):
    """
    A custom Vision Transformer model for wafermap classification.
    This model uses a convolutional layer to project the input wafermap into patches,
    and then applies the Vision Transformer architecture for classification.

    :param model_name: Name of the model (optional).
    :param image_size: Size of the input image (height and width).
    :param patch_size: Size of each patch.
    :param num_layers: Number of transformer layers.
    :param num_heads: Number of attention heads.
    :param hidden_dim: Dimension of the hidden layers.
    :param mlp_dim: Dimension of the MLP layers.
    :param num_classes: Number of output classes.
    :param dropout: Dropout rate for the transformer layers.
    :param attention_dropout: Dropout rate for the attention layers.
    :param representation_size: Size of the representation layer (if used).

    """
    def __init__(self, model_name=None, 
                 image_size=64, patch_size=4, 
                 num_layers=4, num_heads=4, hidden_dim=128, mlp_dim=512, 
                 dropout=0, attention_dropout=0, num_classes=8, 
                 representation_size=None
                 ):
        super(R_ViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes, 
            representation_size=representation_size    
            # ↑ 預設為 None, 代表不使用 heads 層, 即 `heads_layers["head"] = nn.Linear(hidden_dim, num_classes)`. 
            # 若非 None, 則會比 None 多一層 linear 層.
        )

        self.name = model_name if model_name else type(self).__name__
        self.pre_conv = nn.Conv2d(1, 3, kernel_size=1)  # 將單通道的 wafermap 轉換為三通道, 以符合 ViT 的輸入要求.
        self.heads = nn.Linear(hidden_dim, num_classes)

        self.model_config = f'image_size={image_size}, patch_size={patch_size}, \
num_layers={num_layers}, num_heads={num_heads}, \
hidden_dim={hidden_dim}, mlp_dim={mlp_dim}, \n\
dropout_rate={dropout}, attention_dropout={attention_dropout}, \
num_classes={num_classes}, representation_size={representation_size}'

    def forward(self, x):
        x = self.pre_conv(x)  # (B, 3, H, W), 將單通道的 wafermap 轉換為三通道, 以符合 ViT 的輸入要求.

        # Reshape and permute the input tensor
        x = self._process_input(x)  # (B, C, H, W) -> (B, patch_size ** 2, hidden_dim)
        # ↑ 在 _process_input 的原始碼中, 會改變 shape的程式碼有
        # nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        # x = x.reshape(n, self.hidden_dim, n_h * n_w)
        # x = x.permute(0, 2, 1)
        n = x.shape[0]  # 這邊的 n 就是 batch size (B). 以 n 命名是因為在 ViT 的原始碼中, 會將 batch size 命名為 n.

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)    # shape=(B, 1 + patch_size ** 2, hidden_dim)
        
        # x = self.encoder(x) # 這行改成以下程式碼, 因為要蒐集所有 encoder layer 的輸出
        # assert x.shape[1:] == (self.seq_length, self.hidden_dim), f'輸出 x 的 shape 應該是 (B, 1 + N_patches({self.seq_length - 1}), hidden_dim({self.hidden_dim})), 但實際上是 {x.shape}. '
        features = []
        for blk in self.encoder.layers: # blk 是 block 的縮寫
            x = blk(x) 
            features.append(x.clone().detach().cpu())  # 若需要儲存 gradient, 則不可加上 `detach().cpu()`, 但這樣會佔用大量記憶體.

        # # 最後一層 LayerNorm  (不用做, 因為在 ViT 的原始碼中, 已經在每個 encoder layer 的 forward 中做了 LayerNorm)
        # x = self.encoder.ln(x)    

        # 取出 class token
        x = x[:, 0, :]

        logits = self.heads(x)

        return logits, features  # features 是一個 list, 長度 = num_layers



##### ----- R3 ----- #####
# ↑ 這個命名並不精確. ~~這是年輕時犯下的過錯~~ 
# 應該要命名為 `VGG4blocks` 比較合理. 因為這個 class 用了 4 個 blocks, 而不是 4 層.
def vgg_block(in_channels, out_channels):
    """
    `nn.Sequential(Conv2d, BatchNorm2d, ReLU, MaxPool2d)`
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2), 
    )

class MyVGG_4layer(nn.Module):
# ↑ 這個命名並不精確. ~~這是年輕時犯下的過錯~~ 
# 應該要命名為 `VGG4blocks` 比較合理. 因為這個 class 用了 4 個 blocks, 而不是 4 層.
    """
    Custom VGG model with 4 layers.

    :param model_name: Name of the model (optional).
    :param num_classes: Number of output classes (default=8).
    """
    def __init__(self, model_name=None,
                 output_channels=512, 
                 num_classes=8):
        super(MyVGG_4layer, self).__init__()
        
        self.name = model_name if model_name else type(self).__name__
        self.output_channels = output_channels
        self.config = {
            'name': self.name,
            'output_channels': output_channels, 
            'num_classes': num_classes
            }
        # self.model_config = f'output_channels={output_channels}'
        self.model_config = ''  # 舊版的 model_config 已經被棄用, 改用 self.config 字典來儲存模型設定. 考量到舊版程式碼中有使用 model_config 的地方, 為了避免出錯, 這裡先設為空字串.

        assert output_channels % 8 == 0, f'output_channels must be a multiple of 8, but got {output_channels}.'
        self.conv_layers = nn.ModuleList([
            vgg_block(1, output_channels // 8),  # (1, 64, 64) -> (64, 32, 32)
            vgg_block(output_channels // 8, output_channels // 4),  # (64, 32, 32) -> (128, 16, 16)
            vgg_block(output_channels // 4, output_channels // 2),  # (128, 16, 16) -> (256, 8, 8)
            vgg_block(output_channels // 2, output_channels),  # (256, 8, 8) -> (512, 4, 4)
        ])
            
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 將 (B, 512, 4, 4) 展平為 (B, 512 * 4 * 4)
            # nn.Linear(512 * 4 * 4, 1024),  # 全連接層
            nn.Linear(output_channels * 4 * 4, 1024),  # 全連接層
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout 層
            nn.Linear(1024, num_classes)  # 最後的分類層
        )

    def forward(self, x):
        # x = x.to(device)
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            # features.append(x.clone().detach().cpu())
            features.append(x.clone().detach()) # detach 是為了避免佔用過多記憶體, 雖然這樣無法計算 gradient, 但我沒有要算 gradient, 只是想要蒐集 feature map.

        predict = self.classifier(x)
        return predict, features  # `predict` is the output of the classifier, `features` is a list that contains feature maps from each layer.



##### ----- R4 ----- #####
"""
此版本只是`R3_VGG 4 blocks` 的 3 個 blocks 版本加上 1 個 maxpooling.

Q: 在第一個、第二個或第三個 block 後面加上 pooling, 有什麼差別?
* 第一個 block 後加 pooling: 優點是加速計算, 但可能會較早丟失細節特徵, 適合對低階特徵有較高需求的任務. 
* 第二個 block 後加 pooling: 比較平衡, 保留了更多的特徵資訊, 同時適度壓縮空間維度, 適合大部分任務. 
* 第三個 block 後加 pooling: 會保留更多空間細節, 適合需要精細辨識或保留細節特徵的任務, 但計算量較大, 可能不適用於需要快速推理的情境. 
"""
class MyVGG_3blocks(nn.Module):
    """
    Custom VGG model with 3 blocks.

    :param model_name: Name of the model (optional).
    :param n_feature_maps_final: Number of feature maps in the final convolutional layer (default=512).
    :param num_classes: Number of output classes (default=8).
    """
    def __init__(self, model_name=None, 
                 output_channels=512, 
                 num_classes=8):
        super(MyVGG_3blocks, self).__init__()
        
        self.name = model_name if model_name else type(self).__name__
        self.output_channels = output_channels
        self.config = {'output_channels': output_channels}
        self.model_config = ''  # 舊版的 model_config 已經被棄用, 改用 self.config 字典來儲存模型設定. 考量到舊版程式碼中有使用 model_config 的地方, 為了避免出錯, 這裡先設為空字串.

        assert output_channels % 8 == 0, f'output_channels must be a multiple of 8, but got {output_channels}.'
        self.conv_layers = nn.ModuleList([
            vgg_block(1, output_channels // 8),  # (1, 64, 64) -> (64, 32, 32)
            vgg_block(output_channels // 8, output_channels // 4),  # (64, 32, 32) -> (128, 16, 16)
            vgg_block(output_channels // 4, output_channels),  # (256, 16, 16) -> (512, 8, 8)
            nn.MaxPool2d(kernel_size=2, stride=2)   # (512, 8, 8) -> (512, 4, 4)
        ])
            
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 將 (B, 512, 4, 4) 展平為 (B, 512 * 4 * 4)
            nn.Linear(512 * 4 * 4, 1024),  # 全連接層
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout 層
            nn.Linear(1024, num_classes)  # 最後的分類層
        )

    def forward(self, x):
        x = x.to(device)
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            features.append(x.clone().detach().cpu())
        # assert x.shape == (x.shape[0], 512, 4, 4), f'x.shape should be (B, 512, 4, 4), but {x.shape}'

        predict = self.classifier(x)
        return predict, features  # `predict` is the output of the classifier, `features` is a list that contains feature maps from each layer.



##### ----- R5 ----- #####
# # 可以使用 torchvision.ops.DeformConv2d 或 mmcv.ops.DeformConv2d, 例如:

# from torchvision.ops import DeformConv2d

# deform_conv = DeformConv2d(inc=64, outc=128, kernel_size=3, padding=1)
# x = torch.randn(1, 64, 128, 128)
# offset = torch.randn(1, 18, 128, 128)  # 3x3 kernel -> 18 channels
# out = deform_conv(x, offset)
# # 若要加 modulation (DCNv2), 則要額外傳 modulation mask。

