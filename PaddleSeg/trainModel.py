import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader, DistributedBatchSampler
from paddleseg.models import UNetPlusPlus
from paddleseg.models.losses import BCELoss, MixedLoss, LovaszSoftmaxLoss
from paddleseg.core import train
from paddleseg.transforms import Compose, Resize

# 初始化多卡环境-----双卡运行---------------

paddle.distributed.init_parallel_env()

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, dataset_path, mode, transforms=[], num_classes=2, ignore_index=255):
        list_path = os.path.join(dataset_path, (mode + '_list.txt'))
        self.data_list = self.__get_list(list_path)
        self.mode = mode
        self.data_num = len(self.data_list)
        self.transforms = Compose(transforms, to_rgb=False)
        self.is_aug = False if len(transforms) == 0 else True
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
    def __getitem__(self, index):
        A_path, B_path, lab_path = self.data_list[index]
        A_img = cv2.cvtColor(cv2.imread(A_path), cv2.COLOR_BGR2RGB)
        B_img = cv2.cvtColor(cv2.imread(B_path), cv2.COLOR_BGR2RGB)
        image = np.concatenate((A_img, B_img), axis=-1)
        label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        
        if self.is_aug:
            data = self.transforms({'img': image, 'label': label})
            image, label = data['img'], data['label']
        else:
            image = image.transpose(2, 0, 1)

        image = image.astype('float32') / 255.0
        label = label.astype('int64')

        if self.mode != 'test':
            label = label.clip(max=1)
            label = label[np.newaxis, :]
        return {'img': image, 'label': label, 'trans_info': []}
    
    def __len__(self):
        return self.data_num
    
    def __get_list(self, list_path):
        data_list = []
        with open(list_path, 'r') as f:
            data = f.readlines()
            for d in data:
                data_list.append(d.replace('\n', '').split(' '))
        return data_list

# 参数
dataset_path = 'datasets/'
transforms = [Resize([1024, 1024])]
batch_size = 2
epochs = 100
base_lr = 2e-4

# 构建数据集
train_data = MyDataset(dataset_path, 'train', transforms)
val_data = MyDataset(dataset_path, 'val', transforms)

# 分布式 BatchSampler，注意 drop_last=True 否则可能报错
train_sampler = DistributedBatchSampler(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_sampler = DistributedBatchSampler(val_data, batch_size=batch_size, shuffle=False, drop_last=False)

# 构建模型
model = UNetPlusPlus(in_channels=6, num_classes=2, use_deconv=True)
model = paddle.DataParallel(model)

# 学习率与优化器
iters_per_epoch = len(train_sampler)
total_iters = iters_per_epoch * epochs
save_interval = iters_per_epoch * 5
log_iters = max(1, iters_per_epoch // 3)

lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(total_iters // 3), last_epoch=0.5)
optimizer = paddle.optimizer.AdamW(lr, parameters=model.parameters())

# 损失函数
mixtureLosses = [BCELoss(), LovaszSoftmaxLoss()]  
mixtureCoef = [0.8,0.2]  
losses = {'types': [MixedLoss(mixtureLosses, mixtureCoef)], 'coef': [1]}

# 训练
train(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    optimizer=optimizer,
    save_dir='output_2_3_E100_L',
    iters=total_iters,
    batch_size=batch_size,
    save_interval=save_interval,
    log_iters=log_iters,
    num_workers=0,
    losses=losses,
    use_vdl=True)

## 双卡运行 python -m paddle.distributed.launch --log_dir=log trainModel.py
