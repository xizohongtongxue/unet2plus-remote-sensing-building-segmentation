import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset
from paddleseg.models import UNetPlusPlus
from paddleseg.transforms import Compose, Resize
import matplotlib.pyplot as plt


# ------------------ 数据集定义（与mergeTrain一致） ------------------ #
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

        # 自己调整大小
        A_img = cv2.resize(A_img, (1024, 1024))
        B_img = cv2.resize(B_img, (1024, 1024))
        label = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        image = np.concatenate((A_img, B_img), axis=-1)

        # 不使用 transforms
        image = image.transpose(2, 0, 1)
        image = image.astype('float32') / 255.0
        label = label.astype('int64')
        if self.mode != 'test':
            label = label.clip(max=1)
            label = label[np.newaxis, :]
        return {'img': image, 'label': label, 'trans_info': [], 'paths': (A_path, B_path, lab_path)}

    def __len__(self):
        return self.data_num

    def __get_list(self, list_path):
        data_list = []
        with open(list_path, 'r') as f:
            data = f.readlines()
            for d in data:
                data_list.append(d.strip().split(' '))
        return data_list


# ------------------ 加载模型 ------------------ #
model_path = 'model.pdparams'
model = UNetPlusPlus(in_channels=6, num_classes=2, use_deconv=True)
para_state_dict = paddle.load(model_path)
model.set_dict(para_state_dict)
model.eval()  # 设置为评估模式

# ------------------ 加载验证数据 ------------------ #
dataset_path = r'C:\Users\Mayn\Desktop\jlh\jlh\data\datasets'
transforms = [Resize([1024, 1024])]
test_data = MyDataset(dataset_path, 'test', transforms)

# 创建结果保存目录
output_dir = 'test_results'
os.makedirs(output_dir, exist_ok=True)

# ------------------ 测试全部数据 ------------------ #
for idx in range(len(test_data)):
    sample = test_data[idx]
    img = paddle.to_tensor(sample['img']).unsqueeze(0)  # [1, 6, 1024, 1024]
    lab = sample['label']  # [1, 1024, 1024]

    # 获取文件名用于保存结果
    paths = sample['paths']
    base_name = os.path.basename(paths[0]).split('.')[0]  # 使用第一张图片的名称作为基础名

    # 预测
    pred = model(img)
    # 如果pred是列表，取第一个元素
    if isinstance(pred, list):
        pred = pred[0]  # 获取主要的分割结果

    # 准备显示图像
    s_img = img.squeeze(0).numpy().transpose(1, 2, 0)  # [1024, 1024, 6]
    s_A_img = s_img[:, :, 0:3]
    s_B_img = s_img[:, :, 3:6]
    lab_img = lab.reshape((1024, 1024))
    pre_img = paddle.argmax(pred.squeeze(0), axis=0).astype('uint8').numpy()

    # 创建图形并保存
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(s_A_img)
    plt.title('Time 1')

    plt.subplot(2, 2, 2)
    plt.imshow(s_B_img)
    plt.title('Time 2')

    plt.subplot(2, 2, 3)
    plt.imshow(lab_img)
    plt.title('Label')

    plt.subplot(2, 2, 4)
    plt.imshow(pre_img)
    plt.title('Prediction')

    # 保存图像到文件
    output_file = os.path.join(output_dir, f'result_{base_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存

    print(f"Processed sample {idx + 1}/{len(test_data)}: {output_file}")
