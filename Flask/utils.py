import os
import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from paddleseg.models import UNetPlusPlus
from paddleseg.transforms import Compose, Resize


def preprocess_image(image_path1, image_path2, target_size=(1024, 1024)):
    """处理两张时相图像，与MyDataset中的处理方式保持一致"""
    # 读取两个时相图像
    A_img = cv2.cvtColor(cv2.imread(image_path1), cv2.COLOR_BGR2RGB)
    B_img = cv2.cvtColor(cv2.imread(image_path2), cv2.COLOR_BGR2RGB)

    # 调整图像大小
    A_img = cv2.resize(A_img, target_size)
    B_img = cv2.resize(B_img, target_size)

    # 拼接图像
    image = np.concatenate((A_img, B_img), axis=-1)  # 6通道 [H, W, 6]

    # 转换为CHW并归一化
    image = image.transpose(2, 0, 1)  # [6, H, W]
    image = image.astype('float32') / 255.0

    # 添加批次维度
    img_tensor = paddle.to_tensor(image[np.newaxis, :])  # [1, 6, H, W]
    return img_tensor


def load_model(model_path):
    """加载模型，与测试代码保持一致"""
    model = UNetPlusPlus(in_channels=6, num_classes=2, use_deconv=True)
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()
    return model


def predict_image(model, image_path1, image_path2, target_size=(1024, 1024)):
    """预测两个时相图像的变化，返回二值掩码和概率图"""
    # 预处理图像
    input_tensor = preprocess_image(image_path1, image_path2, target_size)

    # 模型预测
    with paddle.no_grad():
        output = model(input_tensor)

        # 处理多输出情况
        if isinstance(output, (list, tuple)):
            output = output[0]  # 获取主要的分割结果

        # 获取预测结果
        pred = paddle.argmax(output[0], axis=0).numpy()  # [H, W] 值为0或1

        # 获取变化概率图
        probs = F.softmax(output[0], axis=0)[1].numpy()  # 变化类别的概率 [H, W]

    # 转换为二值掩码 (0和255)
    binary_mask = pred.astype(np.uint8) * 255

    return binary_mask, probs


def analyze_results(predictions, labels=None):
    """
    分析变化检测结果，与真实标签进行比较

    参数:
        predictions: 预测掩码列表
        labels: 真实标签掩码列表，如果提供则计算准确性指标
    """
    analysis = {
        'total_images': len(predictions),
        'change_coverage': [],  # 预测的变化区域覆盖率(IoU*100)
        'average_coverage': 0,
        'max_coverage': 0,
        'min_coverage': 100,
        'quality_scores': [],  # 将使用准确率作为质量分数
    }

    # 如果提供了标签，添加准确性评估指标
    if labels is not None:
        analysis.update({
            'accuracy': [],  # 准确率
            'precision': [],  # 精确率
            'recall': [],  # 召回率
            'f1_score': [],  # F1分数
            'iou': [],  # 交并比
            'gt_coverage': [],  # 真实标签的变化区域覆盖率
            'original_coverage': [],  # 保存原始像素覆盖率
        })

    for i, pred in enumerate(predictions):
        # 确保mask是NumPy数组
        if isinstance(pred, list):
            pred = np.array(pred)

        # 二值化预测掩码 (确保是0或1)
        pred_binary = (pred > 0).astype(np.uint8)

        # 计算原始预测变化覆盖率（像素比例）
        original_coverage = np.mean(pred_binary) * 100

        # 如果提供了标签，使用IoU作为覆盖率；否则使用原始像素覆盖率
        if labels is not None and i < len(labels):
            label = labels[i]
            if isinstance(label, list):
                label = np.array(label)

            # 确保标签是二值图像 (0或1)
            label_binary = (label > 0).astype(np.uint8)

            # 调整标签大小以匹配预测结果（如果需要）
            if label_binary.shape != pred_binary.shape:
                label_binary = cv2.resize(label_binary,
                                          (pred_binary.shape[1], pred_binary.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

            # 计算真实标签的变化覆盖率
            gt_coverage = np.mean(label_binary) * 100
            analysis['gt_coverage'].append(gt_coverage)
            analysis['original_coverage'].append(original_coverage)

            # 计算评估指标
            tp = np.sum((pred_binary == 1) & (label_binary == 1))  # 真正例
            fp = np.sum((pred_binary == 1) & (label_binary == 0))  # 假正例
            tn = np.sum((pred_binary == 0) & (label_binary == 0))  # 真负例
            fn = np.sum((pred_binary == 0) & (label_binary == 1))  # 假负例

            # 准确率: (TP+TN)/(TP+TN+FP+FN)
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            # 精确率: TP/(TP+FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # 召回率: TP/(TP+FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # F1分数: 2*precision*recall/(precision+recall)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            # IoU: TP/(TP+FP+FN)
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            # 将IoU值乘以100作为change_coverage
            pred_coverage = iou * 100

            # 使用准确率作为质量分数
            quality_score = accuracy

            analysis['accuracy'].append(accuracy)
            analysis['precision'].append(precision)
            analysis['recall'].append(recall)
            analysis['f1_score'].append(f1)
            analysis['iou'].append(iou)
            analysis['quality_scores'].append(quality_score)
        else:
            # 没有标签时，使用原始像素覆盖率
            pred_coverage = original_coverage

            # 没有标签时，使用边缘清晰度作为质量分数（或其他适当的度量）
            edges = cv2.Canny(pred.astype(np.uint8), 50, 150)
            edge_score = np.sum(edges > 0) / np.prod(pred.shape)
            analysis['quality_scores'].append(edge_score)

        analysis['change_coverage'].append(pred_coverage)
        analysis['max_coverage'] = max(analysis['max_coverage'], pred_coverage)
        analysis['min_coverage'] = min(analysis['min_coverage'], pred_coverage)

    # 计算平均值
    analysis['average_coverage'] = np.mean(analysis['change_coverage'])
    analysis['avg_quality'] = np.mean(analysis['quality_scores'])

    if labels is not None:
        analysis['avg_accuracy'] = np.mean(analysis['accuracy'])
        analysis['avg_precision'] = np.mean(analysis['precision'])
        analysis['avg_recall'] = np.mean(analysis['recall'])
        analysis['avg_f1_score'] = np.mean(analysis['f1_score'])
        analysis['avg_iou'] = np.mean(analysis['iou'])
        analysis['avg_gt_coverage'] = np.mean(analysis['gt_coverage'])
        analysis['avg_original_coverage'] = np.mean(analysis['original_coverage'])

    return analysis


def generate_report(analysis):
    """生成分析报告，包含标签比较结果"""
    report = {
        'summary': f"共处理 {analysis['total_images']} 张图像，平均预测变化覆盖率: {analysis['average_coverage']:.2f}%",
        'coverage_range': f"预测覆盖范围: {analysis['min_coverage']:.2f}% - {analysis['max_coverage']:.2f}%",
        'quality': f"平均分割质量: {analysis['avg_quality']:.4f}",
    }

    # 如果有标签评估数据
    if 'avg_accuracy' in analysis:
        report.update({
            'gt_coverage': f"真实标签平均变化覆盖率: {analysis['avg_gt_coverage']:.2f}%",
            'original_coverage': f"原始像素平均覆盖率: {analysis['avg_original_coverage']:.2f}%",
            'accuracy': f"平均准确率: {analysis['avg_accuracy']:.4f}",
            'precision': f"平均精确率: {analysis['avg_precision']:.4f}",
            'recall': f"平均召回率: {analysis['avg_recall']:.4f}",
            'f1_score': f"平均F1分数: {analysis['avg_f1_score']:.4f}",
            'iou': f"平均IoU: {analysis['avg_iou']:.4f}",
        })

    report['recommendations'] = [
        "对于IoU<0.3的图像，建议检查检测质量",
        "对于精确率<0.5的图像，可能存在过多误检",
        "对于召回率<0.5的图像，可能存在漏检",
        "对于真实标签覆盖率与预测覆盖率差异>20%的图像，应检查模型偏差"
    ]

    report['next_steps'] = [
        "增加更多多样化场景的训练数据",
        "针对低IoU样本进行模型微调",
        "考虑使用后处理技术（如形态学操作）提高边界准确性",
        "针对高假阳性/假阴性区域的特征进行分析"
    ]

    return report


def process_dataset(model, dataset_path, output_dir):
    """处理完整数据集并保存结果，包含标签比较"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取测试列表
    list_path = os.path.join(dataset_path, 'test_list.txt')
    data_list = []
    with open(list_path, 'r') as f:
        data = f.readlines()
        for d in data:
            data_list.append(d.strip().split(' '))

    results = []
    predictions = []
    labels = []

    for idx, (img_path1, img_path2, label_path) in enumerate(tqdm(data_list, desc="Processing images")):
        # 预测变化
        binary_mask, probs = predict_image(model, img_path1, img_path2)
        predictions.append(binary_mask)

        # 获取基础文件名
        base_name = os.path.basename(img_path1).split('.')[0]

        # 读取标签（如果有）
        if os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (binary_mask.shape[1], binary_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            labels.append(label)

            # 创建可视化比较
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(cv2.imread(img_path2)[:, :, ::-1])  # BGR to RGB
            axes[0].set_title('第二时相图像')
            axes[0].axis('off')

            axes[1].imshow(label, cmap='gray')
            axes[1].set_title('真实标签')
            axes[1].axis('off')

            axes[2].imshow(binary_mask, cmap='gray')
            axes[2].set_title('预测结果')
            axes[2].axis('off')

            plt.tight_layout()
            comp_file = os.path.join(output_dir, f'compare_{base_name}.png')
            plt.savefig(comp_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            label = None

        # 保存二值掩码
        mask_file = os.path.join(output_dir, f'mask_{base_name}.png')
        cv2.imwrite(mask_file, binary_mask)

        # 保存概率图
        prob_file = os.path.join(output_dir, f'prob_{base_name}.png')
        plt.imsave(prob_file, probs, cmap='jet')

        results.append({
            'image1': img_path1,
            'image2': img_path2,
            'label': label_path,
            'prediction': mask_file,
            'probability': prob_file
        })

    # 生成分析报告
    analysis = analyze_results(predictions, labels if labels else None)
    report = generate_report(analysis)

    # 保存报告
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write(report['summary'] + '\n')
        f.write(report['coverage_range'] + '\n')
        f.write(report['quality'] + '\n')

        if 'accuracy' in report:
            f.write(report['gt_coverage'] + '\n')
            f.write(report['accuracy'] + '\n')
            f.write(report['precision'] + '\n')
            f.write(report['recall'] + '\n')
            f.write(report['f1_score'] + '\n')
            f.write(report['iou'] + '\n')

        f.write('\n推荐:\n')
        for rec in report['recommendations']:
            f.write('- ' + rec + '\n')
        f.write('\n后续步骤:\n')
        for step in report['next_steps']:
            f.write('- ' + step + '\n')

    return results, analysis


def save_all_results(image_pairs, masks, result_dir, labels=None):
    """保存所有结果到指定目录"""
    os.makedirs(result_dir, exist_ok=True)
    results = []

    # 设置中文字体
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    for i, ((img_path1, img_path2), mask) in enumerate(zip(image_pairs, masks)):
        base_name = os.path.basename(img_path1)

        # 保存掩码图像
        mask_file = os.path.join(result_dir, f'mask_{base_name}')
        import cv2
        cv2.imwrite(mask_file, mask)

        # 读取两个时间点的图像
        img1 = cv2.imread(img_path1)[:, :, ::-1]  # BGR to RGB
        img2 = cv2.imread(img_path2)[:, :, ::-1]  # BGR to RGB

        # 创建合成原图（水平拼接两个时间点的图像）
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2

        # 创建空白画布
        combined_img = np.zeros((h, w, 3), dtype=np.uint8)
        combined_img[:h1, :w1] = img1
        combined_img[:h2, w1:w1 + w2] = img2

        # 在两张图之间添加分隔线
        combined_img[:, w1 - 2:w1 + 2, :] = [255, 0, 0]  # 红色分隔线

        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_img, "Time Point A", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_img, "Time Point B", (w1 + 10, 30), font, 1, (255, 255, 255), 2)

        # 保存合成原图
        combined_file = os.path.join(result_dir, f'combined_{base_name}')
        cv2.imwrite(combined_file, combined_img[:, :, ::-1])  # RGB to BGR

        # 创建可视化图
        if labels is not None and i < len(labels) and labels[i] is not None:
            # 如果有标签，创建3图比较
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 第一张图显示合成原图
            axes[0].imshow(combined_img)
            axes[0].set_title('original image A/B')
            axes[0].axis('off')

            axes[1].imshow(labels[i], cmap='gray')
            axes[1].set_title('True Label')
            axes[1].axis('off')

            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('prediction')
            axes[2].axis('off')
        else:
            # 如果没有标签，创建2图比较
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 第一张图显示合成原图
            axes[0].imshow(combined_img)
            axes[0].set_title('original image A/B')
            axes[0].axis('off')

            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('results')
            axes[1].axis('off')

        plt.tight_layout()
        comp_file = os.path.join(result_dir, f'compare_{base_name}')
        plt.savefig(comp_file, dpi=300, bbox_inches='tight')
        plt.close()

        # 创建相对URL路径
        results_relative = os.path.relpath(result_dir, 'static')

        # 初始化result_info
        result_info = {
            'filename': base_name,
            'original': f"/static/{results_relative}/combined_{base_name}",  # 合成原图
            'mask': f"/static/{results_relative}/mask_{base_name}",
            'visualization': f"/static/{results_relative}/compare_{base_name}",
        }

        # 如果有标签，计算IoU并用作change_ratio，accuracy用作quality
        if labels is not None and i < len(labels) and labels[i] is not None:
            label = labels[i]
            true_pos = ((mask > 0) & (label > 0)).sum()
            false_pos = ((mask > 0) & (label == 0)).sum()
            false_neg = ((mask == 0) & (label > 0)).sum()
            true_neg = ((mask == 0) & (label == 0)).sum()

            accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg) if (
                                                                                                        true_pos + true_neg + false_pos + false_neg) > 0 else 0
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            iou = true_pos / (true_pos + false_pos + false_neg) if (true_pos + false_pos + false_neg) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # 使用IoU作为change_ratio
            change_ratio = iou * 100  # 将IoU转换为百分比
            # 使用accuracy作为quality
            quality = accuracy

            # 计算原始覆盖率（仅作为参考）
            original_change_ratio = (mask > 0).sum() / (mask.shape[0] * mask.shape[1]) * 100
            gt_coverage = (label > 0).sum() / (label.shape[0] * label.shape[1]) * 100

            result_info.update({
                'change_ratio': change_ratio,  # 使用IoU作为变化覆盖率
                'quality': quality,  # 使用accuracy作为质量指标
                'has_label': True,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'iou': iou,
                'f1_score': f1_score,
                'original_change_ratio': original_change_ratio,  # 保留原始覆盖率作为额外信息
                'gt_coverage': gt_coverage  # 真实标签覆盖率
            })
        else:
            # 没有标签时，使用原始计算方法并默认quality为0
            original_change_ratio = (mask > 0).sum() / (mask.shape[0] * mask.shape[1]) * 100
            result_info.update({
                'change_ratio': original_change_ratio,  # 没有标签时使用原始覆盖率
                'quality': 0.0,  # 没有标签时默认质量为0
            })

        # 添加调试打印
        print(
            f"生成的URL - 合成原图: {result_info['original']}, 掩码: {result_info['mask']}, 可视化: {result_info['visualization']}")

        if 'iou' in result_info:
            print(
                f"覆盖率(IoU): {result_info['change_ratio']:.4f}%, 质量(Accuracy): {result_info['quality']:.4f}, 原始覆盖率: {result_info['original_change_ratio']:.4f}%")
        else:
            print(f"覆盖率: {result_info['change_ratio']:.4f}%, 质量: {result_info['quality']:.4f}")

        results.append(result_info)

    return results
