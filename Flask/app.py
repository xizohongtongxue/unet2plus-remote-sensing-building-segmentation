import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import shutil
from datetime import datetime
import zipfile
import traceback
from utils import load_model, predict_image, save_all_results, analyze_results, generate_report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# 加载模型
model = load_model("model.pdparams")
app.logger.info("Model loaded successfully.")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_folder():
    image_pairs = []
    masks = []
    labels = []
    extract_dir = None

    try:
        if 'zip_file' not in request.files:
            return '没有收到 zip 文件', 400

        zip_file = request.files['zip_file']

        if zip_file.filename == '':
            return '空文件', 400

        filename = secure_filename(zip_file.filename)
        if not filename.endswith('.zip'):
            return '请上传 zip 文件', 400

        # 创建临时目录
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
        os.makedirs(extract_dir, exist_ok=True)

        # 保存并解压 zip 文件
        zip_path = os.path.join(extract_dir, filename)
        zip_file.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            return '解压失败，请确认文件是有效的 zip 格式', 400

        # 检查必要的文件夹
        dir_A = os.path.join(extract_dir, 'A')
        dir_B = os.path.join(extract_dir, 'B')
        dir_label = os.path.join(extract_dir, 'label')

        if not os.path.exists(dir_A) or not os.path.exists(dir_B):
            return '压缩包中必须包含A和B文件夹', 400

        has_labels = os.path.exists(dir_label)

        # 遍历 A 和 B，找出配对图像
        images_A = {os.path.basename(f): os.path.join(dir_A, f)
                    for f in os.listdir(dir_A) if allowed_file(f)}
        images_B = {os.path.basename(f): os.path.join(dir_B, f)
                    for f in os.listdir(dir_B) if allowed_file(f)}

        # 如果有标签文件夹，也获取标签
        if has_labels:
            images_label = {os.path.basename(f): os.path.join(dir_label, f)
                            for f in os.listdir(dir_label) if allowed_file(f)}
        else:
            images_label = {}

        # 取A和B都有的图像名
        common_filenames = sorted(set(images_A.keys()) & set(images_B.keys()))
        if not common_filenames:
            return '在 A 和 B 文件夹中找不到同名图像文件', 400

        # 处理所有图像对
        for fname in common_filenames:
            img_path1 = images_A[fname]
            img_path2 = images_B[fname]
            mask, _ = predict_image(model, img_path1, img_path2)
            masks.append(mask)
            image_pairs.append((img_path1, img_path2))

            # 如果存在对应的标签，添加到标签列表
            if fname in images_label:
                label_path = images_label[fname]
                import cv2
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                # 确保标签的大小与掩码相同
                if label.shape != mask.shape:
                    label = cv2.resize(label, (mask.shape[1], mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
                labels.append(label)
            else:
                labels.append(None)

        print("找到的 A 路径：", dir_A)
        print("找到的 B 路径：", dir_B)
        print("找到的 label 路径：", dir_label if has_labels else "无标签目录")
        print("公共图像文件：", common_filenames)

        # 保存结果
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], timestamp)
        results = save_all_results(image_pairs, masks, result_dir, labels if has_labels else None)

        # 分析结果
        analysis = analyze_results(masks, labels if has_labels and labels else None)
        report = generate_report(analysis)

        return render_template('results.html',
                               results=results,
                               report=report,
                               timestamp=timestamp,
                               has_labels=has_labels)

    except Exception as e:
        app.logger.error(f"处理错误: {str(e)}")
        app.logger.error(traceback.format_exc())

        # 打印调试信息
        print("解压路径：", extract_dir)
        print("图像对数量：", len(image_pairs))
        print("服务器处理错误：", traceback.format_exc())

        return f'服务器内部错误：{str(e)}', 500


@app.route('/download/<timestamp>')
def download_results(timestamp):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], timestamp)
    zip_path = os.path.join(app.config['RESULTS_FOLDER'], f'results_{timestamp}.zip')

    # 创建ZIP文件
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', result_dir)

    return send_file(zip_path, as_attachment=True)


if __name__ == '__main__':
    # 确保上传和结果目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=6006, debug=False)
