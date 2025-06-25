# unet2plus-remote-sensing-building-segmentation
基于U-Net2+的遥感建筑图像分割系统

**项目运行视频**
https://images-1324418403.cos.ap-nanjing.myqcloud.com/imgs/202506241831880.mp4、

本项目采用了U-Net++模型，这是一种改进的U-Net架构，通过引入密集跳跃连接和深度监督机制，有效解决了传统U-Net中存在的语义鸿沟问题，显著提升了分割精度。
在训练开始之前，首先通过 paddle.distributed.init_parallel_env() 初始化了多卡训练环境，确保模型能够在双卡上并行运行。这为模型训练提供了更强大的计算能力，加速了训练过程。训练过程中使用了自定义的数据集类 MyDataset，它继承自 paddle.io.Dataset。该数据集类从指定路径读取图像对（A 和 B）及其对应的标签，并对图像进行预处理，包括将图像从 BGR 格式转换为 RGB 格式、将图像对拼接在一起、应用数据增强（如果启用）等操作。此外，数据集类还支持从文本文件中读取图像路径列表，方便管理和加载数据。训练使用的模型是 UNetPlusPlus，它是一种改进的 U-Net 架构，具有更好的特征提取能力和分割精度。模型的输入通道数设置为 6（对应于拼接后的图像对），输出类别数为 2。为了支持多卡训练，模型被封装在 paddle.DataParallel 中，这使得模型能够在多个 GPU 上并行计算，从而提高训练效率。
训练过程中使用了 AdamW 优化器，它结合了 Adam 优化器的高效性和权重衰减机制，有助于防止过拟合。学习率采用了 CosineAnnealingDecay 调度策略，这种策略通过周期性地调整学习率，使得模型在训练初期能够快速收敛，而在训练后期能够更精细地调整参数，从而提高模型的最终性能。
为了提高模型的鲁棒性和分割精度，训练过程中使用了混合损失函数 MixedLoss，它结合了二元交叉熵损失（BCELoss）和 Lovasz Softmax 损失（LovaszSoftmaxLoss）。这两种损失函数分别从不同角度优化模型的性能：BCELoss 用于衡量预测值与真实标签之间的差异，而 LovaszSoftmaxLoss 则更注重于优化分割的边界精度。通过调整混合损失函数的权重（mixtureCoef），可以平衡这两种损失函数对模型训练的影响。
为了充分利用多卡资源，训练过程中使用了 DistributedBatchSampler 来对数据进行采样。这种采样器能够确保每个 GPU 上的数据是独立的，并且在每个 epoch 内对数据进行随机打乱，从而提高模型的泛化能力。此外，通过设置 drop_last=True，可以避免最后一个不完整的批次导致的潜在问题。
Batch的大小只能设置为2，否则内存溢出。

![image](https://github.com/user-attachments/assets/aea0b1ac-8bfc-451a-89e5-557a679be764)
参数设置图
模型开始训练界面：
![image](https://github.com/user-attachments/assets/eeffa64d-ab92-478f-a1dc-81b1fd874697)

模型开始训练图

系统部署的实现流程为：用户可以上传一个包含图像数据的 ZIP 文件，系统需要能够接收并解压该文件，同时验证文件的有效性，如是否为 ZIP 格式、是否包含必要的文件夹结构等。系统需要对解压后的图像进行处理，提取配对图像，并使用预加载的深度学习模型进行变化检测，生成预测掩码和概率图。将预测结果保存到指定目录，并对结果进行分析，包括计算变化覆盖率、准确率、精确率、召回率、F1 分数和 IoU 等指标。通过 Web 页面展示处理结果，并提供下载功能，将结果打包为 ZIP 文件供用户下载。可能出现的错误进行捕获和处理，并记录日志以便调试。
Flask Web 应用，负责处理用户请求，包括上传文件、调用模型进行预测、展示结果和提供下载功能。

![image](https://github.com/user-attachments/assets/74cae3e5-864f-4b58-9d13-27e9566489c8)

Flask的web应用配置
模型的加载，本项目选择直接使用model.pdparams模型，不再进行模型之间的转化，可避免后续很多不必要的麻烦。在app.py中调用 load_model 函数加载预训练的深度学习模型。该函数定义在 utils.py 中，使用 PaddlePaddle 框架加载模型参数，并将其设置为评估模式。

![image](https://github.com/user-attachments/assets/779a77a9-5004-4a49-98ca-52d38373c2c9)

加载模型函数定义
由于分割的图像是以图像对的形式出现，于是必须设计出可供zip文件上传和读取到对应img_path以及最后分析完毕数据后可供用户下载zip(数据分析结果)。

![image](https://github.com/user-attachments/assets/c8408df3-900b-4848-9301-5e6ba8370c78)

Flask后端处理文件函数
图像的处理与模型预测，通过遍历 A 和 B 文件夹中的图像文件，生成配对图像的路径映射。然后调用 predict_image 函数对每对图像进行预测，生成掩码和概率图。

![image](https://github.com/user-attachments/assets/6a502e7e-48fd-4b7d-9cb4-a5597564cd87)

predict_image 函数定义
在系统测试阶段，本项目主要采用黑盒测试方法，对系统的功能性和性能进行测试。由于是在AutoDl进行部署，需要自定义服务。

![image](https://github.com/user-attachments/assets/d648bfc0-058d-425f-a3f8-d75a25b11829)

AutoDl服务图

![image](https://github.com/user-attachments/assets/d59922ac-cbaa-4bb7-8af2-f93f79f78a44)

代理端口图

点击网址就会呈现出index.html，点击相应的按钮进行压缩包的上传，

![image](https://github.com/user-attachments/assets/46126109-9279-4885-bd1e-70f3c7a1404b)

 遥感建筑物图像分割系统首页图
 
![image](https://github.com/user-attachments/assets/2568c01f-6f37-4efb-b1be-2df6cb1ba9df)

 服务器端运行图
在系统测试阶段，本项目主要采用黑盒测试方法，对系统的功能性和性能进行测试。上传文件测试：上传不同格式的文件，如 ZIP 文件、非 ZIP 文件、空文件等，验证系统是否能够正确处理并返回相应的提示信息。上传包含不同图像文件的 ZIP 包，验证系统是否能够正确读取、配对和处理图像，并生成预测结果。通过 /download/<timestamp> 路由下载结果文件，验证生成的 ZIP 文件是否完整且包含正确的结果。上传包含大量图像的 ZIP 文件，验证系统是否能够在合理的时间内完成处理，并生成正确的结果。
系统可以通过以上测试，正常上传图片压缩包后处理好的实现界面如下：

![image](https://github.com/user-attachments/assets/b0b3a7a0-d3c1-4efe-91eb-50302a0fe6e4)

 系统处理结果图
 
在服务器端会实时输出处理的信息，信息显示如下：

![image](https://github.com/user-attachments/assets/8524fd72-4ddb-447d-9a63-97372547f38c)
