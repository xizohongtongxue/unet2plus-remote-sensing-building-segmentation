document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const uploadArea = document.querySelector('.upload-area');
    const folderButton = uploadArea.querySelector('.btn-custom');
    // 确保正确获取处理按钮，更明确的选择器
    const processButton = document.querySelector('.card-body .btn-custom[class*="px-4"]') ||
                     document.querySelector('.card-body .btn-custom:has(i.fas.fa-cogs)') ||
                     Array.from(document.querySelectorAll('.card-body .btn-custom')).find(btn =>
                         btn.textContent.trim().includes('开始处理')
                     );

    const saveResultsCheckbox = document.getElementById('saveResults');
    const generateReportCheckbox = document.getElementById('generateReport');
    const backToHomeButton = document.getElementById('backToHome');

    // 创建一个隐藏的文件输入元素
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = false; // 只允许选择一个文件
    fileInput.accept = '.zip';  // 只接受 zip 文件
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    // 选择文件后处理
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file && file.name.endsWith('.zip')) {
            selectedFiles = [file];
            updateUploadAreaStatus();
        } else {
            showNotification('请选择一个 .zip 文件', 'warning');
            selectedFiles = [];
            updateUploadAreaStatus();
        }
    });

    // 文件列表和上传状态
    let selectedFiles = [];
    let isProcessing = false;

    // 拖放事件处理
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];

        if (file && file.name.endsWith('.zip')) {
            selectedFiles = [file];
            updateUploadAreaStatus();
        } else {
            showNotification('请上传一个 .zip 文件', 'warning');
        }
    });

    // 点击上传区域触发文件选择
    uploadArea.addEventListener('click', () => {
        if (!isProcessing) {
            fileInput.click();
        }
    });

    // 点击"选择文件夹"按钮
    folderButton.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!isProcessing) {
            fileInput.click();
        }
    });

    // 点击"开始处理"按钮
    processButton.addEventListener('click', () => {
        if (selectedFiles.length > 0 && !isProcessing) {
            processImages();
        } else if (selectedFiles.length === 0) {
            showNotification('请先选择ZIP文件', 'warning');
        }
    });

    // 如果存在返回首页按钮，添加事件监听
    if (backToHomeButton) {
        backToHomeButton.addEventListener('click', () => {
            window.location.href = '/';
        });
    }

    // 更新上传区域状态
    function updateUploadAreaStatus() {
        if (selectedFiles.length > 0) {
            uploadArea.querySelector('h4').textContent = `已选择: ${selectedFiles[0].name}`;
            uploadArea.querySelector('p').textContent = '点击"开始处理"按钮上传ZIP文件并进行处理';
            uploadArea.classList.add('has-files');
        } else {
            uploadArea.querySelector('h4').textContent = '拖放ZIP文件或点击此处选择';
            uploadArea.querySelector('p').textContent = '请上传包含A和B(以及可选的label)文件夹的ZIP文件';
            uploadArea.classList.remove('has-files');
        }
    }

    // 处理图像
    function processImages() {
        if (selectedFiles.length === 0) return;

        isProcessing = true;

        // 创建并显示进度覆盖层
        showProgressOverlay(1);

        // 创建FormData对象
        const formData = new FormData();
        formData.append('zip_file', selectedFiles[0]);

        // 添加处理选项
        if (saveResultsCheckbox && generateReportCheckbox) {
            formData.append('saveResults', saveResultsCheckbox.checked ? '1' : '0');
            formData.append('generateReport', generateReportCheckbox.checked ? '1' : '0');
        }

        // 发送请求到后端
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(text || '处理请求失败');
                });
            }
            return response.text();
        })
        .then(html => {
            // 处理成功，用返回的HTML替换当前页面内容
            document.open();
            document.write(html);
            document.close();

            // 初始化结果页面的图表
            if (typeof initResultsCharts === 'function') {
                initResultsCharts();
            }
        })
        .catch(error => {
            hideProgressOverlay();
            showNotification(error.message, 'error');
            isProcessing = false;
        });

        // 模拟进度更新
        simulateProgressUpdates(1);
    }

    // 显示进度覆盖层
    function showProgressOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'progress-overlay';
        overlay.innerHTML = `
            <div class="progress-container">
                <h3>正在处理变化检测...</h3>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="processingProgressBar"></div>
                </div>
                <div class="progress-details">
                    <p id="processingStatus">初始化处理...</p>
                    <p>正在处理您的ZIP文件</p>
                </div>
                <div class="processing-stats">
                    <div class="stat-item">
                        <span class="stat-label">预计剩余时间:</span>
                        <span class="stat-value" id="remainingTime">计算中...</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">处理阶段:</span>
                        <span class="stat-value" id="processingStage">解压文件...</span>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);
    }

    // 隐藏进度覆盖层
    function hideProgressOverlay() {
        const overlay = document.querySelector('.progress-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    // 更新进度信息
    function updateProgress(percentage, status, remainingTime, stage) {
        const progressBar = document.getElementById('processingProgressBar');
        const statusElement = document.getElementById('processingStatus');
        const remainingTimeElement = document.getElementById('remainingTime');
        const stageElement = document.getElementById('processingStage');

        if (progressBar && statusElement) {
            progressBar.style.width = `${percentage}%`;
            statusElement.textContent = status;

            if (remainingTimeElement && stageElement) {
                remainingTimeElement.textContent = remainingTime;
                stageElement.textContent = stage;
            }
        }
    }

    // 模拟进度更新（实际项目中应通过API获取真实进度）
    function simulateProgressUpdates() {
        const stages = [
            "解压文件...",
            "查找图像文件...",
            "加载模型...",
            "处理图像对...",
            "生成变化掩码...",
            "计算评估指标...",
            "生成分析报告...",
            "保存结果..."
        ];

        let percentage = 0;
        const startTime = new Date().getTime();

        const updateInterval = setInterval(() => {
            percentage += Math.random() * 5;

            if (percentage > 100) {
                percentage = 100;
                clearInterval(updateInterval);
            }

            // 根据百分比选择当前阶段
            const stageIndex = Math.min(Math.floor(percentage / (100 / stages.length)), stages.length - 1);
            const currentStage = stages[stageIndex];

            // 计算剩余时间
            const currentTime = new Date().getTime();
            const elapsedSeconds = (currentTime - startTime) / 1000;
            let remainingTime = "计算中...";

            if (percentage > 20) {
                const totalEstimatedSeconds = (elapsedSeconds / percentage) * 100;
                const remainingSeconds = totalEstimatedSeconds - elapsedSeconds;

                if (remainingSeconds < 60) {
                    remainingTime = Math.ceil(remainingSeconds) + " 秒";
                } else if (remainingSeconds < 3600) {
                    remainingTime = Math.floor(remainingSeconds / 60) + " 分 " + Math.ceil(remainingSeconds % 60) + " 秒";
                } else {
                    remainingTime = Math.floor(remainingSeconds / 3600) + " 小时 " +
                                   Math.floor((remainingSeconds % 3600) / 60) + " 分";
                }
            }

            // 更新状态文本
            const status = percentage < 100 ?
                          `处理中 (${Math.floor(percentage)}%)` :
                          "处理完成，正在加载结果...";

            updateProgress(percentage, status, remainingTime, currentStage);

            // 进度完成后最后一次更新
            if (percentage >= 100) {
                updateProgress(100, "处理完成，正在加载结果...", "即将完成", "完成");
            }

        }, 500);
    }

    // 初始化结果页面图表（如果在结果页面）
    if (document.getElementById('coverageChart') && document.getElementById('qualityChart')) {
        initResultsCharts();
    }

    // 初始化结果页面的图表
    function initResultsCharts() {
        // 建筑覆盖率分布图
        const coverageCtx = document.getElementById('coverageChart').getContext('2d');
        new Chart(coverageCtx, {
            type: 'bar',
            data: {
                labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
                datasets: [{
                    label: '图像数量',
                    data: window.coverageData || [2, 4, 6, 5, 3, 2, 1, 0, 0, 1],
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '图像数量'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '变化覆盖率'
                        }
                    }
                }
            }
        });

        // 分割质量评分分布图
        const qualityCtx = document.getElementById('qualityChart').getContext('2d');
        new Chart(qualityCtx, {
            type: 'pie',
            data: {
                labels: ['优秀 (90-100)', '良好 (80-90)', '一般 (70-80)', '需改进 (<70)'],
                datasets: [{
                    data: window.qualityData || [12, 8, 3, 1],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(255, 99, 132, 0.7)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right'
                    }
                }
            }
        });
    }

    // 显示通知
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        // 设置动画
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);

        // 几秒后自动关闭
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    // 为进度显示添加一些CSS样式
    addProgressStyles();

    // 添加进度显示所需的CSS样式
    function addProgressStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .progress-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
            }
            
            .progress-container {
                background-color: white;
                border-radius: 8px;
                padding: 30px;
                width: 90%;
                max-width: 600px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            
            .progress-container h3 {
                margin-top: 0;
                margin-bottom: 20px;
                text-align: center;
                color: #333;
            }
            
            .progress-bar-container {
                height: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
                margin-bottom: 15px;
                overflow: hidden;
            }
            
            .progress-bar {
                height: 100%;
                background: linear-gradient(90deg, #4caf50, #8bc34a);
                width: 0%;
                transition: width 0.3s ease;
                border-radius: 10px;
                position: relative;
            }
            
            .progress-details {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
                color: #555;
            }
            
            .processing-stats {
                background-color: #f9f9f9;
                border-radius: 6px;
                padding: 15px;
                margin-top: 15px;
            }
            
            .stat-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }
            
            .stat-label {
                color: #666;
                font-weight: 500;
            }
            
            .stat-value {
                font-weight: 600;
                color: #333;
            }
            
            #processingStatus {
                font-style: italic;
                color: #2196F3;
                max-width: 70%;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 25px;
                border-radius: 5px;
                background-color: #333;
                color: white;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                z-index: 10000;
                opacity: 0;
                transform: translateY(-20px);
                transition: opacity 0.3s, transform 0.3s;
            }
            
            .notification.show {
                opacity: 1;
                transform: translateY(0);
            }
            
            .notification.info {
                background-color: #2196F3;
            }
            
            .notification.success {
                background-color: #4CAF50;
            }
            
            .notification.warning {
                background-color: #FFC107;
                color: #333;
            }
            
            .notification.error {
                background-color: #F44336;
            }
        `;
        document.head.appendChild(style);
    }
});
