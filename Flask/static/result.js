document.addEventListener('DOMContentLoaded', function() {
    // 初始化结果页面的图表
    initResultCharts();

    // 处理结果页面的交互
    const resultCards = document.querySelectorAll('.result-card');
    if (resultCards.length > 0) {
        // 给每个结果卡片添加点击事件，放大查看图片
        resultCards.forEach(card => {
            const imgContainer = card.querySelector('.img-container');
            const img = card.querySelector('.img-container img');

            if (imgContainer && img) {
                img.addEventListener('click', function() {
                    // 创建模态框来显示大图
                    const modal = document.createElement('div');
                    modal.className = 'modal-overlay';
                    modal.style.position = 'fixed';
                    modal.style.top = '0';
                    modal.style.left = '0';
                    modal.style.width = '100%';
                    modal.style.height = '100%';
                    modal.style.background = 'rgba(0,0,0,0.8)';
                    modal.style.display = 'flex';
                    modal.style.justifyContent = 'center';
                    modal.style.alignItems = 'center';
                    modal.style.zIndex = '1000';

                    const modalImg = document.createElement('img');
                    modalImg.src = img.src;
                    modalImg.style.maxWidth = '90%';
                    modalImg.style.maxHeight = '90%';
                    modalImg.style.objectFit = 'contain';

                    modal.appendChild(modalImg);
                    document.body.appendChild(modal);

                    // 点击模态框关闭
                    modal.addEventListener('click', function() {
                        document.body.removeChild(modal);
                    });
                });
            }
        });
    }

    // 初始化Bootstrap的Tab组件
    const reportTabs = document.getElementById('reportTabs');
    if (reportTabs) {
        const tabTriggerList = [].slice.call(reportTabs.querySelectorAll('button[data-bs-toggle="tab"]'));
        tabTriggerList.forEach(function (tabTriggerEl) {
            new bootstrap.Tab(tabTriggerEl);
        });
    }

    // 进度条动画效果（如果有进度条）
    const progressBars = document.querySelectorAll('.progress-bar');
    if (progressBars.length > 0) {
        progressBars.forEach(bar => {
            const width = bar.getAttribute('aria-valuenow') + '%';
            bar.style.width = '0';
            setTimeout(() => {
                bar.style.transition = 'width 1s ease-in-out';
                bar.style.width = width;
            }, 100);
        });
    }

    // 下载结果按钮
    const downloadBtn = document.getElementById('downloadResults');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            const timestamp = this.getAttribute('data-timestamp');
            if (timestamp) {
                window.location.href = `/download/${timestamp}`;
            }
        });
    }

    // 返回首页按钮
    const backToHomeBtn = document.getElementById('backToHome');
    if (backToHomeBtn) {
        backToHomeBtn.addEventListener('click', function() {
            window.location.href = '/';
        });
    }

    // 添加图像对比滑块功能
    setupImageComparison();

    // 添加结果过滤功能
    setupResultFiltering();
});

// 初始化结果页面图表
function initResultCharts() {
    // 变化覆盖率分布图
    const coverageChart = document.getElementById('coverageChart');
    if (coverageChart) {
        const ctx = coverageChart.getContext('2d');
        new Chart(ctx, {
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
    }

    // 检测质量评分分布图
    const qualityChart = document.getElementById('qualityChart');
    if (qualityChart) {
        const ctx = qualityChart.getContext('2d');
        new Chart(ctx, {
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

    // 如果有标签数据，添加评估指标图表
    const metricsChart = document.getElementById('metricsChart');
    if (metricsChart && window.hasLabels) {
        const ctx = metricsChart.getContext('2d');
        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['准确率', '精确率', '召回率', 'F1分数', 'IoU'],
                datasets: [{
                    label: '平均评估指标',
                    data: [
                        window.avgAccuracy || 0.85,
                        window.avgPrecision || 0.78,
                        window.avgRecall || 0.82,
                        window.avgF1 || 0.80,
                        window.avgIoU || 0.75
                    ],
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.2
                        }
                    }
                }
            }
        });
    }
}

// 设置图像对比滑块功能
function setupImageComparison() {
    const comparisons = document.querySelectorAll('.image-comparison');
    if (comparisons.length > 0) {
        comparisons.forEach(container => {
            const slider = container.querySelector('.comparison-slider');
            const beforeImage = container.querySelector('.before-image');
            const afterImage = container.querySelector('.after-image');

            if (slider && beforeImage && afterImage) {
                // 设置初始位置
                slider.style.left = '50%';
                afterImage.style.width = '50%';

                // 添加拖动功能
                let isDragging = false;

                slider.addEventListener('mousedown', () => {
                    isDragging = true;
                });

                window.addEventListener('mouseup', () => {
                    isDragging = false;
                });

                window.addEventListener('mousemove', (e) => {
                    if (!isDragging) return;

                    const containerRect = container.getBoundingClientRect();
                    const x = e.clientX - containerRect.left;
                    const percentage = Math.max(0, Math.min(100, (x / containerRect.width) * 100));

                    slider.style.left = `${percentage}%`;
                    afterImage.style.width = `${percentage}%`;
                });
            }
        });
    }
}

// 设置结果过滤功能
function setupResultFiltering() {
    const filterInput = document.getElementById('resultFilter');
    const sortSelect = document.getElementById('resultSort');

    if (filterInput) {
        filterInput.addEventListener('input', filterResults);
    }

    if (sortSelect) {
        sortSelect.addEventListener('change', sortResults);
    }

    function filterResults() {
        const searchTerm = filterInput.value.toLowerCase();
        const resultCards = document.querySelectorAll('.result-card');

        resultCards.forEach(card => {
            const fileName = card.getAttribute('data-filename').toLowerCase();
            if (fileName.includes(searchTerm)) {
                card.style.display = '';
            } else {
                card.style.display = 'none';
            }
        });
    }

    function sortResults() {
        const sortValue = sortSelect.value;
        const resultContainer = document.querySelector('.results-container');
        const resultCards = Array.from(document.querySelectorAll('.result-card'));

        resultCards.sort((a, b) => {
            if (sortValue === 'name-asc') {
                return a.getAttribute('data-filename').localeCompare(b.getAttribute('data-filename'));
            } else if (sortValue === 'name-desc') {
                return b.getAttribute('data-filename').localeCompare(a.getAttribute('data-filename'));
            } else if (sortValue === 'change-asc') {
                return parseFloat(a.getAttribute('data-change')) - parseFloat(b.getAttribute('data-change'));
            } else if (sortValue === 'change-desc') {
                return parseFloat(b.getAttribute('data-change')) - parseFloat(a.getAttribute('data-change'));
            }

            // 如果有评估指标，添加按评估指标排序
            if (window.hasLabels) {
                if (sortValue === 'iou-desc') {
                    return parseFloat(b.getAttribute('data-iou') || 0) - parseFloat(a.getAttribute('data-iou') || 0);
                } else if (sortValue === 'iou-asc') {
                    return parseFloat(a.getAttribute('data-iou') || 0) - parseFloat(b.getAttribute('data-iou') || 0);
                }
            }

            return 0;
        });

        // 重新排列DOM元素
        resultCards.forEach(card => {
            resultContainer.appendChild(card);
        });
    }
}

// 下载单个图像结果
function downloadImage(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// 显示通知消息
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// 添加必要的样式
(function() {
    const style = document.createElement('style');
    style.textContent = `
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            cursor: pointer;
        }
        
        .image-comparison {
            position: relative;
            width: 100%;
            height: 300px;
            overflow: hidden;
        }
        
        .before-image, .after-image {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            overflow: hidden;
        }
        
        .before-image {
            width: 100%;
        }
        
        .after-image {
            width: 50%;
        }
        
        .before-image img, .after-image img {
            height: 100%;
            width: auto;
            object-fit: cover;
        }
        
        .comparison-slider {
            position: absolute;
            top: 0;
            left: 50%;
            height: 100%;
            width: 4px;
            background: #fff;
            cursor: ew-resize;
            z-index: 10;
        }
        
        .comparison-slider::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #fff;
            box-shadow: 0 0 5px rgba(0,0,0,0.5);
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
        
        .results-filter-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        
        .results-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .result-card {
            border: 1px solid #dee2e6;
            border-radius: 5px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .img-container {
            height: 200px;
            overflow: hidden;
            cursor: pointer;
        }
        
        .img-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s;
        }
        
        .img-container:hover img {
            transform: scale(1.05);
        }
        
        .card-body {
            padding: 15px;
        }
        
        .metrics-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
            background: #f8f9fa;
        }
        
        .metrics-badge.good {
            background-color: #d4edda;
            color: #155724;
        }
        
        .metrics-badge.medium {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .metrics-badge.poor {
            background-color: #f8d7da;
            color: #721c24;
        }
    `;
    document.head.appendChild(style);
})();
