# Nodata 疲劳驾驶检测系统

## 项目概述

一个使用计算机视觉和深度学习技术的综合疲劳驾驶检测系统。该系统实时监控驾驶员状态，检测疲劳迹象并及时提供警报，以防止事故发生。

## 功能特点

- 使用面部标志点实时检测驾驶员疲劳状态
- 眼睛闭合检测和眨眼频率分析
- 打哈欠检测
- 头部姿态监控
- 检测到疲劳时实时警报
- 基于Web的监控仪表板
- 基于Socket的实时通信

## 安装说明

### 前置条件

- Python 3.8 或更高版本
- 虚拟环境（推荐）

### 安装步骤

1. **克隆仓库**

2. **创建并激活虚拟环境**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 项目结构

```
Nodata_Fatigue_driving/
├── backend/
│   ├── app/
│   │   ├── main.py          # 主应用入口
│   │   ├── routes/          # API路由
│   │   ├── services/        # 业务逻辑
│   │   ├── models/          # 数据库模型
│   │   └── utils/           # 工具函数
│   └── config/              # 配置文件
├── frontend/                # 前端代码（如果有）
├── requirements.txt         # 依赖项
└── README.md                # 本文档
```

## 使用方法

### 启动服务器

```bash
# 从项目根目录
cd backend
python app/main.py
```

### 访问仪表板

打开浏览器并导航到 `http://localhost:5000` 访问监控仪表板。

## 依赖项

- **Web框架**：Flask, Flask-CORS, Flask-SocketIO
- **深度学习**：Ultralytics, PyTorch, TorchVision
- **计算机视觉**：OpenCV, MediaPipe
- **数据处理**：NumPy, Pandas, scikit-learn
- **实时通信**：python-socketio, python-engineio
- **图像/音频处理**：Pillow, pydub
- **数据库**：SQLAlchemy, PyMySQL
- **视频处理**：moviepy
- **其他**：requests, pyyaml, python-dotenv

## 工作原理

1. **视频捕获**：系统从面向驾驶员的摄像头捕获视频
2. **面部检测**：使用MediaPipe检测面部标志点
3. **特征提取**：分析眼睛状态、嘴巴状态和头部姿态
4. **疲劳评估**：使用深度学习模型评估驾驶员疲劳程度
5. **警报系统**：当检测到疲劳时触发警报
6. **数据存储**：将检测结果存储在数据库中
7. **实时监控**：将数据实时发送到仪表板

## 配置

更新 `backend/config/` 目录中的配置文件以自定义：
- 摄像头设置
- 检测阈值
- 警报偏好
- 数据库连接

## 贡献

欢迎贡献！请随时提交Pull Request。

## 许可证

本项目采用MIT许可证 - 详情请参阅LICENSE文件。
