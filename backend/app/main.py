# Flask backend for web-based fatigue detection
import os
import sys
import io
import csv
import cv2
import json
import time
import uuid
import hashlib
import numpy as np
import logging
import pymysql
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template, current_app, make_response, session, redirect
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from threading import Thread

# 添加项目根目录到Python路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

# Disable server-side voice in web mode by stubbing speak_with_interval before importing detector
import core.voice as voice
voice.speak_with_interval = lambda text, seconds: False  # no-op in web mode

from core.detector import FatigueDetector

log = logging.getLogger("backend")
# 设置日志级别为DEBUG以查看详细信息
log.setLevel(logging.DEBUG)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'fuzhujiashi'
}

# 获取数据库连接
def get_db_connection() -> Optional[pymysql.Connection]:
    """获取数据库连接"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        log.error(f"数据库连接失败: {str(e)}")
        return None

def hash_password(password: str) -> str:
    """简单的密码哈希函数"""
    # 注意：实际生产环境应该使用更强的密码哈希算法如bcrypt
    return hashlib.md5(password.encode()).hexdigest()

def verify_password(stored_password: str, provided_password: str) -> bool:
    """验证密码"""
    return stored_password == hash_password(provided_password)

def is_admin() -> bool:
    """检查当前用户是否为管理员"""
    return session.get('role') == 'admin'

def is_authenticated() -> bool:
    """检查用户是否已认证"""
    # 添加详细调试日志
    session_content = dict(session)  # 转换为字典以便记录
    log.debug(f"Session contents: {session_content}")
    log.debug(f"Session has user_id: {'user_id' in session}")
    return 'user_id' in session

# 记录系统日志
def log_system_event(event_type: str, message: str, user_ip: str = None, additional_info: Any = None) -> None:
    try:
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        timestamp = datetime.now()
        additional_info_json = json.dumps(additional_info) if additional_info else None
        
        cursor.execute(
            "INSERT INTO fatigue_driving_system_logs (timestamp, event_type, message, user_ip, additional_info) VALUES (%s, %s, %s, %s, %s)",
            (timestamp, event_type, message, user_ip, additional_info_json)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        log.error(f"记录日志失败: {str(e)}")

# 记录检测结果
def log_detection_record(detection_type: str, status: str, details: Dict[str, Any], user_ip: str = None) -> None:
    try:
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        timestamp = datetime.now()
        details_json = json.dumps(details)
        
        cursor.execute(
            "INSERT INTO fatigue_driving_detection_records (timestamp, detection_type, status, details, user_ip) VALUES (%s, %s, %s, %s, %s)",
            (timestamp, detection_type, status, details_json, user_ip)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        log.error(f"记录检测结果失败: {str(e)}")

# Flask应用配置
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fatigue-detection-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB文件上传限制
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# 使用before_request替代已废弃的before_first_request
# 设置一个标志来确保只在第一次请求时执行清理
first_request = True

@app.before_request
def before_request_handler():
    global first_request
    if first_request:
        session.clear()
        log.info("Initial session cleared on first request")
        first_request = False

# SocketIO配置
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化连接客户端集合
connected_clients = set()

# 全局变量 - 支持独立的模型配置
CAMERA_MODEL_PATH: Optional[str] = None
VIDEO_MODEL_PATH: Optional[str] = None
CAMERA_DETECTOR: Optional[FatigueDetector] = None
VIDEO_DETECTOR: Optional[FatigueDetector] = None
JOBS = {}

# 性能优化变量
FRAME_SKIP_COUNTER = 0
LAST_PROCESS_TIME = 0
MIN_PROCESS_INTERVAL = 0.1  # 最小处理间隔100ms (10 FPS)

# 项目根目录
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "video_results")

# 初始化模型 - 完全独立配置，不使用默认模型
def init_models():
    global CAMERA_MODEL_PATH, VIDEO_MODEL_PATH, CAMERA_DETECTOR, VIDEO_DETECTOR
    # 不加载任何默认模型，完全独立配置
    CAMERA_MODEL_PATH = None
    VIDEO_MODEL_PATH = None
    CAMERA_DETECTOR = None
    VIDEO_DETECTOR = None
    log.info("[STARTUP] Models not loaded. Camera and video detection require separate model configuration.")
    
    # 记录系统启动事件
    log_system_event('system_startup', '疲劳驾驶检测系统后端服务启动', 'localhost')

# 检查数据库连接
def check_database_connection():
    conn = get_db_connection()
    if conn:
        log.info("数据库连接成功")
        conn.close()
        return True
    else:
        log.error("数据库连接失败")
        return False

# 应用启动时初始化
init_models()
check_database_connection()

# 定义公开路由，这些路由不需要登录验证
PUBLIC_ROUTES = [
    '/login', 
    '/login.html', 
    '/register',
    '/register.html',
    '/api/auth/login', 
    '/api/auth/register'
]

# 登录页面需要的静态资源
LOGIN_STATIC_RESOURCES = [
    '/css/', '/js/', '/img/', '/images/', '/fonts/', '/icons/', '/assets/',
    '/favicon.ico'
]

# 全局请求前处理器，检查所有非公开路由是否需要登录
@app.before_request
def before_request():
    # 添加调试日志
    log.info(f"Request path: {request.path}, Authenticated: {is_authenticated()}")
    
    # 检查是否为公开路由
    if request.path in PUBLIC_ROUTES:
        log.info(f"Public route accessed: {request.path}")
        return
    
    # 对于登录页面需要的静态资源，允许访问
    if not is_authenticated():
        for resource in LOGIN_STATIC_RESOURCES:
            if request.path.startswith(resource):
                log.info(f"Login resource accessed: {request.path}")
                return
    
    # 所有其他请求都需要验证
    if not is_authenticated():
        log.warning(f"Unauthenticated access attempt to: {request.path}")
        # 对于API请求，返回401未授权
        if request.path.startswith('/api/'):
            return jsonify({"error": "未登录", "code": 401}), 401
        # 对于页面请求，重定向到登录页面
        return redirect('/login')

# 添加一个路由来清除所有session数据，用于调试和重置
@app.route('/clear_session')
def clear_session():
    session.clear()
    log.info("Session cleared completely")
    return redirect('/login')

# 静态文件服务
@app.route('/')
def index():
    # 检查用户是否已登录
    if not is_authenticated():
        log.warning("Unauthenticated access attempt to index page")
        return redirect('/login')
    # 已登录用户返回主页面
    return send_from_directory(FRONTEND_DIR, 'index.html')

# 登录页面路由
@app.route('/login')
def login_page():
    # 直接返回登录页面，不需要认证检查
    return send_from_directory(FRONTEND_DIR, 'login.html')

# 确保login.html也能直接访问
@app.route('/login.html')
def login_html_page():
    # 重定向到标准的登录路由
    return redirect('/login')

# 注册页面路由
@app.route('/register')
def register_page():
    # 直接返回注册页面，不需要认证检查
    return send_from_directory(FRONTEND_DIR, 'register.html')

# 确保register.html也能直接访问
@app.route('/register.html')
def register_html_page():
    # 重定向到标准的注册路由
    return redirect('/register')

@app.route('/<path:filename>')
def static_files(filename):
    # 对于非静态资源文件（如admin、dashboard等页面），需要验证用户是否已登录
    # 静态资源文件后缀列表
    static_extensions = ('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', 
                        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.ico', '.ttf', 
                        '.woff', '.woff2', '.eot')
    
    # 检查是否为需要认证的路径
    needs_authentication = False
    
    # 如果不是静态资源文件，并且不是登录相关路径
    if not any(filename.lower().endswith(ext) for ext in static_extensions):
        # 检查是否为admin或其他敏感路径
        sensitive_paths = ['admin', 'dashboard', 'profile', 'settings']
        if any(path in filename.lower().split('/') for path in sensitive_paths):
            needs_authentication = True
    
    # 如果需要认证但用户未登录，重定向到登录页面
    if needs_authentication and not is_authenticated():
        log.warning(f"Unauthorized access attempt to protected resource: {filename}")
        return redirect('/login')
    
    # 检查文件是否存在
    file_path = os.path.join(FRONTEND_DIR, filename)
    if not os.path.exists(file_path):
        # 对于常见的静态资源文件，如果不存在，返回一个空的响应而不是404
        if filename.lower().endswith(('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg')):
            # 记录缺少的文件，便于后续修复
            log.warning(f"Missing static file: {filename}")
            # 根据文件类型返回适当的空响应
            if filename.lower().endswith(('.css', '.js')):
                return "/* Missing file */", 200
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                return "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='1' height='1'></svg>", 200
        # 对于视频文件，继续返回404
        elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            return "Video not found", 404
    return send_from_directory(FRONTEND_DIR, filename)

# API路由
@app.route('/api/version')
def version():
    return jsonify({
        "name": "fatigue-backend", 
        "version": "0.1.0", 
        "camera_model": CAMERA_MODEL_PATH,
        "video_model": VIDEO_MODEL_PATH
    })

@app.route('/api/status')
def status():
    """检查后端服务状态"""
    return jsonify({
        "status": "running",
        "camera_model_loaded": CAMERA_MODEL_PATH is not None,
        "video_model_loaded": VIDEO_MODEL_PATH is not None,
        "camera_model_path": CAMERA_MODEL_PATH,
        "video_model_path": VIDEO_MODEL_PATH,
        "camera_detector_ready": CAMERA_DETECTOR is not None,
        "video_detector_ready": VIDEO_DETECTOR is not None,
        "performance": {
            "min_process_interval": MIN_PROCESS_INTERVAL,
            "skipped_frames": FRAME_SKIP_COUNTER,
            "last_process_time": LAST_PROCESS_TIME
        }
    })

# 用户认证相关API
@app.route('/api/auth/register', methods=['POST'])
def register():
    """用户注册接口"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        # 参数验证
        if not all([username, password, email]):
            return jsonify({"error": "缺少必要参数"}), 400
        
        if len(username) < 3 or len(username) > 50:
            return jsonify({"error": "用户名长度必须在3-50个字符之间"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "密码长度必须至少为6个字符"}), 400
        
        # 检查用户名和邮箱是否已存在
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "数据库连接失败"}), 500
        
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM fatigue_driving_users WHERE username = %s OR email = %s", (username, email))
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "用户名或邮箱已被使用"}), 400
        
        # 创建新用户
        hashed_password = hash_password(password)
        cursor.execute(
            "INSERT INTO fatigue_driving_users (username, password, email) VALUES (%s, %s, %s)",
            (username, hashed_password, email)
        )
        conn.commit()
        user_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        log_system_event('user_register', f'新用户注册: {username}', request.remote_addr)
        return jsonify({"ok": True, "user_id": user_id, "username": username}), 201
        
    except Exception as e:
        log.error(f"用户注册失败: {str(e)}")
        return jsonify({"error": f"注册失败: {str(e)}"}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """用户登录接口"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # 参数验证
        if not all([username, password]):
            return jsonify({"error": "缺少必要参数"}), 400
        
        # 验证用户
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "数据库连接失败"}), 500
        
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM fatigue_driving_users WHERE username = %s AND is_active = TRUE", (username,))
        user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not user or not verify_password(user['password'], password):
            log.warning(f"登录失败: 用户名或密码错误 - {username}")
            return jsonify({"error": "用户名或密码错误"}), 401
        
        # 更新最后登录时间
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE fatigue_driving_users SET last_login = CURRENT_TIMESTAMP WHERE id = %s", (user['id'],))
            conn.commit()
            cursor.close()
            conn.close()
        
        # 设置用户会话
        session.permanent = True
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session['email'] = user['email']
        
        log_system_event('user_login', f'用户登录成功: {username}', request.remote_addr, {"user_id": user['id'], "role": user['role']})
        return jsonify({
            "ok": True,
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "role": user['role'],
                "created_at": user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            }
        })
        
    except Exception as e:
        log.error(f"用户登录失败: {str(e)}")
        return jsonify({"error": f"登录失败: {str(e)}"}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """用户登出接口"""
    username = session.get('username')
    session.clear()
    if username:
        log_system_event('user_logout', f'用户登出: {username}', request.remote_addr)
    return jsonify({"ok": True, "message": "登出成功"})

@app.route('/api/auth/profile', methods=['GET'])
def get_profile():
    """获取当前用户信息"""
    if not is_authenticated():
        return jsonify({"error": "未登录"}), 401
    
    return jsonify({
        "ok": True,
        "user": {
            "id": session.get('user_id'),
            "username": session.get('username'),
            "email": session.get('email'),
            "role": session.get('role')
        }
    })

@app.route('/change_password', methods=['POST'])
def change_password():
    """修改密码功能"""
    if not is_authenticated():
        return jsonify({"error": "未登录", "success": False}), 401
    
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求数据格式错误", "success": False}), 400
        
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        # 参数验证
        if not current_password or not new_password:
            return jsonify({"error": "缺少必要参数", "success": False}), 400
        
        if len(new_password) < 6:
            return jsonify({"error": "新密码长度至少为6位", "success": False}), 400
        
        # 获取数据库连接
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "数据库连接失败", "success": False}), 500
        
        cursor = conn.cursor()
        user_id = session.get('user_id')
        
        # 获取当前密码并验证
        cursor.execute("SELECT password FROM fatigue_driving_users WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            return jsonify({"error": "用户不存在", "success": False}), 404
        
        stored_password = result[0]
        if not verify_password(stored_password, current_password):
            cursor.close()
            conn.close()
            return jsonify({"message": "当前密码错误", "success": False}), 400
        
        # 更新密码
        new_hashed_password = hash_password(new_password)
        cursor.execute(
            "UPDATE fatigue_driving_users SET password = %s WHERE id = %s",
            (new_hashed_password, user_id)
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        # 记录系统日志
        username = session.get('username')
        log_system_event('password_change', f'用户修改密码: {username}', request.remote_addr)
        
        return jsonify({"success": True, "message": "密码修改成功"})
        
    except Exception as e:
        log.error(f"修改密码失败: {str(e)}")
        return jsonify({"error": f"修改密码失败: {str(e)}", "success": False}), 500

@app.route('/api/model/select', methods=['POST'])
def select_model():
    """选择已存在于服务器磁盘的模型路径。"""
    global CAMERA_MODEL_PATH, VIDEO_MODEL_PATH, CAMERA_DETECTOR, VIDEO_DETECTOR
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({"error": "缺少path参数"}), 400
    
    path = data['path']
    model_type = data.get('type')  # 必须指定 'camera' 或 'video'
    
    if not model_type or model_type not in ['camera', 'video']:
        return jsonify({"error": "必须指定模型类型: 'camera' 或 'video'"}), 400
    
    if not os.path.exists(path):
        return jsonify({"error": "模型路径不存在"}), 400
    
    try:
        if model_type == 'camera':
            CAMERA_MODEL_PATH = path
            CAMERA_DETECTOR = FatigueDetector(model_path=path, show_text=True)
            CAMERA_DETECTOR.model_path = path  # 确保model_path被设置
            log.info(f"Camera model loaded: {path}")
            log_system_event('model_select', f'选择摄像头模型: {path}', request.remote_addr)
            
        elif model_type == 'video':
            VIDEO_MODEL_PATH = path
            VIDEO_DETECTOR = FatigueDetector(model_path=path, show_text=True)
            VIDEO_DETECTOR.model_path = path  # 确保model_path被设置
            log.info(f"Video model loaded: {path}")
            log_system_event('model_select', f'选择视频模型: {path}', request.remote_addr)
            
        # 广播模型加载成功事件给所有连接的客户端
        socketio.emit('model_loaded', {
            "type": model_type,
            "path": path
        })
            
        return jsonify({
            "ok": True, 
            "camera_model": CAMERA_MODEL_PATH,
            "video_model": VIDEO_MODEL_PATH,
            "type": model_type
        })
    except FileNotFoundError as e:
        log_system_event('model_select_error', f'模型选择失败 - 文件不存在: {str(e)}', request.remote_addr)
        return jsonify({"error": f"模型文件不存在: {str(e)}"}), 404
    except RuntimeError as e:
        log_system_event('model_select_error', f'模型选择失败 - 加载错误: {str(e)}', request.remote_addr)
        return jsonify({"error": f"模型加载失败: {str(e)}。请注意：系统不使用默认模型，请确保选择的是有效的yolov8模型文件。"}), 500
    except Exception as e:
        log_system_event('model_select_error', f'模型选择失败: {str(e)}', request.remote_addr)
        return jsonify({"error": f"模型加载失败: {str(e)}。请注意：系统不使用默认模型，请确保选择的是有效的yolov8模型文件。"}), 500

@app.route('/api/model/upload', methods=['POST'])
def upload_model():
    """上传模型文件（.pt）。存入 models/ 目录并加载。"""
    global CAMERA_MODEL_PATH, VIDEO_MODEL_PATH, CAMERA_DETECTOR, VIDEO_DETECTOR
    
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    model_type = request.form.get('type')  # 必须指定 'camera' 或 'video'
    
    if not model_type or model_type not in ['camera', 'video']:
        return jsonify({"error": "必须指定模型类型: 'camera' 或 'video'"}), 400
    
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
    
    if not file.filename.endswith(".pt"):
        return jsonify({"error": "请上传 .pt 模型文件"}), 400
    
    save_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, secure_filename(file.filename))
    
    file.save(save_path)
    
    try:
        # 直接创建检测器实例并设置模型路径
        if model_type == 'camera':
            CAMERA_MODEL_PATH = save_path
            CAMERA_DETECTOR = FatigueDetector(model_path=save_path, show_text=True)  # 确保赋值正确
            CAMERA_DETECTOR.model_path = save_path  # 确保model_path被设置
            log.info(f"摄像头模型上传并加载成功: {save_path}")
            log_system_event('model_upload', f'上传摄像头模型: {file.filename}', request.remote_addr)
            
        elif model_type == 'video':
            VIDEO_MODEL_PATH = save_path
            VIDEO_DETECTOR = FatigueDetector(model_path=save_path, show_text=True)  # 确保赋值正确
            VIDEO_DETECTOR.model_path = save_path  # 确保model_path被设置
            log.info(f"视频模型上传并加载成功: {save_path}")
            log_system_event('model_upload', f'上传视频模型: {file.filename}', request.remote_addr)
        
        # 广播模型加载成功事件给所有连接的客户端
        socketio.emit('model_loaded', {
            "type": model_type,
            "path": save_path
        })
            
        return jsonify({
            "ok": True, 
            "camera_model": CAMERA_MODEL_PATH,
            "video_model": VIDEO_MODEL_PATH,
            "type": model_type,
            "model_loaded": True
        })
    except FileNotFoundError as e:
        # 加载失败时删除文件
        if os.path.exists(save_path):
            os.remove(save_path)
        log_system_event('model_upload_error', f'模型上传失败 - 文件不存在: {str(e)}', request.remote_addr)
        return jsonify({"error": f"模型文件不存在: {str(e)}"}), 404
    except RuntimeError as e:
        # 加载失败时删除文件
        if os.path.exists(save_path):
            os.remove(save_path)
        log_system_event('model_upload_error', f'模型上传失败 - 加载错误: {str(e)}', request.remote_addr)
        return jsonify({"error": f"模型加载失败: {str(e)}。请注意：系统不使用默认模型，请确保上传的是有效的yolov8模型文件。"}), 500
    except Exception as e:
        # 加载失败时删除文件
        if os.path.exists(save_path):
            os.remove(save_path)
        log.error(f"模型上传失败: {str(e)}")
        log_system_event('model_upload_error', f'模型上传失败: {str(e)}', request.remote_addr)
        return jsonify({"error": f"模型加载失败: {str(e)}。请注意：系统不使用默认模型，请确保上传的是有效的yolov8模型文件。"}), 500

@app.route('/api/performance/config', methods=['POST'])
def configure_performance():
    """配置性能参数"""
    global MIN_PROCESS_INTERVAL
    data = request.get_json()
    
    if 'min_interval' in data:
        interval = float(data['min_interval'])
        if 0.05 <= interval <= 1.0:  # 50ms到1000ms之间
            MIN_PROCESS_INTERVAL = interval
            log.info(f"Performance interval updated to {interval}s")
        else:
            return jsonify({"error": "间隔必须在0.05-1.0秒之间"}), 400
    
    return jsonify({
        "ok": True,
        "min_process_interval": MIN_PROCESS_INTERVAL,
        "max_fps": 1.0 / MIN_PROCESS_INTERVAL
    })

# 视频处理
@app.route('/api/video/upload_process', methods=['POST'])
def upload_process_video():
    if VIDEO_DETECTOR is None:
        return jsonify({"error": "视频检测模型未加载"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
    
    if not file.filename.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        return jsonify({"error": "请上传视频文件"}), 400
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    in_path = os.path.join(UPLOAD_DIR, secure_filename(file.filename))
    file.save(in_path)
    
    job_id = str(int(time.time()*1000))
    JOBS[job_id] = {"status": "queued", "progress": 0, "in": in_path, "out": None, "error": None}
    log.info(f"Job created: {job_id}. Current JOBS: {list(JOBS.keys())}")
    log_system_event('video_process_start', f'开始处理视频: {file.filename}', request.remote_addr, {"job_id": job_id})
    
    def worker(job_id, in_path, user_ip):
        try:
            JOBS[job_id]["status"] = "running"
            cap = cv2.VideoCapture(in_path)
            if not cap.isOpened():
                raise RuntimeError("无法打开视频")
            
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            
            out_path = os.path.join(OUTPUT_DIR, f"output_{os.path.basename(in_path)}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            
            # 用于统计视频中疲劳状态的变量
            total_fatigue_frames = 0
            total_frames = 0
            max_yawn_count = 0
            max_blink_count = 0
            
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_det, stat = VIDEO_DETECTOR.process_frame(frame)
                writer.write(frame_det)
                i += 1
                total_frames += 1
                
                # 统计疲劳状态
                if stat:
                    yawn_count = stat.get('yawn', 0)  # 使用正确的字段名
                    blink_count = stat.get('blink', 0)  # 使用正确的字段名
                    max_yawn_count = max(max_yawn_count, yawn_count)
                    max_blink_count = max(max_blink_count, blink_count)
                    
                    if yawn_count > 5 or blink_count > 10:
                        total_fatigue_frames += 1
                
                if total:
                    JOBS[job_id]["progress"] = int(i * 100 / total)
            
            writer.release()
            cap.release()
            
            # 计算视频总体状态
            fatigue_percentage = (total_fatigue_frames / total_frames * 100) if total_frames > 0 else 0
            overall_status = 'fatigue' if fatigue_percentage > 30 else 'normal'
            
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["out"] = out_path
            
            # 保存检测结果到job.result，供报告生成使用，保持字段名称与报告生成函数一致
            JOBS[job_id]["result"] = {
                'time': total_frames / fps if fps > 0 else 0,  # 保持与detector.py返回的字段名称一致
                'fatigue_index': fatigue_percentage / 100 if fatigue_percentage > 0 else 0,  # 转换为0-1范围
                'fatigue_score': int(fatigue_percentage * 1.5) if fatigue_percentage > 0 else 0,
                'fatigue_level': 3 if fatigue_percentage > 70 else 2 if fatigue_percentage > 30 else 1,
                'yawn': max_yawn_count,  # 使用与detector.py一致的字段名
                'blink': max_blink_count,  # 使用与detector.py一致的字段名
                'long_closed': False,  # 默认值，根据实际检测结果可修改
                # 保留原始统计信息，便于后续扩展使用
                'total_frames': total_frames,
                'fatigue_frames': total_fatigue_frames,
                'fatigue_percentage': fatigue_percentage,
                'is_fatigue': overall_status == 'fatigue'
            }
            
            # 记录系统事件
            log_system_event('video_process_complete', f'视频处理完成: {os.path.basename(in_path)}', user_ip, 
                           {"job_id": job_id, "fatigue_percentage": fatigue_percentage})
            
            # 记录视频检测结果到数据库
            log_detection_record(
                detection_type='video_file',
                status=overall_status,
                details={
                    'filename': os.path.basename(in_path),
                    'total_frames': total_frames,
                    'fatigue_frames': total_fatigue_frames,
                    'fatigue_percentage': fatigue_percentage,
                    'max_yawn_count': max_yawn_count,
                    'max_blink_count': max_blink_count,
                    'duration_seconds': total_frames / fps if fps > 0 else 0
                },
                user_ip=user_ip
            )
            
            # 移除重复的系统事件记录
            # log_system_event('video_process_complete', f'视频处理完成: {os.path.basename(in_path)}', user_ip, 
            #                {"job_id": job_id, "fatigue_percentage": fatigue_percentage})
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            log_system_event('video_process_error', f'视频处理失败: {str(e)}', user_ip, {"job_id": job_id})
    
    # 保存当前请求的用户IP，然后在线程中使用
    user_ip = request.remote_addr
    Thread(target=worker, args=(job_id, in_path, user_ip), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route('/api/video/status')
def video_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "缺少job_id参数"}), 400
    
    job = JOBS.get(job_id)
    if not job:
        log.error(f"Job not found: {job_id}. Current JOBS: {list(JOBS.keys())}")
        return jsonify({"error": "job not found"}), 404
    
    return jsonify(job)

@app.route('/api/video/download')
def video_download():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "缺少job_id参数"}), 400
    
    job = JOBS.get(job_id)
    if not job:
        log.error(f"Job not found: {job_id}. Current JOBS: {list(JOBS.keys())}")
        return jsonify({"error": "任务不存在"}), 404
    
    if job.get("status") != "done":
        return jsonify({"error": f"任务未完成，当前状态: {job.get('status')}"}), 400
    
    output_path = job.get("out")
    if not output_path:
        return jsonify({"error": "输出文件路径不存在"}), 400
    
    # 确保文件存在
    if not os.path.exists(output_path):
        log.error(f"Output file not found: {output_path}")
        return jsonify({"error": "处理后的视频文件不存在"}), 404
    
    # 获取文件名，使用原始文件名加上处理标记
    if job.get("in"):
        original_filename = os.path.basename(job["in"])
        name_without_ext, ext = os.path.splitext(original_filename)
        download_filename = f"processed_{name_without_ext}{ext}"
    else:
        download_filename = f"processed_video{os.path.splitext(output_path)[1]}"
    
    try:
        return send_file(output_path, as_attachment=True, download_name=download_filename)
    except Exception as e:
        log.error(f"Video download error: {str(e)}")
        return jsonify({"error": f"文件下载失败: {str(e)}"}), 500

@app.route('/api/generate_report/<job_id>', methods=['GET', 'OPTIONS'])
def generate_report(job_id):
    """生成并返回检测报告（CSV格式）"""
    # 处理预检请求
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.set('Access-Control-Allow-Origin', '*')
        response.headers.set('Access-Control-Allow-Methods', 'GET, OPTIONS')
        response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        return response
    
    try:
        # 添加详细日志记录请求
        log.info(f"收到报告生成请求，job_id: {job_id}, 客户端IP: {request.remote_addr}")
        
        # 检查任务是否存在
        job = JOBS.get(job_id)
        if not job:
            log.error(f"任务不存在: {job_id}。现有任务: {list(JOBS.keys())}")
            return jsonify({"error": "任务不存在，请确认任务ID是否正确"}), 404
        
        # 检查任务状态
        if job.get("status") != "done":
            log.warning(f"任务未完成，无法生成报告。job_id: {job_id}, 状态: {job.get('status')}")
            return jsonify({"error": f"任务未完成，当前状态: {job.get('status')}"}), 400
        
        # 获取基本任务信息
        import os
        in_path = job.get("in")
        filename = os.path.basename(in_path) if in_path else "未知视频"
        
        # 获取任务结果数据
        result_data = job.get('result', {})
        
        # 创建CSV报告
        import csv
        from io import StringIO
        from datetime import datetime
        
        # 使用StringIO创建内存中的CSV文件
        csv_buffer = StringIO()
        
        # 使用utf-8-sig编码确保中文正确显示
        csv_writer = csv.writer(csv_buffer, dialect='excel')
        
        # 写入报告头部
        csv_writer.writerow(['疲劳驾驶检测报告', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 写入生成时间（使用文本格式防止Excel自动转换）
        csv_writer.writerow(['生成时间', f'"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 写入视频文件信息
        csv_writer.writerow(['视频文件', filename, '', '', '', '', '', '', '', '', '', '', ''])
        
        # 写入任务ID
        csv_writer.writerow(['任务ID', job_id, '', '', '', '', '', '', '', '', '', '', ''])
        
        # 空行
        csv_writer.writerow(['', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 基本信息部分
        csv_writer.writerow(['基本信息:', '', '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['检测状态:', '已完成', '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['输入文件路径:', in_path or '未知', '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['输出文件路径:', job.get('out', '未知'), '', '', '', '', '', '', '', '', '', '', ''])
        
        # 空行
        csv_writer.writerow(['', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 检测统计信息部分
        csv_writer.writerow(['检测统计信息:', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 初始化默认值
        detection_time = "未知"
        fatigue_index = "未知"
        fatigue_percentage = "0.00%"
        fatigue_score = "0"
        
        # 安全获取数据
        if isinstance(result_data, dict):
            time_value = result_data.get('time')
            if time_value is not None:
                try:
                    detection_time = f"{float(time_value)}秒"
                except (ValueError, TypeError):
                    detection_time = "未知"
            
            index_value = result_data.get('fatigue_index')
            if index_value is not None:
                try:
                    index_value = float(index_value)
                    fatigue_index = str(index_value)
                    fatigue_percentage = f"{round(index_value * 100, 2):.2f}%"
                except (ValueError, TypeError):
                    fatigue_index = "未知"
            
            # 获取疲劳分数
            score_value = result_data.get('fatigue_score')
            if score_value is not None:
                try:
                    fatigue_score = str(float(score_value))
                except (ValueError, TypeError):
                    fatigue_score = "未知"
        
        csv_writer.writerow(['检测时长:', detection_time, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['疲劳指数:', fatigue_index, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['疲劳百分比:', fatigue_percentage, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['疲劳分数:', fatigue_score, '', '', '', '', '', '', '', '', '', '', ''])
        
        # 空行
        csv_writer.writerow(['', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 疲劳检测结果部分
        csv_writer.writerow(['疲劳检测结果:', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 初始化默认值
        is_fatigue = 0
        fatigue_level = 0
        fatigue_level_text = "清醒"
        yawn_count = "未知"
        blink_count = "未知"
        long_closed_detected = "否"
        
        # 安全获取并处理疲劳数据
        if isinstance(result_data, dict):
            # 疲劳等级映射
            fatigue_level_map = {0: '清醒', 1: '轻度疲劳', 2: '中度疲劳', 3: '严重疲劳'}
            
            # 获取疲劳等级
            level_value = result_data.get('fatigue_level', 0)
            try:
                fatigue_level = int(level_value)
                is_fatigue = 1 if fatigue_level > 0 else 0
                fatigue_level_text = fatigue_level_map.get(fatigue_level, '未知')
            except (ValueError, TypeError):
                pass
            
            # 获取打哈欠次数
            yawn_value = result_data.get('yawn')
            if yawn_value is not None:
                yawn_count = str(yawn_value)
            else:
                yawn_count = "未知"
            
            # 获取眨眼次数
            blink_value = result_data.get('blink')
            if blink_value is not None:
                blink_count = str(blink_value)
            else:
                blink_count = "未知"
            
            # 获取长时间闭眼检测结果
            long_closed_value = result_data.get('long_closed')
            if long_closed_value is not None:
                long_closed_detected = "是" if bool(long_closed_value) else "否"
        
        # 确保所有字段名和值正确对齐
        csv_writer.writerow(['是否检测到:', is_fatigue, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['疲劳等级:', fatigue_level, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['疲劳等级描述:', fatigue_level_text, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['哈欠次数:', yawn_count, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['眨眼次数:', blink_count, '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['长时间闭眼:', long_closed_detected, '', '', '', '', '', '', '', '', '', '', ''])
        
        # 空行
        csv_writer.writerow(['', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 注意事项部分
        csv_writer.writerow(['注意事项:', '', '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['1. 此报告基于内存中的检测结果生成', '', '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['2. 系统不使用数据库存储检测记录', '', '', '', '', '', '', '', '', '', '', '', ''])
        csv_writer.writerow(['3. 报告仅包含当前会话中的检测信息', '', '', '', '', '', '', '', '', '', '', '', ''])
        
        # 获取CSV内容并重新编码为utf-8-sig（确保Excel正确显示中文）
        csv_content = csv_buffer.getvalue()
        csv_buffer.close()
        
        log.info(f"CSV报告生成完成，job_id: {job_id}")
        
        # 设置文件名
        download_filename = f"疲劳驾驶检测报告_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 重新创建BytesIO对象用于每次下载
        from io import BytesIO
        byte_buffer = BytesIO()
        byte_buffer.write(csv_content.encode('utf-8-sig'))
        byte_buffer.seek(0)
        
        # 使用send_file确保正确处理文件下载
        return send_file(
            byte_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name=download_filename
        )
    except Exception as e:
        log.error(f"报告生成错误 (job_id: {job_id}): {str(e)}", exc_info=True)
        return jsonify({"error": f"报告生成失败: {str(e)}"}), 500

# SocketIO事件处理
@socketio.on('connect')
def handle_connect():
    log.info("客户端连接")
    log_system_event('client_connect', '客户端连接成功', request.remote_addr)
    # 添加当前客户端到已连接客户端集合
    connected_clients.add(request.sid)
    
    # 获取当前模型加载状态
    camera_model_loaded = CAMERA_DETECTOR is not None
    camera_model_path = CAMERA_MODEL_PATH
    video_model_loaded = VIDEO_DETECTOR is not None
    video_model_path = VIDEO_MODEL_PATH
    
    # 发送连接成功消息，并附带模型状态信息
    model_status = {
        'camera_model_loaded': camera_model_loaded,
        'camera_model_path': camera_model_path,
        'video_model_loaded': video_model_loaded,
        'video_model_path': video_model_path
    }
    
    # 直接将模型状态发送给客户端，作为连接响应
    emit('connected', {'message': '连接成功', **model_status})
    
    # 如果模型已加载，主动通知客户端
    if camera_model_loaded:
        emit('model_loaded', {'type': 'camera', 'path': camera_model_path})
    if video_model_loaded:
        emit('model_loaded', {'type': 'video', 'path': video_model_path})
    
    log.info(f"发送模型状态: 摄像头={camera_model_loaded}, 视频={video_model_loaded}")

@socketio.on('frame')
def handle_frame(data):
    global FRAME_SKIP_COUNTER, LAST_PROCESS_TIME
    
    try:
        # 性能优化：帧率控制
        current_time = time.time()
        if current_time - LAST_PROCESS_TIME < MIN_PROCESS_INTERVAL:
            FRAME_SKIP_COUNTER += 1
            return  # 跳过此帧
        
        # 解码图像数据
        nparr = np.frombuffer(data['image'], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            emit('error', {'message': '图像解码失败'})
            return
        
        # 获取检测类型，默认为摄像头
        detection_type = data.get('type', 'camera')
        
        # 根据检测类型选择相应的检测器
        if detection_type == 'camera':
            if CAMERA_DETECTOR is None or not hasattr(CAMERA_DETECTOR, 'model') or CAMERA_DETECTOR.model is None:
                # 更详细的错误信息
                detector_status = "检测器存在但模型未加载" if CAMERA_DETECTOR else "检测器不存在"
                status_info = f"摄像头模型路径: {'已设置' if CAMERA_MODEL_PATH else '未设置'}"
                error_message = f'摄像头检测模型未加载，请先在页面选择或上传模型。{detector_status}，{status_info}。请注意：系统不使用默认模型，必须手动上传或选择有效的yolov8模型文件。'
                emit('error', {'message': error_message})
                log.warning(error_message)
                return
            detector = CAMERA_DETECTOR
        elif detection_type == 'video':
            if VIDEO_DETECTOR is None or not hasattr(VIDEO_DETECTOR, 'model') or VIDEO_DETECTOR.model is None:
                detector_status = "检测器存在但模型未加载" if VIDEO_DETECTOR else "检测器不存在"
                status_info = f"视频模型路径: {'已设置' if VIDEO_MODEL_PATH else '未设置'}"
                error_message = f'视频检测模型未加载，请先在页面选择或上传模型。{detector_status}，{status_info}。请注意：系统不使用默认模型，必须手动上传或选择有效的yolov8模型文件。'
                emit('error', {'message': error_message})
                log.warning(error_message)
                return
            detector = VIDEO_DETECTOR
        else:
            emit('error', {'message': f'未知的检测类型: {detection_type}'})
            return
        
        # 性能优化：降低图像分辨率
        height, width = img.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # 处理帧
        t0 = time.time()
        
        # 保存原始show_text值
        original_show_text = getattr(detector, 'show_text', False)
        # 为摄像头模式确保启用文字显示
        if detection_type == 'camera':
            detector.show_text = True
        # 视频预览模式（通过Socket.IO的frame事件）不显示文字
        elif detection_type == 'video':
            detector.show_text = False
        
        try:
            frame_det, stat = detector.process_frame(img)
        finally:
            # 恢复原始show_text值，确保不影响其他调用
            detector.show_text = original_show_text
            
        LAST_PROCESS_TIME = time.time()
        
        # 生成唯一的帧ID，用于前端匹配状态和图像
        frame_id = str(uuid.uuid4())
        
        # 准备结果，包含帧ID以便前端同步状态和图像
        payload = {
            "type": "result",
            "frame_id": frame_id,  # 添加帧ID
            "ts": int(time.time() * 1000),
            "latency_ms": int((time.time() - t0) * 1000),
            "result": stat,
            "fps_info": {
                "skipped_frames": FRAME_SKIP_COUNTER,
                "process_interval": MIN_PROCESS_INTERVAL
            }
        }
        # 添加调试日志
        print(f"发送到前端的状态: blink={stat.get('blink')}, yawn={stat.get('yawn')}, fatigue_level={stat.get('fatigue_level')}")
        
        # 性能优化：降低JPEG质量和分辨率
        height, width = frame_det.shape[:2]
        if width > 480:
            scale = 480 / width
            new_width = 480
            new_height = int(height * scale)
            frame_det = cv2.resize(frame_det, (new_width, new_height))
        
        # 编码处理后的图像（降低质量）
        ok, buf = cv2.imencode('.jpg', frame_det, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        
        # 优化发送顺序，确保状态和图像数据紧密发送
        # 1. 先发送JSON统计信息
        emit('result', payload)
        
        # 2. 立即发送对应的图像数据（如果编码成功）
        if ok:
            # 发送图像时也附带帧ID，便于前端匹配
            emit('frame_result', {"frame_id": frame_id, "image": buf.tobytes()})
        
        # 重置跳帧计数器
        FRAME_SKIP_COUNTER = 0
        
        # 异步记录检测结果到数据库，避免阻塞Socket.IO消息发送
        # 使用线程池执行数据库操作，提高响应速度
        # 添加时间间隔控制，减少数据库记录频率（默认5秒记录一次）
        if stat:
            # 在请求上下文中预先获取所需信息
            remote_addr = request.remote_addr
            sid = request.sid
            
            def log_to_db():
                try:
                    # 只创建应用上下文，不依赖请求上下文
                    with app.app_context():
                        # 检查是否到达记录间隔（从5秒增加到15秒以减少数据库写入频率）
                        current_time = time.time()
                        # 为每个客户端维护独立的最后记录时间
                        client_last_log_time = getattr(current_app, 'client_last_log_times', {})
                        # 使用预先获取的客户端信息
                        client_id = remote_addr or sid
                        
                        # 首次记录或达到记录间隔
                        if client_id not in client_last_log_time or current_time - client_last_log_time[client_id] >= 15:
                            # 使用正确的字段名 'blink' 和 'yawn'
                            status_db = 'fatigue' if (stat.get('yawn', 0) > 5 or stat.get('blink', 0) > 10) else 'normal'
                            log_detection_record(
                                detection_type=detection_type,
                                status=status_db,
                                details={
                                    'yawn_count': stat.get('yawn', 0),
                                    'blink_count': stat.get('blink', 0),
                                    'eye_ratio': stat.get('eye_ratio', 0),
                                    'mouth_ratio': stat.get('mouth_ratio', 0),
                                    'processing_time_ms': payload['latency_ms']
                                },
                                user_ip=remote_addr
                            )
                            # 更新最后记录时间
                            client_last_log_time[client_id] = current_time
                            current_app.client_last_log_times = client_last_log_time
                except Exception as db_e:
                    log.error(f"数据库记录失败: {str(db_e)}")
            
            # 异步执行数据库记录，不阻塞消息发送
            socketio.start_background_task(log_to_db)
            
    except Exception as e:
        log_system_event('detection_error', f'检测过程中出错: {str(e)}', request.remote_addr)
        emit('error', {'message': str(e)})

@socketio.on('disconnect')
def handle_disconnect():
    log.info("客户端断开连接")
    log_system_event('client_disconnect', '客户端断开连接', request.remote_addr)
    # 从已连接客户端集合中移除当前客户端
    if request.sid in connected_clients:
        connected_clients.remove(request.sid)

@app.route('/api/admin/dashboard')
def admin_dashboard_api():
    """管理员仪表板API，返回系统日志和检测记录（不依赖数据库）"""
    # 添加权限控制
    if not is_authenticated():
        return jsonify({"error": "请先登录"}), 401
    if not is_admin():
        return jsonify({"error": "权限不足，需要管理员权限"}), 403
        
    try:
        # 返回模拟数据，不再依赖数据库
        return jsonify({
            "logs": [
                {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "event_type": "system_startup",
                    "message": "系统启动成功",
                    "user_ip": "localhost"
                },
                {
                    "timestamp": (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
                    "event_type": "model_loaded",
                    "message": "模型加载完成",
                    "user_ip": "127.0.0.1"
                }
            ],
            "records": []  # 不返回检测记录，避免数据库依赖
        })
    except Exception as e:
        log.error(f"获取仪表板数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/admin')
def admin_dashboard():
    """管理员仪表板页面（不依赖数据库）"""
    # 添加权限控制
    if not is_authenticated():
        return redirect('/login.html')  # 重定向到登录页面
    if not is_admin():
        return render_template('error.html', message="权限不足，需要管理员权限"), 403
        
    try:
        # 获取系统统计信息（只使用内存中的数据）
        online_clients = len(connected_clients)
        today_detections = 0  # 不再从数据库获取
        fatigue_alerts = 0    # 不再从数据库获取
        
        # 系统运行时间（模拟）
        system_uptime = "24h 30m"
        
        # 传递数据到模板
        return render_template('dashboard.html', 
                             online_clients=online_clients,
                             today_detections=today_detections,
                             fatigue_alerts=fatigue_alerts,
                             system_uptime=system_uptime)
    except Exception as e:
        app.logger.error(f"管理员页面加载失败: {str(e)}")
        return render_template('dashboard.html', error=str(e))

@app.route('/api/admin/dashboard_data')
def get_dashboard_data():
    """获取仪表板实时数据，从数据库获取统计信息"""
    # 添加权限控制
    if not is_authenticated():
        return jsonify({"error": "请先登录"}), 401
    if not is_admin():
        return jsonify({"error": "权限不足，需要管理员权限"}), 403
        
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 获取今日检测次数
        cursor.execute("""
            SELECT COUNT(*) FROM fatigue_driving_detection_records 
            WHERE DATE(timestamp) = DATE(NOW())
        """)
        today_detections = cursor.fetchone()[0]
        
        # 获取昨日检测次数
        cursor.execute("""
            SELECT COUNT(*) FROM fatigue_driving_detection_records 
            WHERE DATE(timestamp) = DATE(NOW()) - INTERVAL 1 DAY
        """)
        yesterday_detections = cursor.fetchone()[0]
        
        # 计算检测次数的变化百分比
        if yesterday_detections > 0:
            detection_change_percent = round((today_detections - yesterday_detections) / yesterday_detections * 100)
        else:
            detection_change_percent = 100 if today_detections > 0 else 0
        
        # 获取今日疲劳驾驶警报数
        cursor.execute("""
            SELECT COUNT(*) FROM fatigue_driving_detection_records 
            WHERE DATE(timestamp) = DATE(NOW()) AND details LIKE '%fatigue%'
        """)
        fatigue_alerts = cursor.fetchone()[0]
        
        # 获取昨日疲劳驾驶警报数
        cursor.execute("""
            SELECT COUNT(*) FROM fatigue_driving_detection_records 
            WHERE DATE(timestamp) = DATE(NOW()) - INTERVAL 1 DAY AND details LIKE '%fatigue%'
        """)
        yesterday_alerts = cursor.fetchone()[0]
        
        # 计算警报数的变化百分比
        if yesterday_alerts > 0:
            alert_change_percent = round((fatigue_alerts - yesterday_alerts) / yesterday_alerts * 100)
        else:
            alert_change_percent = 100 if fatigue_alerts > 0 else 0
        
        cursor.close()
        conn.close()
        
        # 计算系统运行时间（简化计算，实际可能需要记录启动时间）
        system_uptime = '未知'
        
        data = {
            'today_detections': today_detections,
            'yesterday_detections': yesterday_detections,
            'detection_change_percent': detection_change_percent,
            'fatigue_alerts': fatigue_alerts,
            'yesterday_alerts': yesterday_alerts,
            'alert_change_percent': alert_change_percent,
            'system_uptime': system_uptime,
            'system_status': '正常运行',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"获取仪表板数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/detection_records')
def get_detection_records():
    """获取检测记录，支持分页和过滤"""
    try:
        # 获取分页参数
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 5))
        
        # 获取过滤参数
        search_term = request.args.get('search', '')
        record_type = request.args.get('type', '')
        
        # 参数验证
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 10
        
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 构建查询条件
        conditions = []
        params = []
        
        if search_term:
            # 搜索多个字段
            conditions.append("(id LIKE %s OR timestamp LIKE %s OR user_ip LIKE %s OR details LIKE %s)")
            search_param = f"%{search_term}%"
            params.extend([search_param, search_param, search_param, search_param])
        
        if record_type:
            conditions.append("detection_type = %s")
            params.append(record_type)
        
        # 构建WHERE子句
        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        
        # 获取总记录数（应用过滤条件）
        total_count_query = f"SELECT COUNT(*) FROM fatigue_driving_detection_records{where_clause}"
        cursor.execute(total_count_query, params)
        total_count = cursor.fetchone()[0]
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 获取分页数据（应用过滤条件）
        data_query = f"""
            SELECT 
                id, 
                timestamp, 
                detection_type, 
                status, 
                details, 
                user_ip 
            FROM fatigue_driving_detection_records 
            {where_clause}
            ORDER BY timestamp DESC 
            LIMIT %s OFFSET %s
        """
        
        # 添加分页参数
        pagination_params = params + [page_size, offset]
        cursor.execute(data_query, pagination_params)
        
        # 获取列名
        column_names = [desc[0] for desc in cursor.description]
        # 将结果转换为字典列表
        records = []
        for row in cursor.fetchall():
            record_dict = {column_names[i]: row[i] for i in range(len(column_names))}
            records.append(record_dict)
        
        cursor.close()
        conn.close()
        
        # 格式化记录数据以匹配前端期望的格式
        formatted_records = []
        for record in records:
            # 生成检测ID
            detection_id = f"DT-{record['timestamp'].strftime('%Y%m%d')}-{record['id']:04d}"
            
            # 解析details字段获取结果信息
            result = '未知'
            try:
                if record['details']:
                    details = json.loads(record['details'])
                    # 根据检测类型和details内容确定结果
                    if record['status'] == 'fatigue':
                        result = '疲劳驾驶'
                    elif record['status'] == 'normal':
                        result = '正常驾驶'
                    # 视频文件检测特有字段解析
                    elif 'fatigue_percentage' in details:
                        percentage = details['fatigue_percentage']
                        if percentage >= 70:
                            result = '重度疲劳'
                        elif percentage >= 40:
                            result = '中度疲劳'
                        elif percentage >= 10:
                            result = '轻度疲劳'
                        else:
                            result = '正常驾驶'
                    # 摄像头实时检测特有字段解析
                    elif 'yawn_count' in details and 'blink_count' in details:
                        yawn_count = details['yawn_count']
                        blink_count = details['blink_count']
                        if yawn_count > 5 or blink_count > 10:
                            result = '疲劳驾驶'
                        else:
                            result = '正常驾驶'
            except:
                result = record['details'] if record['details'] else '未知'
            
            formatted_records.append({
                'id': detection_id,
                'time': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'ip': record['user_ip'] or '-',
                'type': record['detection_type'],
                'status': record['status'],
                'result': result
            })
        
        # 返回分页数据和元数据
        return jsonify({
            'records': formatted_records,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': (total_count + page_size - 1) // page_size
            }
        })
    except Exception as e:
        app.logger.error(f"获取检测记录失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/detection_record/<int:record_id>')
def get_detection_record_detail(record_id):
    # 添加权限控制
    if not is_authenticated():
        return jsonify({"error": "请先登录"}), 401
    if not is_admin():
        return jsonify({"error": "权限不足，需要管理员权限"}), 403
    """获取检测记录详情"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM fatigue_driving_detection_records WHERE id = %s
        """, (record_id,))
        
        record = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not record:
            return jsonify({"error": "记录不存在"}), 404
        
        # 转换为字典格式
        column_names = [desc[0] for desc in cursor.description]
        record_dict = {column_names[i]: record[i] for i in range(len(column_names))}
        
        # 解析details字段
        if record_dict['details']:
            try:
                record_dict['details'] = json.loads(record_dict['details'])
            except:
                pass
        
        # 格式化时间戳
        if isinstance(record_dict['timestamp'], datetime):
            record_dict['timestamp'] = record_dict['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify(record_dict)
    except Exception as e:
        app.logger.error(f"获取检测记录详情失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/download_detection_records')
def download_detection_records():
    """下载检测记录为CSV格式，支持单条记录或所有记录"""
    try:
        record_id = request.args.get('record_id')
        
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        if record_id:
            # 从格式化的ID中提取实际数据库ID (DT-20241201-0001 -> 1)
            actual_id = record_id.split('-').pop()
            # 获取单条记录
            cursor.execute("""
                SELECT * FROM fatigue_driving_detection_records WHERE id = %s
            """, (actual_id,))
        else:
            # 获取所有检测记录
            cursor.execute("""
                SELECT * FROM fatigue_driving_detection_records ORDER BY timestamp DESC
            """)
        
        records = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        cursor.close()
        conn.close()
        
        # 创建CSV内容
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        writer.writerow(column_names)
        
        # 写入数据
        for record in records:
            # 处理details字段，确保它是字符串格式
            record_list = list(record)
            if record_list[4] and isinstance(record_list[4], str) and record_list[4].strip().startswith('{'):
                try:
                    # 美化JSON格式
                    details_dict = json.loads(record_list[4])
                    record_list[4] = json.dumps(details_dict, ensure_ascii=False, indent=2)
                except:
                    pass
            writer.writerow(record_list)
        
        # 创建响应
        output.seek(0)
        filename = f"fatigue_detection_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if record_id:
            filename = f"fatigue_detection_record_{record_id}.csv"
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-type"] = "text/csv; charset=utf-8"
        
        return response
    except Exception as e:
        app.logger.error(f"下载检测记录失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/system_logs')
def get_system_logs():
    """获取系统日志，从数据库获取实际日志数据"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 获取最近的系统日志
        cursor.execute("""
            SELECT 
                timestamp, 
                event_type, 
                message, 
                user_ip 
            FROM fatigue_driving_system_logs 
            ORDER BY timestamp DESC 
            LIMIT 100
        """)
        
        # 获取列名
        column_names = [desc[0] for desc in cursor.description]
        # 将结果转换为字典列表
        logs = []
        for row in cursor.fetchall():
            log_dict = {column_names[i]: row[i] for i in range(len(column_names))}
            logs.append(log_dict)
        
        cursor.close()
        conn.close()
        
        # 格式化日志数据以匹配前端期望的格式
        formatted_logs = []
        for log in logs:
            # 根据事件类型确定日志级别
            level = 'INFO'
            if log['event_type'] in ['error', 'exception', 'model_select_error', 'model_upload_error']:
                level = 'ERROR'
            elif log['event_type'] in ['warning', 'model_load_delay']:
                level = 'WARNING'
            
            formatted_logs.append({
                'time': log['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'level': level,
                'message': log['message'] or '',
                'user_ip': log['user_ip']
            })
        
        return jsonify(formatted_logs)
    except Exception as e:
        app.logger.error(f"获取系统日志失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("疲劳驾驶检测系统后端服务启动中...")
    print("正在初始化数据库连接...")
    print("正在配置SocketIO服务...")
    print("前端页面请访问: http://localhost:5004/login")
    print("=========================================")
    socketio.run(app, host="0.0.0.0", port=5004, debug=False, use_reloader=False)
