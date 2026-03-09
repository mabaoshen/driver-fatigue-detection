import pymysql
import json
from datetime import datetime
import hashlib

def hash_password(password: str) -> str:
    """简单的密码哈希函数，与主应用保持一致"""
    return hashlib.md5(password.encode()).hexdigest()

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'fuzhujiashi'
}

def init_fatigue_driving_db():
    """初始化疲劳驾驶检测系统所需的数据库表"""
    try:
        # 连接数据库
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("正在初始化疲劳驾驶检测系统数据库表...")
        
        # 创建系统日志表（如果不存在）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fatigue_driving_system_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type VARCHAR(255) NOT NULL,
            message TEXT,
            user_ip VARCHAR(45),
            additional_info TEXT
        )''')
        print("✓ fatigue_driving_system_logs 表创建成功")
        
        # 创建检测记录表（如果不存在）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fatigue_driving_detection_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            detection_type VARCHAR(50) NOT NULL,
            status VARCHAR(50) NOT NULL,
            details TEXT,
            user_ip VARCHAR(45)
        )''')
        print("✓ fatigue_driving_detection_records 表创建成功")
        
        # 检查并添加缺失的字段
        # 检查表结构中的字段
        cursor.execute("SHOW COLUMNS FROM fatigue_driving_detection_records LIKE 'details'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE fatigue_driving_detection_records ADD COLUMN details TEXT")
            print("✓ 已添加 details 字段到 fatigue_driving_detection_records 表")
        
        # 创建用户表（如果不存在）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fatigue_driving_users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(100) NOT NULL UNIQUE,
            role ENUM('user', 'admin') DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME,
            is_active BOOLEAN DEFAULT TRUE
        )''')
        print("✓ fatigue_driving_users 表创建成功")
        
        # 创建默认管理员账户（如果不存在）
        cursor.execute("SELECT COUNT(*) FROM fatigue_driving_users WHERE role = 'admin'")
        if cursor.fetchone()[0] == 0:
            # 创建默认管理员，使用哈希算法处理密码，与主应用保持一致
            admin_password = hash_password('admin123')  # 使用与main.py一致的哈希算法
            cursor.execute(
                "INSERT INTO fatigue_driving_users (username, password, email, role) VALUES (%s, %s, %s, %s)",
                ('admin', admin_password, 'admin@example.com', 'admin')
            )
            print("✓ 默认管理员账户已创建 (username: admin, password: admin123)")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n数据库初始化完成！所有表结构已准备就绪。")
        print("疲劳驾驶检测系统可以正常运行了。")
        
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("=== 疲劳驾驶检测系统数据库初始化工具 ===")
    print(f"连接数据库: {DB_CONFIG['host']}/{DB_CONFIG['database']}")
    print(f"用户名: {DB_CONFIG['user']}")
    print("-" * 50)
    
    init_fatigue_driving_db()