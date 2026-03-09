import pymysql
import hashlib

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'fuzhujiashi',
    'charset': 'utf8mb4'  # 显式指定编码，避免字符编码问题
}


def hash_password(password: str) -> str:
    """简单的密码哈希函数，与主应用保持一致"""
    return hashlib.md5(password.encode()).hexdigest()


def update_admin_password():
    """更新管理员密码为正确的哈希值（修正逻辑版）"""
    conn = None
    cursor = None
    try:
        # 1. 建立数据库连接（增加编码配置，确保兼容性）
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("正在更新管理员密码...")
        target_username = 'admin'
        new_password_hash = hash_password('admin123')

        # 2. 第一步：验证管理员账户是否存在（核心修正点）
        cursor.execute(
            "SELECT id, password FROM fatigue_driving_users WHERE username = %s",
            (target_username,)
        )
        admin_record = cursor.fetchone()

        if not admin_record:
            # 真正的“未找到账户”场景
            print("✗ 未找到管理员账户，请先运行init_db.py创建管理员账户。")
            return

        # 账户存在，提取当前密码哈希
        admin_id, current_password_hash = admin_record

        # 3. 第二步：判断密码是否需要更新
        if current_password_hash == new_password_hash:
            print(f"✓ 管理员账户存在（ID: {admin_id}），密码已是 admin123，无需更新。")
            return

        # 4. 第三步：执行密码更新
        cursor.execute(
            "UPDATE fatigue_driving_users SET password = %s WHERE username = %s",
            (new_password_hash, target_username)
        )
        affected_rows = cursor.rowcount
        conn.commit()

        if affected_rows > 0:
            print(f"✓ 管理员密码更新成功！")
            print(f"  用户名: {target_username}")
            print(f"  原始密码: admin123")
            print(f"  哈希密码: {new_password_hash}")
        else:
            print("✗ 密码更新失败：找到账户但未修改任何行（未知异常）。")

    except Exception as e:
        # 出错时回滚事务，避免数据异常
        if conn:
            conn.rollback()
        print(f"更新密码失败: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保游标和连接最终关闭（资源释放）
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == '__main__':
    print("=== 管理员密码更新工具（修正版） ===")
    print(f"连接数据库: {DB_CONFIG['host']}/{DB_CONFIG['database']}")
    print("目标：将admin用户密码更新为 admin123")
    print("-" * 50)

    update_admin_password()