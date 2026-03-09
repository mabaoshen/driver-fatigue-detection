# path: core/detector.py
import cv2
import time
import os
import math
import numpy as np
import logging
import threading

# 确保logs目录存在
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 配置基础日志记录器 - 修复编码问题
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'app.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
from collections import deque
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from core.face_mesh import FaceMeshHelper
from core.voice import speak_with_interval, shutdown as voice_shutdown  # 改为按间隔播报


class P:
    # 推理节奏与阈值（进一步降低阈值，确保检测到更多框）
    FPS = 12  # 提高帧率以获得更流畅的检测体验
    IMG_MAIN = 640  # 保持分辨率
    IMG_RETRY = 480  # 保持重试分辨率
    CONF_MAIN = 0.15  # 进一步降低置信度阈值，确保能检测到更多框
    CONF_RETRY = 0.12  # 进一步降低重试置信度阈值
    IOU = 0.35  # 更低的IOU阈值，允许更多重叠框

    # 统计周期
    PERIOD_SEC = 60  # 保持周期

    # 可视化
    C_EYE = (0, 165, 255)  # 橙色：眼睛 - 更明显的颜色
    C_MOUTH = (0, 255, 255)  # 亮黄：嘴巴 - 更明显的颜色
    C_INFO_OK = (0, 255, 0)
    C_INFO_WARN = (0, 0, 255)
    C_HINT = (255, 255, 0)
    THICK = 1  # 细线宽，更美观

    # 平滑/保持（优化响应速度）
    EMA_ALPHA = 0.55  # 保持EMA权重
    HOLD_TTL = 15  # 降低保持时间，提高响应速度
    ON_K = 1  # 最小确认帧数
    OFF_K = 2  # 确认帧数

    # 计数逻辑
    EYE_CLOSED_MIN = 1  # 闭眼最小帧数
    YAWN_OPEN_MIN_SEC = 1.2  # 哈欠持续时间阈值
    REFRACT_BLINK = 0.3  # 眨眼间隔限制
    YAWN_REFRACT = 2.5  # 哈欠间隔限制

    # 过滤极小框（允许更多小框）
    MIN_AREA_RATIO = 0.0003  # 极低的最小面积比例，确保几乎所有框都能通过过滤

    # 静态 EAR/MAR 双阈值（兜底）
    EAR_CLOSE_TH = 0.14  # 进一步降低闭眼阈值
    EAR_OPEN_TH = 0.16  # 进一步降低睁眼阈值，更容易检测到睁眼状态
    MAR_OPEN_TH = 0.55  # 更宽松的张嘴阈值
    MAR_CLOSE_TH = 0.35  # 更宽松的闭嘴阈值

    # EAR/MAR 时序平滑窗口
    SMOOTH_WIN = 5  # 减少平滑窗口，提高响应速度

    # 自适应标定（启动前几秒自动适配）
    CALIB_USE = False  # 临时禁用自动校准，使用固定阈值
    CALIB_SECONDS = 5.0  # 减少校准时间，更快进入正常检测状态
    CALIB_MIN_SAMPLES = 30  # 减少校准样本数
    CALIB_FACTOR_CLOSE = 0.65  # 进一步提高校准因子，使阈值更接近基准值
    CALIB_FACTOR_OPEN = 0.60  # 进一步降低校准因子，使动态校准后的睁眼阈值更低

    # 疲劳定义
    BLINK_FATIGUE_N = 12  # 眨眼次数阈值
    YAWN_FATIGUE_N = 1  # 哈欠次数阈值
    LONG_EYE_CLOSED_SEC = 3.0  # 增加长闭眼判定时间到3秒

    # 语音播报冷却（秒）
    VOICE_COOLDOWN_FATIGUE = 8.0
    VOICE_COOLDOWN_YAWN = 6.0
    VOICE_COOLDOWN_LONGCLOSE = 8.0
    # 全局播报最小间隔（秒）
    VOICE_GLOBAL_GAP = 2.5

    # YOLO 优先相关配置
    YOLO_FIRST = True  # 启用 YOLO 优先策略
    YOLO_UNCERT_TH = 0.35  # 更高的不确定阈值，使Face Mesh更频繁介入
    YOLO_DIFF_TH = 0.08  # 更小的分数差阈值，更容易触发Face Mesh

    # Face Mesh 触发策略 - 更频繁使用Face Mesh辅助
    FM_EVERY_N = 1  # 每帧都使用Face Mesh，确保最大辅助
    FM_ON_UNCERT = True  # 当 YOLO 不确定时，立即触发一次 Face Mesh
    FM_REFINE_POS = True  # 用 Face Mesh 的框精修位置
    FM_REFINE_ONLY_ON_UNCERT = False  # 始终使用FM优化，提高检测框数量
    FM_CACHE_EXPIRE = 8  # 延长缓存过期帧，让Face Mesh结果持续有效
    FM_RELIABILITY_THRESHOLD = 0.6  # Face Mesh结果可靠性阈值


# Face Mesh 精确索引
FM_LEFT_H = (33, 133)
FM_LEFT_V = [(159, 145), (160, 144)]
FM_RIGHT_H = (362, 263)
FM_RIGHT_V = [(386, 374), (385, 380)]
FM_MOUTH_W = (61, 291)
FM_MOUTH_V = (13, 14)


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1.0, (area_a + area_b - inter))


def nms_xyxy(cands: List[Tuple[Tuple[int, int, int, int], float]], iou_th=0.5):
    if not cands:
        return []
    idxs = sorted(range(len(cands)), key=lambda i: cands[i][1], reverse=True)
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(cands[i])
        idxs = [j for j in idxs if iou_xyxy(cands[i][0], cands[j][0]) < iou_th]
    return keep


def ema_box(prev_box: Optional[Tuple[int, int, int, int]],
            new_box: Tuple[int, int, int, int],
            alpha: float = P.EMA_ALPHA) -> Tuple[int, int, int, int]:
    if prev_box is None:
        return new_box
    px1, py1, px2, py2 = prev_box
    nx1, ny1, nx2, ny2 = new_box
    x1 = int(alpha * nx1 + (1 - alpha) * px1)
    y1 = int(alpha * ny1 + (1 - alpha) * py1)
    x2 = int(alpha * nx2 + (1 - alpha) * px2)
    y2 = int(alpha * ny2 + (1 - alpha) * py2)
    return (x1, y1, x2, y2)


class BoolSmoother:
    def __init__(self, on_k=P.ON_K, off_k=P.OFF_K, hold=P.HOLD_TTL):
        self.on_k = on_k
        self.off_k = off_k
        self.hold_n = hold
        self.on = 0
        self.off = 0
        self.state = False
        self.hold_left = 0

    def update(self, val: bool) -> bool:
        if val:
            self.on += 1
            self.off = 0
        else:
            self.off += 1
            self.on = 0
        if not self.state and self.on >= self.on_k:
            self.state = True
            self.hold_left = self.hold_n
        elif self.state and self.off >= self.off_k:
            if self.hold_left > 0:
                self.hold_left -= 1
            else:
                self.state = False
        return self.state


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear_from_all_pts(all_pts: np.ndarray, left: bool, is_wearing_glasses=False) -> Optional[float]:
    """计算EAR值，支持眼镜优化"""
    if all_pts is None or len(all_pts) < 468:
        return None
        
    # 使用FaceMesh.py中定义的扩展关键点索引
    from core.face_mesh import LEFT_EYE_IDX, RIGHT_EYE_IDX
    
    if left:
        hl, hr = FM_LEFT_H
        v_pairs = FM_LEFT_V
        # 眼镜检测时使用额外的关键点，避免受眼镜边框影响
        if is_wearing_glasses:
            # 使用核心眼内部关键点，避开眼镜边框
            v_pairs = [(159, 145), (160, 144), (158, 153), (147, 163)]
    else:
        hl, hr = FM_RIGHT_H
        v_pairs = FM_RIGHT_V
        if is_wearing_glasses:
            # 使用核心眼内部关键点，避开眼镜边框
            v_pairs = [(386, 374), (385, 380), (384, 398), (382, 373)]
    
    # 计算水平距离
    horiz = _dist(all_pts[hl], all_pts[hr]) + 1e-6
    
    # 计算垂直距离，增加异常值过滤
    vert = 0.0
    valid_pairs = 0
    dists = []
    
    for a, b in v_pairs:
        # 检查关键点是否有效（在合理范围内）
        if 0 <= a < len(all_pts) and 0 <= b < len(all_pts):
            dist = _dist(all_pts[a], all_pts[b])
            dists.append(dist)
    
    # 对距离进行中值滤波，去除眼镜边框导致的异常值
    if len(dists) > 0:
        # 中值滤波对异常值更鲁棒
        vert = np.median(dists)
        valid_pairs = 1
    
    if valid_pairs == 0:
        return None
    
    ear = vert / horiz
    
    # 眼镜优化：不再降低EAR值，而是增加它以便更容易检测到睁眼
    if is_wearing_glasses:
        # 对EAR值进行微调，根据经验值调整
        ear *= 1.30  # 显著增加EAR值，更容易检测到睁眼
        # 限制EAR值范围，避免异常值
        ear = max(0.1, min(0.6, ear))
    
    # 整体优化：进一步增加EAR值，使其更容易被检测为睁眼
    ear = min(0.6, ear * 1.35)
    
    return ear


def _mar_from_all_pts(all_pts: np.ndarray) -> Optional[float]:
    if all_pts is None or len(all_pts) < 468:
        return None
    wl, wr = FM_MOUTH_W
    vt, vb = FM_MOUTH_V
    horiz = _dist(all_pts[wl], all_pts[wr]) + 1e-6
    vert = _dist(all_pts[vt], all_pts[vb])
    return vert / horiz


# ---------- YOLO优先辅助 ----------
def _best_overlap_score(roi_box, cand_list):
    if roi_box is None or not cand_list:
        return 0.0
    best = 0.0
    for (box, sc) in cand_list:
        ov = iou_xyxy(roi_box, box)
        best = max(best, float(sc) * float(ov))
    return best


def yolo_pair_scores(roi_box, open_list, closed_list):
    open_s = _best_overlap_score(roi_box, open_list) if open_list else 0.0
    closed_s = _best_overlap_score(roi_box, closed_list) if closed_list else 0.0
    return open_s, closed_s


def yolo_is_uncertain(open_s, closed_s, th=P.YOLO_UNCERT_TH, diff_th=P.YOLO_DIFF_TH):
    m = max(open_s, closed_s)
    d = abs(open_s - closed_s)
    return (m < th) or (d < diff_th)


class FatigueDetector:
    def __init__(self, model_path: str, status_text=None, show_text=False):
        self.log = logging.getLogger(__name__)
        self.status_text = status_text
        self.running = True
        self.show_text = show_text  # 控制是否在画面上显示文字

        # 节流
        self.last_det_t = 0
        self.det_interval = 1.0 / P.FPS

        # 模型加载 - 不使用默认模型
        if not (model_path and os.path.exists(model_path)):
            raise FileNotFoundError("未找到模型，请选择有效的模型文件 (runs/train/**/best.pt)")
        mp = model_path

        out = {"model": None, "error": None}

        def _load(path, o):
            try:
                # Windows兼容性修复：禁用SIGALRM相关功能
                import os
                os.environ['YOLO_VERBOSE'] = 'False'  # 减少日志输出

                self.log.info(f"加载模型: {path}")
                o["model"] = YOLO(path)
            except Exception as e:
                o["error"] = str(e)

        t = threading.Thread(target=_load, args=(mp, out), daemon=True)
        t.start();
        t.join(timeout=20)
        if t.is_alive() or out["error"] is not None:
            raise RuntimeError(f"模型加载失败: {out['error'] or '超时'}")
        self.model = out["model"]

        names = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
        self.cls = {
            "closed_eye": self._find_id(names, "closed_eye"),
            "open_eye": self._find_id(names, "open_eye"),
            "closed_mouth": self._find_id(names, "closed_mouth"),
            "open_mouth": self._find_id(names, "open_mouth"),
        }
        self.log.info(f"模型类别: {names}")
        self.log.info(f"类别ID映射: {self.cls}")
        if self.status_text:
            self.status_text.AppendText(f"模型类别: {names}\n")

        self.prev = {"left_eye": {"box": None, "ttl": 0},
                     "right_eye": {"box": None, "ttl": 0},
                     "mouth": {"box": None, "ttl": 0}}
        
        # 存储历史框位置用于平滑处理
        self.history = {
            "left_eye": [],
            "right_eye": [],
            "mouth": []
        }
        # 历史框存储数量
        self.history_length = 3

        # 状态同步锁 - 确保多线程环境下状态更新的原子性
        self.state_lock = threading.RLock()
        
        # 初始化平滑器
        self.sm = {
            "left_closed": BoolSmoother(on_k=P.ON_K, off_k=P.OFF_K, hold=P.HOLD_TTL),
            "right_closed": BoolSmoother(on_k=P.ON_K, off_k=P.OFF_K, hold=P.HOLD_TTL),
            "mouth_open": BoolSmoother(on_k=P.ON_K, off_k=P.OFF_K, hold=P.HOLD_TTL),
        }
        
        # Face Mesh初始化
        self.fmh = None
        self.fmh_last_frames = 0
        self.fmh_last_uncert = False
        self.fmh_cache = {}
        self.fmh_cache_frames = {}
        
        # 眼镜检测相关状态
        self.is_wearing_glasses = False
        self.glasses_confidence = 0.0
        self.glasses_history = []
        self.glasses_detection_interval = 5  # 每隔几帧进行一次眼镜检测
        
        # 初始化其他组件
        self.font = self._load_font()
        self.last_flags = {"left_closed": False, "right_closed": False, "mouth_open": False}
        
        # 初始化校准相关属性
        self.calib_enabled = P.CALIB_USE
        self.calibrated = False
        self.calib_deadline = 0
        
        # 初始化检测参数，根据FPS计算帧数阈值
        self.yawn_open_min_frames = int(P.YAWN_OPEN_MIN_SEC * P.FPS)
        self.long_eye_closed_frames = int(P.LONG_EYE_CLOSED_SEC * P.FPS)

        # 重置所有状态
        self.reset_all_states()

        # Face Mesh初始化
        try:
            self.fm = FaceMeshHelper(max_faces=1)
            if not self.fm.ready and self.status_text:
                self.status_text.AppendText("MediaPipe未就绪，跳过关键点校正。\n")
        except Exception as e:
            self.fm = None
            self.log.warning(f"初始化FaceMesh失败: {e}")

    # ------------ 基础 ------------
    def stop(self):
        self.running = False
        try:
            if hasattr(self, "fm") and self.fm:
                self.fm.close()
        except Exception:
            pass
        try:
            voice_shutdown()
        except Exception:
            pass

    def reset_counters(self):
        """重置计数相关状态"""
        with self.state_lock:
            now = time.time()
            self.blink_count = 0
            self.yawn_count = 0
            self.fatigue_index = 0.0
            self.fatigue_score = 0.0  # 初始化疲劳分数
            self.time_start = now
            self.last_period_t = now
            self.last_blink_t = now - P.REFRACT_BLINK
            self.last_yawn_t = now - P.YAWN_REFRACT
            self.left_closed_frames = 0
            self.right_closed_frames = 0
            self.mouth_open_frames = 0
            
            # 状态追踪变量
            self.both_closed_start_time = None
            self.both_closed_duration = None
            self.closed_frames_count = 0
            self.blink_sequence = "open"
            self.yawn_sequence = "closed"
            self.yawn_open_start_time = None
            self.prev_eye_closed = False
            self.eye_closed_start_time = None
            self.prev_fatigue_status = False
            self.fatigue_start_time = None
            # 眨眼检测增强属性
            self.was_both_closed = False
            # 重置疲劳候选状态
            if hasattr(self, '_fatigue_candidate_start'):
                delattr(self, '_fatigue_candidate_start')
            
            # 重置疲劳等级相关属性
            self.prev_fatigue_level = 0
            self.fatigue_level = 0  # 0-清醒, 1-轻度疲劳, 2-中度疲劳, 3-严重疲劳
            
            # 重置长闭眼检测相关属性
            self.long_eye_close_count = 0
            self.long_eye_close_start_time = None
            self.last_long_close_time = 0
            # 新增长闭眼追踪属性
            self.current_long_close_frames = 0
            self.current_long_close_both_eyes = False
            
            # 重置疲劳评分更新时间
            self.last_score_update_time = 0
            
            # 语音播报状态
            self.voice_last_any = 0
            self.voice_last_fatigue = 0
            self.voice_last_yawn = 0
            self.voice_last_longclose = 0
            self.last_status_spoken = "清醒"
    
    def reset_all_states(self):
        """重置所有状态，包括计数和检测状态"""
        with self.state_lock:
            # 重置计数
            self.reset_counters()
            
            # 重置检测状态
            self.prev = {"left_eye": {"box": None, "ttl": 0},
                        "right_eye": {"box": None, "ttl": 0},
                        "mouth": {"box": None, "ttl": 0}}
            
            # 重置Face Mesh缓存
            self.last_all_pts = None
            self.last_fm_boxes = {"left": None, "right": None, "mouth": None}
            self.last_fm_frame = -999
            
            # 重置帧索引
            self.frame_idx = 0
            
            # 重置平滑缓存
            self.buf_ear_l = deque(maxlen=P.SMOOTH_WIN)
            self.buf_ear_r = deque(maxlen=P.SMOOTH_WIN)
            self.buf_mar = deque(maxlen=P.SMOOTH_WIN)
            
            # 重置自适应阈值
            self.calibrated = False
            self.calib_samples = deque(maxlen=256)
            self.ear_close_th = P.EAR_CLOSE_TH
            self.ear_open_th = P.EAR_OPEN_TH
            if hasattr(self, 'calib_enabled') and self.calib_enabled:
                self.calib_deadline = time.time() + P.CALIB_SECONDS

    def _find_id(self, names: List[str], target: str) -> Optional[int]:
        try:
            return next((i for i, n in enumerate(names) if str(n).lower() == target), None)
        except Exception:
            return None

    def _load_font(self):
        # 简化字体加载，不再依赖PIL
        return None

    def _draw_text(self, img, text, pos, color, font_scale=0.5, thickness=1):
        # 确保text是字符串类型
        text = str(text)
        try:
            # 检查是否包含中文
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                # 使用PIL来支持中文显示
                try:
                    import numpy as np
                    from PIL import Image, ImageDraw, ImageFont
                    
                    # 将OpenCV的BGR图像转换为PIL的RGB图像
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    
                    # 尝试加载中文字体
                    try:
                        # Windows系统默认中文字体路径
                        font_paths = [
                            "simhei.ttf",  # 当前目录
                            "C:\\Windows\\Fonts\\simhei.ttf",  # Windows字体目录
                            "C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
                            "C:\\Windows\\Fonts\\simsun.ttc"  # 宋体
                        ]
                        
                        font = None
                        for font_path in font_paths:
                            try:
                                font = ImageFont.truetype(font_path, size=int(font_scale * 20))
                                break
                            except:
                                continue
                        
                        # 如果找不到字体，使用默认字体
                        if font is None:
                            font = ImageFont.load_default()
                            
                    except Exception as font_err:
                        self.log.warning(f"加载字体失败: {font_err}")
                        font = ImageFont.load_default()
                    
                    # 绘制文本 - 添加描边效果，解决中文显示问题
                    # 先绘制文字描边
                    stroke_width = 2
                    for offset_x in range(-stroke_width, stroke_width + 1):
                        for offset_y in range(-stroke_width, stroke_width + 1):
                            if offset_x != 0 or offset_y != 0:
                                draw.text((pos[0] + offset_x, pos[1] + offset_y), text, 
                                         font=font, fill=(0, 0, 0))  # 黑色描边
                    # 再绘制文字主体
                    draw.text(pos, text, font=font, fill=tuple(reversed(color)))  # PIL的颜色是RGB
                    
                    # 将PIL图像转换回OpenCV图像
                    img_array = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    img[:] = img_array  # 原地修改原图
                    return img
                except Exception as pil_err:
                    self.log.warning(f"使用PIL绘制中文失败: {pil_err}")
                    # 如果PIL方法失败，回退到OpenCV方法
                    pass
            
            # 如果不包含中文或PIL方法失败，使用OpenCV默认方法
            return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        except Exception as e:
            self.log.warning(f"文字绘制失败: {e}")
            return img

    @staticmethod
    def _ensure_bgr(fr):
        if fr is None or fr.size == 0:
            return fr
        if len(fr.shape) == 2:
            return cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        if fr.shape[2] == 4:
            return cv2.cvtColor(fr, cv2.COLOR_BGRA2BGR)
        return fr[:, :, :3]

    # ------------ 推理 ------------
    def _infer(self, img, conf, imgsz):
        if not self.running or self.model is None:
            class _EmptyRes:
                boxes = type("B", (), {"xyxy": None, "cls": None, "conf": None})()

            return [_EmptyRes()]
        # 禁用NMS以确保保留所有检测框，使用agnostic_nms提高多类别检测效果
        return self.model(img, conf=conf, iou=P.IOU, imgsz=imgsz, verbose=False, nms=False, agnostic_nms=True)

    def _group_by_class(self, results, hw):
        h, w = hw
        min_area = P.MIN_AREA_RATIO * (h * w)
        data = {"open_eye": [], "closed_eye": [], "open_mouth": [], "closed_mouth": []}
        for r in results:
            if r.boxes is None or r.boxes.xyxy is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, cf in zip(boxes, cls, conf):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                if min_area > 0 and (x2 - x1) * (y2 - y1) < min_area:
                    continue
                if self.cls["open_eye"] is not None and c == self.cls["open_eye"]:
                    data["open_eye"].append(((x1, y1, x2, y2), float(cf)))
                elif self.cls["closed_eye"] is not None and c == self.cls["closed_eye"]:
                    data["closed_eye"].append(((x1, y1, x2, y2), float(cf)))
                elif self.cls["open_mouth"] is not None and c == self.cls["open_mouth"]:
                    data["open_mouth"].append(((x1, y1, x2, y2), float(cf)))
                elif self.cls["closed_mouth"] is not None and c == self.cls["closed_mouth"]:
                    data["closed_mouth"].append(((x1, y1, x2, y2), float(cf)))
        for k in data:
            data[k] = nms_xyxy(data[k], iou_th=0.5)
        return data

    def _validate_eye_box(self, box, img_w, img_h, side="left"):
        """验证一个框是否包含有效的眼睛区域"""
        if not box:
            return False, 0.0
        
        x1, y1, x2, y2 = box
        # 计算框的基本属性
        width = x2 - x1
        height = y2 - y1
        area = width * height
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        aspect_ratio = width / max(1.0, height)
        
        # 1. 最小/最大尺寸检查
        min_eye_size = min(img_w, img_h) * 0.03  # 略微提高最小眼睛尺寸要求，提高稳定性
        max_eye_size = min(img_w, img_h) * 0.20  # 略微降低最大眼睛尺寸限制，减少过大框
        if width < min_eye_size or height < min_eye_size or width > max_eye_size or height > max_eye_size:
            return False, 0.0
        
        # 2. 位置合理性检查
        # 眼睛应该在图像上半部分
        if cy > img_h * 0.55:  # 稍微收紧垂直位置限制，提高稳定性
            return False, 0.0
        
        # 3. 宽高比检查（眼睛通常是水平的矩形）
        min_ratio = 1.2  # 稍微收紧宽高比要求
        max_ratio = 3.0  # 稍微收紧宽高比要求
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            return False, 0.0
        
        # 4. 计算眼睛区域的合理性分数 - 放宽条件以更好地适应戴眼镜的情况
        # 计算预期眼睛位置
        face_center_y = img_h * 0.45  # 稍微下移预期位置
        face_center_x = img_w * 0.5
        expected_eye_x = face_center_x - img_w * 0.15 if side == "left" else face_center_x + img_w * 0.15  # 放宽水平预期范围
        
        # 位置合理性评分 - 更宽松的标准
        x_dist = abs(cx - expected_eye_x) / (img_w * 0.25)  # 扩大水平位置评分范围
        y_dist = abs(cy - face_center_y) / (img_h * 0.3)   # 扩大垂直位置评分范围
        pos_score = max(0, 1.0 - (x_dist + y_dist) / 2.0)
        
        # 宽高比评分 - 适应戴眼镜时的宽框
        aspect_score = 1.0 - abs(aspect_ratio - 2.2) / 3.0  # 更宽松的宽高比要求
        aspect_score = max(0, aspect_score)
        
        # 综合分数 - 降低位置权重，更注重宽高比
        validity_score = 0.6 * pos_score + 0.4 * aspect_score  # 降低位置评分权重，提高宽高比权重
        
        return True, validity_score
    
    def _clamp_box(self, box, img_w, img_h):
        """确保检测框不会越界"""
        if not box or len(box) != 4:
            return box
        
        x1, y1, x2, y2 = box
        
        # 确保坐标在有效范围内
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        x2 = max(x1 + 1, min(int(x2), img_w - 1))  # 确保x2 > x1
        y2 = max(y1 + 1, min(int(y2), img_h - 1))  # 确保y2 > y1
        
        return (x1, y1, x2, y2)
        
    def _smooth_box(self, box, box_type, img_w=None, img_h=None):
        """使用历史框位置对当前框进行平滑处理，确保稳定性"""
        if not box or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            # 如果当前框无效，返回上一个有效框
            if self.history[box_type]:
                return self.history[box_type][-1]
            return box
        
        # 存储当前框
        self.history[box_type].append(box)
        
        # 保持历史记录长度
        if len(self.history[box_type]) > self.history_length:
            self.history[box_type] = self.history[box_type][-self.history_length:]
        
        # 如果历史记录较少，直接返回当前框
        if len(self.history[box_type]) < 2:
            # 如果提供了图像尺寸，进行边界检查
            if img_w and img_h:
                return self._clamp_box(box, img_w, img_h)
            return box
        
        # 计算与上一帧的IOU，如果变化太大，可能是错误检测，降低权重
        prev_box = self.history[box_type][-2]
        current_iou = iou_xyxy(prev_box, box)
        
        # 使用加权平均进行平滑
        # 根据IOU动态调整权重，IOU越低，当前框权重越低
        recent_weight = 0.7 if current_iou > 0.5 else 0.5
        older_weight = (1.0 - recent_weight) / (len(self.history[box_type]) - 1)
        
        weights = [older_weight] * (len(self.history[box_type]) - 1) + [recent_weight]
        
        # 对每个坐标进行加权平均
        x1 = sum(w * b[0] for w, b in zip(weights, self.history[box_type]))
        y1 = sum(w * b[1] for w, b in zip(weights, self.history[box_type]))
        x2 = sum(w * b[2] for w, b in zip(weights, self.history[box_type]))
        y2 = sum(w * b[3] for w, b in zip(weights, self.history[box_type]))
        
        smoothed_box = (int(x1), int(y1), int(x2), int(y2))
        
        # 如果提供了图像尺寸，进行边界检查
        if img_w and img_h:
            return self._clamp_box(smoothed_box, img_w, img_h)
        
        return smoothed_box
        
    def _pick_eyes(self, eye_open, eye_closed, img_w, img_h):
        # 过滤无效的眼睛框，提高选择质量
        valid_eyes = []
        all_eye = [(b, c, "open") for b, c in eye_open] + [(b, c, "closed") for b, c in eye_closed]
        
        # 对所有检测框进行初步验证
        for b, c, st in all_eye:
            # 验证框的基本有效性
            if b[2] > b[0] and b[3] > b[1]:  # 确保框有效
                valid_eyes.append((b, c, st))
        
        # 如果没有有效眼睛框，直接返回
        if not valid_eyes:
            return None, None
        
        # 先按置信度排序，但在选择时考虑更多因素
        valid_eyes.sort(key=lambda x: x[1], reverse=True)
        
        # 确保返回两个眼睛框的策略优化
        # 即使只有一个框，也尝试基于位置复制或调整以创建第二个框
        if len(valid_eyes) <= 2:
            if len(valid_eyes) == 1:
                # 只有一个框时，不仅根据位置决定眼别，还尝试基于对称性创建另一个眼睛框
                bx, by, bw, bh = valid_eyes[0][0]
                cx = (bx + bw) / 2
                cy = (by + bh) / 2
                width = bw - bx
                height = bh - by
                
                # 确定主眼
                if cx < img_w * 0.5:
                    primary_eye = valid_eyes[0]  # 左眼
                    # 尝试创建右眼框（基于左眼对称位置）
                    right_cx = img_w - cx
                    right_box = (right_cx - width//2, cy - height//2, right_cx + width//2, cy + height//2)
                    secondary_eye = (right_box, valid_eyes[0][1] * 0.7, valid_eyes[0][2])  # 降低置信度
                    return primary_eye, secondary_eye
                else:
                    secondary_eye = valid_eyes[0]  # 右眼
                    # 尝试创建左眼框（基于右眼对称位置）
                    left_cx = img_w - cx
                    left_box = (left_cx - width//2, cy - height//2, left_cx + width//2, cy + height//2)
                    primary_eye = (left_box, valid_eyes[0][1] * 0.7, valid_eyes[0][2])  # 降低置信度
                    return primary_eye, secondary_eye
            else:
                # 两个框时，根据中心点x坐标决定左右眼
                cx1 = (valid_eyes[0][0][0] + valid_eyes[0][0][2]) / 2
                cx2 = (valid_eyes[1][0][0] + valid_eyes[1][0][2]) / 2
                if cx1 < cx2:
                    return valid_eyes[0], valid_eyes[1]
                else:
                    return valid_eyes[1], valid_eyes[0]
        
        def pick_for(side: str, banned: set):
            target_left = (side == "left")
            best_idx, best_score, best = -1, -1.0, None
            
            # 计算预期眼睛位置（基于面部对称性）
            face_center_y = img_h * 0.4  # 眼睛区域通常在图像垂直方向的40%位置
            face_center_x = img_w * 0.5  # 图像水平中心
            expected_eye_x = face_center_x - img_w * 0.15 if target_left else face_center_x + img_w * 0.15
            
            # 主要评分系统：考虑多个因素，确保选择的框真正包含眼睛
            for i, (b, c, st) in enumerate(valid_eyes):
                if i in banned:
                    continue
                
                x1, y1, x2, y2 = b
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                box_height = y2 - y1
                box_width = x2 - x1
                aspect_ratio = box_width / max(1.0, box_height)
                
                # 1. 验证框是否可能包含有效的眼睛
                is_valid, validity_score = self._validate_eye_box(b, img_w, img_h, side)
                if not is_valid:
                    continue  # 跳过无效的框
                
                # 2. 位置评分：基于预期眼睛位置（提高权重到40%）
                horizontal_dist = abs(cx - expected_eye_x)
                horizontal_score = max(0, 1.0 - horizontal_dist / (img_w * 0.15))  # 更宽松的水平位置容忍度
                
                # 垂直位置权重（眼睛应该在面部上半部分）
                vertical_dist = abs(cy - face_center_y)
                vertical_score = max(0, 1.0 - vertical_dist / (img_h * 0.25))  # 更宽松的垂直位置容忍度
                
                position_score = 0.6 * horizontal_score + 0.4 * vertical_score
                
                # 3. 大小和形状评分（提高权重到35%）
                expected_eye_height = img_h * 0.07  # 调整预期眼睛高度
                expected_eye_width = img_w * 0.14   # 调整预期眼睛宽度
                height_score = 1.0 - min(abs(box_height - expected_eye_height) / expected_eye_height, 1.0)
                width_score = 1.0 - min(abs(box_width - expected_eye_width) / expected_eye_width, 1.0)
                
                # 宽高比评分（眼睛通常是水平的矩形）
                target_ratio = 2.2  # 调整为更适合眼睛的宽高比
                aspect_score = max(0, 1.0 - abs(aspect_ratio - target_ratio) / target_ratio)
                
                shape_score = 0.3 * height_score + 0.3 * width_score + 0.4 * aspect_score
                
                # 4. 置信度评分（降低权重到15%，因为置信度不一定表示定位准确性）
                confidence_score = c
                
                # 5. 连贯性评分（10%）- 与前一帧的一致性
                coherence_score = 0.5  # 默认中等分数
                prev_box = self.prev[f"{side}_eye"]["box"]
                if prev_box is not None:
                    coherence_score = iou_xyxy(prev_box, b)
                
                # 综合评分，优化权重分布
                total_score = 0.40 * position_score + 0.35 * shape_score + 0.15 * confidence_score + 0.10 * coherence_score
                
                # 乘以验证分数，进一步强调框的有效性
                total_score *= validity_score
                
                if total_score > best_score:
                    best_idx, best_score, best = i, total_score, (b, c, st)
            
            # 如果找不到符合条件的框，但有足够的有效框，尝试放宽条件
            if best is None and len(valid_eyes) > len(banned):
                for i, (b, c, st) in enumerate(valid_eyes):
                    if i in banned:
                        continue
                    
                    # 仅基于位置和置信度进行简单评分
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                    expected_eye_x = face_center_x - img_w * 0.25 if target_left else face_center_x + img_w * 0.25  # 更大的容忍范围
                    dist = abs(cx - expected_eye_x)
                    position_score = max(0, 1.0 - dist / (img_w * 0.3))
                    
                    # 简化的综合评分
                    simple_score = 0.7 * position_score + 0.3 * c
                    
                    if simple_score > best_score:
                        best_idx, best_score, best = i, simple_score, (b, c, st)
            
            # 如果仍然找不到，尝试使用基于中心x坐标的简单判断（适用于剩下的框很少的情况）
            if best is None and len(valid_eyes) > len(banned):
                center_x = img_w * 0.5
                for i, (b, c, st) in enumerate(valid_eyes):
                    if i in banned:
                        continue
                    
                    cx = (b[0] + b[2]) / 2
                    # 简单地基于x坐标判断左右眼
                    if (target_left and cx < center_x) or (not target_left and cx > center_x):
                        best_idx, best_score, best = i, c, (b, c, st)
                        break
            
            if best_idx >= 0:
                banned.add(best_idx)
            return best

        banned = set()
        left = pick_for("left", banned)
        right = pick_for("right", banned)
        return left, right

    def _pick_mouth(self, mouth_open, mouth_closed):
        all_m = [(b, c, "open") for b, c in mouth_open] + [(b, c, "closed") for b, c in mouth_closed]
        if not all_m:
            return None
        all_m.sort(key=lambda x: x[1], reverse=True)
        return all_m[0]

    def detect_facial_states(self, frame):
        if not self.running or frame is None or frame.size == 0:
            return frame, self.last_flags.copy()

        now = time.time()
        if now - self.last_det_t < self.det_interval:
            return frame, self.last_flags.copy()
        self.last_det_t = now

        frame = self._ensure_bgr(frame)
        h, w = frame.shape[:2]
        self.frame_idx += 1

        # 1) YOLO 主/重试 - 总是使用更低的置信度以确保检测
        # 先使用较低分辨率和较低置信度进行初步检测
        results = self._infer(frame, P.CONF_RETRY, P.IMG_RETRY)  # 直接使用更低的阈值，确保能检测到目标
        data = self._group_by_class(results, (h, w))
        
        # 如果检测结果较少，尝试更高分辨率的检测以提高准确率
        det_count = sum(len(data[k]) for k in ["open_eye", "closed_eye", "open_mouth", "closed_mouth"])
        if det_count < 3:  # 如果检测到的面部特征少于3个
            self.log.debug(f"检测到面部特征较少 ({det_count}个)，尝试更高分辨率检测")
            high_res_results = self._infer(frame, max(P.CONF_RETRY + 0.05, 0.15), P.IMG_MAIN)  # 使用更高分辨率
            high_res_data = self._group_by_class(high_res_results, (h, w))
            
            # 合并两种分辨率的检测结果，优先选择高分辨率结果
            for k in ["open_eye", "closed_eye", "open_mouth", "closed_mouth"]:
                if high_res_data[k]:  # 如果高分辨率有结果，优先使用
                    data[k] = high_res_data[k]
        
        # 记录检测统计信息，帮助调试
        det_count = sum(len(data[k]) for k in ["open_eye", "closed_eye", "open_mouth", "closed_mouth"])
        if det_count == 0:
            self.log.debug(f"未检测到任何面部特征")
        else:
            self.log.debug(f"检测到 {det_count} 个面部特征")

        def _valid_box(b):
            if b is None:
                return False
            x1, y1, x2, y2 = b
            if x2 <= x1 or y2 <= y1:
                return False
            area = (x2 - x1) * (y2 - y1)
            return area >= (P.MIN_AREA_RATIO * (h * w))

        flags = {"left_closed": False, "right_closed": False, "mouth_open": False}

        # 2) 用 YOLO 选 ROI（左眼/右眼/嘴）
        le = re = None
        if data["open_eye"] or data["closed_eye"]:
            le, re = self._pick_eyes(data["open_eye"], data["closed_eye"], w, h)
        m = None
        if data["open_mouth"] or data["closed_mouth"]:
            m = self._pick_mouth(data["open_mouth"], data["closed_mouth"])

        # 获得当前 ROI 框（可能为空）
        left_eye_box = le[0] if le else None
        right_eye_box = re[0] if re else None
        mouth_box = m[0] if m else None

        # 3) YOLO 不确定性评估
        le_open_s, le_closed_s = yolo_pair_scores(left_eye_box, data["open_eye"], data["closed_eye"])
        re_open_s, re_closed_s = yolo_pair_scores(right_eye_box, data["open_eye"], data["closed_eye"])
        m_open_s, m_closed_s = yolo_pair_scores(mouth_box, data["open_mouth"], data["closed_mouth"])

        le_unc = yolo_is_uncertain(le_open_s, le_closed_s)
        re_unc = yolo_is_uncertain(re_open_s, re_closed_s)
        m_unc = yolo_is_uncertain(m_open_s, m_closed_s)

        # 4) 增强的Face Mesh触发策略 + 周期维护 + 优先级
        run_fm = False
        # 更积极地触发Face Mesh：当YOLO不确定或检测结果不足时
        if getattr(P, "FM_ON_UNCERT", True) and (le_unc or re_unc or m_unc):
            run_fm = True
        if (self.frame_idx - self.last_fm_frame) >= getattr(P, "FM_EVERY_N", 3):
            run_fm = True
        # 当YOLO检测结果不足时，强制使用Face Mesh
        if det_count < 2:  # 如果检测到的面部特征少于2个
            run_fm = True

        # 保存之前的FM结果作为备份
        prev_fm_boxes = self.last_fm_boxes.copy() if self.last_fm_boxes else None
        
        if run_fm and (hasattr(self, "fm") and self.fm and getattr(self.fm, "ready", False)):
            try:
                fm_res = self.fm.infer(frame)
                
                # 评估Face Mesh结果的可靠性
                fm_valid_boxes = 0
                for box_key in ["left_eye_box", "right_eye_box", "mouth_box"]:
                    if fm_res.get(box_key) is not None:
                        fm_valid_boxes += 1
                
                # 只有当Face Mesh检测到足够多的有效框时才更新结果
                # 或者当前没有任何YOLO检测结果时也接受Face Mesh结果
                fm_reliability = fm_valid_boxes / 3.0  # 计算可靠性分数
                threshold = getattr(P, "FM_RELIABILITY_THRESHOLD", 0.6)
                
                # 如果Face Mesh结果可靠，或者当前没有任何检测结果，就使用它
                if fm_reliability >= threshold or det_count == 0:
                    self.last_all_pts = fm_res.get("all_pts", None)
                    self.last_fm_boxes = {
                        "left": fm_res.get("left_eye_box"),
                        "right": fm_res.get("right_eye_box"),
                        "mouth": fm_res.get("mouth_box")
                    }
                    self.last_fm_frame = self.frame_idx
                    self.log.debug(f"FaceMesh结果可靠性: {fm_reliability:.2f}, 有效框数: {fm_valid_boxes}")
                elif prev_fm_boxes:  # 如果当前FM结果不可靠但有之前的结果，继续使用之前的
                    self.log.debug(f"FaceMesh结果不可靠 ({fm_reliability:.2f} < {threshold}), 保留之前的结果")
                
            except Exception as e:
                self.log.debug(f"FaceMesh推理失败: {e}")
                # 出错时保持之前的Face Mesh结果不变

        # 5) 增强的位置精修策略 - 混合YOLO和Face Mesh的优势
        if getattr(P, "FM_REFINE_POS", True) and self.last_fm_boxes:
            def _v(b):
                return b and (b[2] > b[0]) and (b[3] > b[1])

            only_on_unc = getattr(P, "FM_REFINE_ONLY_ON_UNCERT", False)
            
            # 定义一个函数来智能合并YOLO和Face Mesh的检测框
            def smart_merge_boxes(yolo_box, fm_box, conf=None, region_type="eye"):
                """智能合并YOLO和Face Mesh的检测框，根据两者的特点和区域类型"""
                if not _v(yolo_box) and _v(fm_box):
                    # 只有Face Mesh框有效时，使用它但进行适当调整以确保质量
                    if region_type == "eye":
                        # 对眼睛区域进行额外验证和微调
                        cx = (fm_box[0] + fm_box[2]) // 2
                        cy = (fm_box[1] + fm_box[3]) // 2
                        width = fm_box[2] - fm_box[0]
                        height = fm_box[3] - fm_box[1]
                        # 轻微缩小框以提高精度，但保持稳定性
                        new_width = int(width * 0.9)  # 减少缩小比例，提高稳定性
                        new_height = int(height * 0.9)  # 减少缩小比例，提高稳定性
                        # 确保框不会越界
                        x1 = max(0, cx - new_width // 2)
                        y1 = max(0, cy - new_height // 2)
                        x2 = min(w - 1, cx + new_width // 2)
                        y2 = min(h - 1, cy + new_height // 2)
                        return (x1, y1, x2, y2)
                    # 对非眼睛区域也进行边界检查
                    x1, y1, x2, y2 = fm_box
                    return (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
                
                if _v(yolo_box) and not _v(fm_box):
                    # 确保YOLO框不会越界
                    x1, y1, x2, y2 = yolo_box
                    return (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
                
                if not _v(yolo_box) and not _v(fm_box):
                    return None
                
                # 两者都有效时，进行智能合并
                # Face Mesh在定位上更准确，YOLO在分类上更可靠
                # 对于眼睛区域，优先使用Face Mesh的精确位置
                if region_type == "eye":
                    # 降低FaceMesh权重，增加YOLO权重，提高稳定性
                    weight_fm = 0.70  # 降低Face Mesh权重到70%
                    weight_yolo = 0.30  # 增加YOLO权重到30%
                    
                    # 根据YOLO置信度动态调整权重，但更保守
                    if conf is not None and conf > 0.5:  # 降低置信度阈值
                        weight_yolo = min(0.45, conf * 0.5)  # 置信度较高时适度增加YOLO权重
                        weight_fm = 1.0 - weight_yolo
                    
                    # 合并框坐标
                    yx1, yy1, yx2, yy2 = yolo_box
                    fx1, fy1, fx2, fy2 = fm_box
                    
                    # 计算加权平均
                    mx1 = int(yx1 * weight_yolo + fx1 * weight_fm)
                    my1 = int(yy1 * weight_yolo + fy1 * weight_fm)
                    mx2 = int(yx2 * weight_yolo + fx2 * weight_fm)
                    my2 = int(yy2 * weight_yolo + fy2 * weight_fm)
                    
                    # 对眼睛区域进行特殊处理：确保框更精确地包含眼睛
                    cx = (mx1 + mx2) // 2
                    cy = (my1 + my2) // 2
                    width = mx2 - mx1
                    height = my2 - my1
                    
                    # 对于眼睛，保持较小的框但确保宽高比，减少调整幅度
                    target_aspect_ratio = 2.0  # 更宽松的眼睛理想宽高比
                    current_aspect_ratio = width / max(1.0, height)
                    
                    # 只在宽高比差异较大时进行调整，减少频繁变化
                    if abs(current_aspect_ratio - target_aspect_ratio) > 0.8:
                        # 调整框以更好地匹配眼睛的预期宽高比
                        if current_aspect_ratio > target_aspect_ratio:
                            # 如果框太宽，调整高度
                            ideal_height = max(1, int(width / target_aspect_ratio))
                            height = min(height, ideal_height)  # 不超过当前高度
                        else:
                            # 如果框太窄，调整宽度
                            ideal_width = max(1, int(height * target_aspect_ratio))
                            width = min(width, ideal_width)  # 不超过当前宽度
                    
                    # 轻微缩小框以提高精度，但保持稳定性
                    new_width = int(width * 0.9)  # 减少缩小比例，提高稳定性
                    new_height = int(height * 0.9)  # 减少缩小比例，提高稳定性
                    
                    # 重新计算坐标并确保不会越界
                    mx1 = max(0, cx - new_width // 2)
                    my1 = max(0, cy - new_height // 2)
                    mx2 = min(w - 1, cx + new_width // 2)
                    my2 = min(h - 1, cy + new_height // 2)
                    
                    return (mx1, my1, mx2, my2)
                else:
                    # 非眼睛区域使用原来的合并逻辑
                    weight_fm = 0.65  # Face Mesh权重
                    weight_yolo = 0.35  # YOLO权重
                    
                    if conf is not None and conf > 0.4:
                        weight_yolo = min(0.5, conf * 0.7)
                        weight_fm = 1.0 - weight_yolo
                    
                    yx1, yy1, yx2, yy2 = yolo_box
                    fx1, fy1, fx2, fy2 = fm_box
                    
                    mx1 = int(yx1 * weight_yolo + fx1 * weight_fm)
                    my1 = int(yy1 * weight_yolo + fy1 * weight_fm)
                    mx2 = int(yx2 * weight_yolo + fx2 * weight_fm)
                    my2 = int(yy2 * weight_yolo + fy2 * weight_fm)
                    
                    cx = (mx1 + mx2) // 2
                    cy = (my1 + my2) // 2
                    width = mx2 - mx1
                    height = my2 - my1
                    new_width = int(width * 0.80)
                    new_height = int(height * 0.80)
                    
                    mx1 = cx - new_width // 2
                    my1 = cy - new_height // 2
                    mx2 = cx + new_width // 2
                    my2 = cy + new_height // 2
                    
                    return (mx1, my1, mx2, my2)
            
            # 获取YOLO检测框的置信度
            le_conf = le[1] if le else None
            re_conf = re[1] if re else None
            m_conf = m[1] if m else None
            
            # 对每个部位进行智能合并或替换，优先使用FaceMesh进行眼睛定位
            # 左眼处理
            if _v(self.last_fm_boxes.get("left")):
                # 对于眼睛区域，优先使用FaceMesh结果，因为它更准确
                fm_left_box = self.last_fm_boxes["left"]
                
                # 验证FaceMesh框的有效性 - 使用更宽松的条件
                is_valid, _ = self._validate_eye_box(fm_left_box, w, h, "left")
                
                if is_valid or not _v(left_eye_box):  # 如果FaceMesh框有效或YOLO没有检测框，就使用FaceMesh
                    if _v(left_eye_box):
                        # 如果YOLO也有检测框，使用智能合并函数
                        left_eye_box = smart_merge_boxes(left_eye_box, fm_left_box, le_conf, "eye")
                    else:
                        # 只有FaceMesh框有效时，直接使用它，但进行精度优化
                        left_eye_box = fm_left_box
                        # 对眼睛框进行精细调整，使用更稳定的参数
                        if _v(left_eye_box):
                            mx1, my1, mx2, my2 = left_eye_box
                            cx = (mx1 + mx2) // 2
                            cy = (my1 + my2) // 2
                            width = mx2 - mx1
                            height = my2 - my1
                            # 针对眼睛区域的精确调整 - 使用稳定的缩放比例
                            new_width = int(width * 0.92)  # 更少的缩小，保持稳定性
                            new_height = int(height * 0.92)  # 更少的缩小，保持稳定性
                            # 确保宽高比接近理想值，但不做过度调整
                            target_ratio = 2.0
                            if width > 0 and height > 0:
                                current_ratio = width / height
                                if current_ratio > target_ratio + 1.0:
                                    # 只在宽高比差异较大时调整
                                    new_height = min(new_height, int(new_width / target_ratio))
                                elif current_ratio < target_ratio - 1.0:
                                    # 只在宽高比差异较大时调整
                                    new_width = min(new_width, int(new_height * target_ratio))
                            left_eye_box = (cx - new_width // 2, cy - new_height // 2, 
                                           cx + new_width // 2, cy + new_height // 2)
            
            # 对左眼框进行平滑处理，并确保不越界
            left_eye_box = self._smooth_box(left_eye_box, "left_eye", w, h)
            
            # 右眼处理 - 与左眼相同的处理逻辑
            if _v(self.last_fm_boxes.get("right")):
                # 对于眼睛区域，优先使用FaceMesh结果
                fm_right_box = self.last_fm_boxes["right"]
                
                # 验证FaceMesh框的有效性 - 使用更宽松的条件
                is_valid, _ = self._validate_eye_box(fm_right_box, w, h, "right")
                
                if is_valid or not _v(right_eye_box):  # 如果FaceMesh框有效或YOLO没有检测框，就使用FaceMesh
                    if _v(right_eye_box):
                        # 如果YOLO也有检测框，使用智能合并函数
                        right_eye_box = smart_merge_boxes(right_eye_box, fm_right_box, re_conf, "eye")
                    else:
                        # 只有FaceMesh框有效时，直接使用它，但进行精度优化
                        right_eye_box = fm_right_box
                        # 对眼睛框进行精细调整，使用更稳定的参数
                        if _v(right_eye_box):
                            mx1, my1, mx2, my2 = right_eye_box
                            cx = (mx1 + mx2) // 2
                            cy = (my1 + my2) // 2
                            width = mx2 - mx1
                            height = my2 - my1
                            # 针对眼睛区域的精确调整 - 使用稳定的缩放比例
                            new_width = int(width * 0.92)  # 更少的缩小，保持稳定性
                            new_height = int(height * 0.92)  # 更少的缩小，保持稳定性
                            # 确保宽高比接近理想值，但不做过度调整
                            target_ratio = 2.0
                            if width > 0 and height > 0:
                                current_ratio = width / height
                                if current_ratio > target_ratio + 1.0:
                                    # 只在宽高比差异较大时调整
                                    new_height = min(new_height, int(new_width / target_ratio))
                                elif current_ratio < target_ratio - 1.0:
                                    # 只在宽高比差异较大时调整
                                    new_width = min(new_width, int(new_height * target_ratio))
                            right_eye_box = (cx - new_width // 2, cy - new_height // 2, 
                                            cx + new_width // 2, cy + new_height // 2)
            
            # 对右眼框进行平滑处理，并确保不越界
            right_eye_box = self._smooth_box(right_eye_box, "right_eye", w, h)
            
            # 如果只有一只眼睛被检测到，尝试基于对称性创建另一只眼睛，但添加稳定性检查
            if (_v(left_eye_box) and not _v(right_eye_box)) or (_v(right_eye_box) and not _v(left_eye_box)):
                # 检查是否有历史的另一只眼睛框，如果有，优先使用历史框
                if _v(left_eye_box) and self.history["right_eye"]:
                    # 使用历史右眼框，保持稳定性
                    right_eye_box = self.history["right_eye"][-1]
                elif _v(right_eye_box) and self.history["left_eye"]:
                    # 使用历史左眼框，保持稳定性
                    left_eye_box = self.history["left_eye"][-1]
                else:
                    # 没有历史框时，基于对称性创建另一只眼睛框
                    existing_box = left_eye_box if _v(left_eye_box) else right_eye_box
                    is_left = _v(left_eye_box)
                    
                    # 基于存在的眼睛框创建对称的另一只眼睛框 - 改进版本
                    ex1, ey1, ex2, ey2 = existing_box
                    ecx = (ex1 + ex2) // 2
                    ecy = (ey1 + ey2) // 2
                    width = ex2 - ex1
                    height = ey2 - ey1
                    
                    # 计算对称眼睛位置 - 使用更精确的面部中心点和距离计算
                    # 考虑到人眼通常不是完全对称的，使用更自然的偏移
                    face_center = w // 2
                    distance_from_center = abs(ecx - face_center)
                    
                    # 增加一个微调因子，使生成的眼睛框更自然
                    adjustment_factor = 0.95  # 稍微调整距离，避免完全对称
                    
                    if is_left:
                        # 基于左眼创建右眼
                        right_cx = face_center + int(distance_from_center * adjustment_factor)
                        # 稍微调整垂直位置，使眼睛框更自然
                        right_ey = ecy + 1  # 右眼通常略微低于左眼
                        right_box = (right_cx - width//2, right_ey - height//2, 
                                    right_cx + width//2, right_ey + height//2)
                        right_eye_box = self._clamp_box(right_box, w, h)
                    else:
                        # 基于右眼创建左眼
                        left_cx = face_center - int(distance_from_center * adjustment_factor)
                        # 稍微调整垂直位置，使眼睛框更自然
                        left_ey = ecy - 1  # 左眼通常略微高于右眼
                        left_box = (left_cx - width//2, left_ey - height//2, 
                                   left_cx + width//2, left_ey + height//2)
                        left_eye_box = self._clamp_box(left_box, w, h)
            
            # 嘴巴 - 与眼睛使用相同的处理逻辑结构
            if _v(self.last_fm_boxes.get("mouth")):
                # 嘴巴使用与眼睛相同的判断条件
                if (not only_on_unc or m_unc) or (not _v(mouth_box)):
                    # 嘴巴和眼睛使用相同的处理逻辑
                    mouth_box = self.last_fm_boxes["mouth"]
                    # 保持一致的框处理
                    if _v(mouth_box):
                        mx1, my1, mx2, my2 = mouth_box
                        cx = (mx1 + mx2) // 2
                        cy = (my1 + my2) // 2
                        # 嘴巴使用稍大的缩放比例，确保包含整个嘴巴区域
                        width = mx2 - mx1
                        height = my2 - my1
                        new_width = int(width * 0.85)  # 嘴巴略微小一点，更精确
                        new_height = int(height * 0.85)  # 嘴巴略微小一点，更精确
                        mouth_box = (cx - new_width // 2, cy - new_height // 2, 
                                    cx + new_width // 2, cy + new_height // 2)
                else:
                    # 嘴巴使用与眼睛相同的融合权重
                    weight_fm = 0.70  # 与眼睛相同的权重设置
                    weight_yolo = 0.30
                    
                    yx1, yy1, yx2, yy2 = mouth_box
                    fx1, fy1, fx2, fy2 = self.last_fm_boxes["mouth"]
                    
                    # 嘴巴使用与眼睛相同的加权融合逻辑
                    mx1 = int(yx1 * weight_yolo + fx1 * weight_fm)
                    my1 = int(yy1 * weight_yolo + fy1 * weight_fm)
                    mx2 = int(yx2 * weight_yolo + fx2 * weight_fm)
                    my2 = int(yy2 * weight_yolo + fy2 * weight_fm)
                    
                    # 确保嘴巴框与眼睛框保持一致的处理方式
                    mouth_box = (mx1, my1, mx2, my2)

        # 6) 左右修正和单边情况矫正（与原逻辑一致）
        def _center(box):
            if box is None:
                return None
            return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

        mid_x = w / 2.0
        cle = _center(left_eye_box)
        cre = _center(right_eye_box)
        if cle and cre:
            mid_x = (cle[0] + cre[0]) / 2.0
        if left_eye_box and right_eye_box and cle and cre and cle[0] > mid_x and cre[0] < mid_x:
            left_eye_box, right_eye_box = right_eye_box, left_eye_box

        # 7) 增强的EAR/MAR计算策略 - 更频繁地使用几何特征作为补充
        use_all_pts = None
        # 除了YOLO不确定时，当检测结果较少或置信度较低时也使用Face Mesh的几何特征
        if (le_unc or re_unc or m_unc or det_count < 3):
            if self.last_all_pts is not None and (self.frame_idx - self.last_fm_frame) <= getattr(P, "FM_CACHE_EXPIRE",
                                                                                                  10):
                use_all_pts = self.last_all_pts
        
        # 扩展EAR/MAR的使用场景：当YOLO检测到框但置信度较低时，也使用几何特征进行验证
        low_confidence = False
        if le and le[1] < 0.3:  # 如果左眼置信度低于0.3
            low_confidence = True
        if re and re[1] < 0.3:  # 如果右眼置信度低于0.3
            low_confidence = True
        if m and m[1] < 0.3:  # 如果嘴巴置信度低于0.3
            low_confidence = True
            
        if low_confidence and self.last_all_pts is not None and (self.frame_idx - self.last_fm_frame) <= getattr(P, "FM_CACHE_EXPIRE", 10):
            use_all_pts = self.last_all_pts

        # 获取眼镜状态并传递给EAR计算函数
        is_wearing_glasses = getattr(self, 'is_wearing_glasses', False)
        ear_l = _ear_from_all_pts(use_all_pts, left=True, is_wearing_glasses=is_wearing_glasses) if (use_all_pts is not None and le_unc) else None
        ear_r = _ear_from_all_pts(use_all_pts, left=False, is_wearing_glasses=is_wearing_glasses) if (use_all_pts is not None and re_unc) else None
        mar = _mar_from_all_pts(use_all_pts) if (use_all_pts is not None and m_unc) else None

        # 优化的自适应阈值采样 - 更智能地选择有效样本
        if self.calib_enabled and not self.calibrated and now < self.calib_deadline:
            # 只有当YOLO和EAR都有结果且一致时，才认为这是一个可靠的样本
            if (ear_l is not None and ear_r is not None and 0.0 < ear_l < 1.0 and 0.0 < ear_r < 1.0 and
                # 确保EAR值在合理范围内，避免异常值
                not (le_unc and re_unc)):  # 避免在YOLO都不确定时采样
                # 计算两个眼睛的EAR平均值，但给每个眼睛添加权重
                # 如果某只眼睛的YOLO结果更确定，给予更高权重
                le_weight = 0.5
                re_weight = 0.5
                
                # 根据YOLO的置信度差异动态调整权重
                if not le_unc and re_unc:
                    le_weight = 0.7
                    re_weight = 0.3
                elif le_unc and not re_unc:
                    le_weight = 0.3
                    re_weight = 0.7
                
                # 计算加权平均值
                ear_mean = float((ear_l * le_weight + ear_r * re_weight) / (le_weight + re_weight))
                
                # 考虑眼镜因素进行异常值过滤
                if hasattr(self, 'is_wearing_glasses') and self.is_wearing_glasses:
                    # 戴眼镜时调整EAR值范围
                    if 0.12 < ear_mean < 0.35:  # 眼镜佩戴者的典型EAR值范围稍低
                        self.calib_samples.append(ear_mean)
                else:
                    # 不戴眼镜时的正常范围
                    if 0.15 < ear_mean < 0.4:  # 典型的EAR值范围
                        self.calib_samples.append(ear_mean)

        if self.calib_enabled and not self.calibrated and (
                now >= self.calib_deadline or len(self.calib_samples) >= P.CALIB_MIN_SAMPLES):
            if len(self.calib_samples) >= max(10, P.CALIB_MIN_SAMPLES // 2):
                baseline = float(np.median(np.array(self.calib_samples)))
                baseline = float(np.clip(baseline, 0.10, 0.45))
                self.ear_close_th = baseline * P.CALIB_FACTOR_CLOSE
                self.ear_open_th = baseline * P.CALIB_FACTOR_OPEN
                if self.ear_open_th <= self.ear_close_th + 0.04:
                    self.ear_open_th = self.ear_close_th + 0.04
                self.calibrated = True
                self.log.info(
                    f"[CALIB] EAR baseline={baseline:.3f}, close={self.ear_close_th:.3f}, open={self.ear_open_th:.3f}")
                if self.status_text:
                    self.status_text.AppendText(
                        f"已标定 EAR 基线: {baseline:.3f} (关:{self.ear_close_th:.3f}/开:{self.ear_open_th:.3f})\n")

        def _smooth_push(buf: deque, val: Optional[float]) -> Optional[float]:
            if val is not None and 0.0 < val < 1.5:
                buf.append(val)
            if len(buf) == 0:
                return None
            return float(np.mean(list(buf)))

        # 应用眼镜检测优化的EAR计算
        ear_l_s = _smooth_push(self.buf_ear_l, ear_l) if le_unc else None
        ear_r_s = _smooth_push(self.buf_ear_r, ear_r) if re_unc else None
        mar_s = _smooth_push(self.buf_mar, mar) if m_unc else None

        def _eye_closed_from_ear(ear_s):
            if ear_s is None:
                return None
                
            # 获取基本阈值
            close_th = getattr(self, "ear_close_th", P.EAR_CLOSE_TH)
            open_th = getattr(self, "ear_open_th", P.EAR_OPEN_TH)
            
            # 根据是否戴眼镜调整阈值
            if hasattr(self, 'is_wearing_glasses') and self.is_wearing_glasses:
                # 戴眼镜时调整阈值，更合理地适应眼镜影响
                close_th *= 0.95  # 略微降低闭眼阈值
                open_th *= 0.65  # 大幅降低戴眼镜时的睁眼阈值，更容易检测到睁眼
            
            # 强制睁眼逻辑：如果EAR值接近阈值，倾向于判定为睁眼
            if abs(ear_s - open_th) < 0.02:
                return False
            
            if ear_s <= close_th:
                return True
            if ear_s >= open_th:
                return False
            return None

        def _mouth_open_from_mar(mar_s):
            if mar_s is None:
                return None
            if mar_s >= P.MAR_OPEN_TH:
                return True
            if mar_s <= P.MAR_CLOSE_TH:
                return False
            return None

        # 眼镜检测逻辑 - 使用FaceMeshHelper中的_detect_glasses方法
        is_wearing_glasses = getattr(self, 'is_wearing_glasses', False)
        
        # 定期进行眼镜检测（每几帧检测一次）
        if self.frame_idx % self.glasses_detection_interval == 0 and use_all_pts is not None:
            try:
                # 直接调用FaceMeshHelper中的眼镜检测方法
                if self.fmh is not None and hasattr(self.fmh, '_detect_glasses'):
                    is_wearing_glasses, confidence = self.fmh._detect_glasses(use_all_pts)
                    self.is_wearing_glasses = is_wearing_glasses
                    self.glasses_confidence = confidence
                    self.log.debug(f"眼镜检测结果: {is_wearing_glasses}, 置信度: {confidence:.2f}")
                    
                    # 调用参数调整方法
                    if hasattr(self.fmh, 'adjust_parameters_for_glasses'):
                        self.fmh.adjust_parameters_for_glasses()
                else:
                    # 备用检测逻辑
                    if ear_l_s is not None and ear_r_s is not None:
                        avg_ear = (ear_l_s + ear_r_s) / 2
                        ear_diff = abs(ear_l_s - ear_r_s)
                        
                        if left_eye_box and right_eye_box:
                            left_area = (left_eye_box[2] - left_eye_box[0]) * (left_eye_box[3] - left_eye_box[1])
                            right_area = (right_eye_box[2] - right_eye_box[0]) * (right_eye_box[3] - right_eye_box[1])
                            area_ratio = left_area / right_area if right_area > 0 else 0
                            
                            # 眼镜检测条件
                            if (0.25 < avg_ear < 0.4) and (ear_diff < 0.08) and (0.8 < area_ratio < 1.2):
                                self.glasses_history.append(1.0)
                            else:
                                self.glasses_history.append(0.0)
                            
                            if len(self.glasses_history) > 10:
                                self.glasses_history.pop(0)
                            
                            if len(self.glasses_history) > 0:
                                self.glasses_confidence = sum(self.glasses_history) / len(self.glasses_history)
                                self.is_wearing_glasses = self.glasses_confidence > 0.6
                                is_wearing_glasses = self.is_wearing_glasses
            except Exception as e:
                self.log.debug(f"眼镜检测失败: {e}")
        else:
            is_wearing_glasses = getattr(self, 'is_wearing_glasses', False)
        
        # 8) 增强的最终状态判断 - 混合YOLO和几何特征，优化决策逻辑
        # 左眼
        le_yolo_closed = (le_closed_s >= le_open_s) and (max(le_closed_s, le_open_s) >= P.YOLO_UNCERT_TH) and (
                    abs(le_closed_s - le_open_s) >= P.YOLO_DIFF_TH)
        le_yolo_open = (le_open_s > le_closed_s) and (max(le_closed_s, le_open_s) >= P.YOLO_UNCERT_TH) and (
                    abs(le_closed_s - le_open_s) >= P.YOLO_DIFF_TH)
        
        # 改进的决策逻辑：更频繁地使用EAR作为补充或验证，考虑眼镜因素
        ear_flag = _eye_closed_from_ear(ear_l_s) if ear_l_s is not None else None
        
        # 戴眼镜时增加EAR的权重
        if is_wearing_glasses and ear_flag is not None:
            # 对于戴眼镜的用户，EAR特征更可靠
            if le_unc or abs(le_closed_s - le_open_s) < 0.15:
                # 如果YOLO不确定或差异较小，优先使用EAR
                le_closed_flag = ear_flag
            else:
                # 否则使用标准逻辑
                yolo_decision = le_yolo_closed
                if ear_flag != yolo_decision:
                    # 降低YOLO的置信度差异阈值
                    if abs(le_closed_s - le_open_s) < 0.15:
                        le_closed_flag = ear_flag
                        self.log.debug(f"[眼镜模式] EAR结果与YOLO矛盾，采用EAR: {ear_flag}")
                    else:
                        le_closed_flag = yolo_decision
                else:
                    le_closed_flag = yolo_decision
        else:
            # 不戴眼镜时的标准逻辑
            # 当YOLO结果明确时，优先使用YOLO，但考虑EAR作为辅助验证
            if le_yolo_closed or le_yolo_open:
                yolo_decision = le_yolo_closed
                # 如果EAR有明确结果且与YOLO结果矛盾，降低YOLO的置信度
                if ear_flag is not None and ear_flag != yolo_decision:
                    # 检查YOLO的置信度差异
                    if abs(le_closed_s - le_open_s) < 0.2:  # 如果差异不大
                        # 使用EAR结果作为最终判断
                        le_closed_flag = ear_flag
                        self.log.debug(f"EAR结果与YOLO矛盾，YOLO差异小，采用EAR: {ear_flag}")
                    else:
                        # 否则仍然优先使用YOLO，但记录矛盾情况
                        le_closed_flag = yolo_decision
                        self.log.debug(f"EAR结果与YOLO矛盾，YOLO差异大，采用YOLO: {yolo_decision}")
                else:
                    le_closed_flag = yolo_decision
            else:
                # 当YOLO结果不明确时，优先使用EAR结果
                if ear_flag is not None:
                    le_closed_flag = ear_flag
                else:
                    # 都不明确时，使用YOLO分数比较
                    le_closed_flag = (le_closed_s >= le_open_s)
                    # 但如果分数非常接近，倾向于认为是睁开的（避免误报）
                    if abs(le_closed_s - le_open_s) < 0.05:
                        le_closed_flag = False

        # 右眼
        re_yolo_closed = (re_closed_s >= re_open_s) and (max(re_closed_s, re_open_s) >= P.YOLO_UNCERT_TH) and (
                    abs(re_closed_s - re_open_s) >= P.YOLO_DIFF_TH)
        re_yolo_open = (re_open_s > re_closed_s) and (max(re_closed_s, re_open_s) >= P.YOLO_UNCERT_TH) and (
                    abs(re_closed_s - re_open_s) >= P.YOLO_DIFF_TH)
        
        # 改进的右眼决策逻辑，考虑眼镜因素
        ear_flag = _eye_closed_from_ear(ear_r_s) if ear_r_s is not None else None
        
        # 戴眼镜时增加EAR的权重
        if is_wearing_glasses and ear_flag is not None:
            # 对于戴眼镜的用户，EAR特征更可靠
            if re_unc or abs(re_closed_s - re_open_s) < 0.15:
                # 如果YOLO不确定或差异较小，优先使用EAR
                re_closed_flag = ear_flag
            else:
                # 否则使用标准逻辑
                yolo_decision = re_yolo_closed
                if ear_flag != yolo_decision:
                    # 降低YOLO的置信度差异阈值
                    if abs(re_closed_s - re_open_s) < 0.15:
                        re_closed_flag = ear_flag
                        self.log.debug(f"[眼镜模式] EAR结果与YOLO矛盾，采用EAR: {ear_flag}")
                    else:
                        re_closed_flag = yolo_decision
                else:
                    re_closed_flag = yolo_decision
        else:
            # 不戴眼镜时的标准逻辑
            if re_yolo_closed or re_yolo_open:
                yolo_decision = re_yolo_closed
                if ear_flag is not None and ear_flag != yolo_decision:
                    if abs(re_closed_s - re_open_s) < 0.2:
                        re_closed_flag = ear_flag
                        self.log.debug(f"EAR结果与YOLO矛盾，YOLO差异小，采用EAR: {ear_flag}")
                    else:
                        re_closed_flag = yolo_decision
                        self.log.debug(f"EAR结果与YOLO矛盾，YOLO差异大，采用YOLO: {yolo_decision}")
                else:
                    re_closed_flag = yolo_decision
            else:
                if ear_flag is not None:
                    re_closed_flag = ear_flag
                else:
                    re_closed_flag = (re_closed_s >= re_open_s)
                    if abs(re_closed_s - re_open_s) < 0.05:
                        re_closed_flag = False

        # 嘴巴
        m_yolo_open = (m_open_s >= m_closed_s) and (max(m_open_s, m_closed_s) >= P.YOLO_UNCERT_TH) and (
                    abs(m_open_s - m_closed_s) >= P.YOLO_DIFF_TH)
        m_yolo_close = (m_closed_s > m_open_s) and (max(m_open_s, m_closed_s) >= P.YOLO_UNCERT_TH) and (
                    abs(m_open_s - m_closed_s) >= P.YOLO_DIFF_TH)
        
        # 改进的嘴巴决策逻辑，类似眼睛但有特殊处理
        mar_flag = _mouth_open_from_mar(mar_s) if mar_s is not None else None
        
        if m_yolo_open or m_yolo_close:
            yolo_decision = m_yolo_open
            # 对于嘴巴，MAR特征通常更准确，所以更倾向于使用MAR结果
            if mar_flag is not None and mar_flag != yolo_decision:
                # 对于嘴巴，即使YOLO差异较大，也更倾向于使用MAR
                if abs(m_open_s - m_closed_s) < 0.25:
                    mouth_open_flag = mar_flag
                    self.log.debug(f"MAR结果与YOLO矛盾，YOLO差异小，采用MAR: {mar_flag}")
                else:
                    # 但如果YOLO非常确定，还是使用YOLO
                    mouth_open_flag = yolo_decision
                    self.log.debug(f"MAR结果与YOLO矛盾，YOLO差异大，采用YOLO: {yolo_decision}")
            else:
                mouth_open_flag = yolo_decision
        else:
            # 当YOLO结果不明确时，优先使用MAR结果
            if mar_flag is not None:
                mouth_open_flag = mar_flag
            else:
                # 都不明确时，使用YOLO分数比较
                mouth_open_flag = (m_open_s >= m_closed_s)
                # 但如果分数非常接近，倾向于认为是闭合的（避免误报哈欠）
                if abs(m_open_s - m_closed_s) < 0.05:
                    mouth_open_flag = False

        # 9) 增强的画框与状态输出逻辑 - 优化框的显示和稳定性
        # 左眼 - 使用完全相同的处理逻辑
        # 左眼
        if left_eye_box is not None:
            # 优化EMA参数，让框更平滑但又能快速响应变化
            b = ema_box(self.prev["left_eye"]["box"], left_eye_box)
            self.prev["left_eye"]["box"] = b
            self.prev["left_eye"]["ttl"] = P.HOLD_TTL * 2  # 延长保持时间，确保框持续显示
            
            # 应用平滑器，增加状态的稳定性
            flags["left_closed"] = self.sm["left_closed"].update(bool(le_closed_flag))
            
            # 左眼框暂不直接绘制，等待后处理统一绘制
        elif self.prev["left_eye"]["ttl"] > 0 and self.prev["left_eye"]["box"] is not None:
            # 即使当前没有新检测，也保持框的显示
            self.prev["left_eye"]["ttl"] -= 1

        # 右眼
        if right_eye_box is not None:
            # 优化EMA参数，让框更平滑但又能快速响应变化
            b = ema_box(self.prev["right_eye"]["box"], right_eye_box)
            self.prev["right_eye"]["box"] = b
            self.prev["right_eye"]["ttl"] = P.HOLD_TTL * 2  # 延长保持时间，确保框持续显示
            
            # 应用平滑器，增加状态的稳定性
            flags["right_closed"] = self.sm["right_closed"].update(bool(re_closed_flag))
            
            # 右眼框暂不直接绘制，等待后处理统一绘制
        elif self.prev["right_eye"]["ttl"] > 0 and self.prev["right_eye"]["box"] is not None:
            # 即使当前没有新检测，也保持框的显示
            self.prev["right_eye"]["ttl"] -= 1
        
        # 确保左右眼框大小一致的后处理 - 简单暴力实现，强制使用完全相同的大小和精确的眼睛定位
        # 只在这个统一的后处理阶段绘制眼睛框，避免重复绘制
        if self.prev["left_eye"]["box"] is not None:
            l_box = self.prev["left_eye"]["box"]
            # 使用更适合眼睛大小的固定尺寸
            fixed_width = 32  # 更适合眼睛大小的宽度
            fixed_height = 20  # 更适合眼睛大小的高度
            
            # 不做任何偏移
            y_offset = 0  # 保持居中位置
            
            # 重新计算左眼框的坐标
            l_cx = (l_box[0] + l_box[2]) // 2
            l_cy = (l_box[1] + l_box[3]) // 2
            new_l_box = (l_cx - fixed_width // 2, l_cy - fixed_height // 2 + y_offset, 
                         l_cx + fixed_width // 2, l_cy + fixed_height // 2 + y_offset)
            
            # 确保框不会超出图像边界
            h, w = frame.shape[:2]
            new_l_box = (max(0, new_l_box[0]), max(0, new_l_box[1]), 
                         min(w, new_l_box[2]), min(h, new_l_box[3]))
            
            # 更新框的坐标
            self.prev["left_eye"]["box"] = new_l_box
            
            # 绘制左眼框
            try:
                cv2.rectangle(frame, (new_l_box[0], new_l_box[1]), (new_l_box[2], new_l_box[3]), P.C_EYE, P.THICK)
            except Exception as e:
                self.log.warning(f"绘制左眼框失败: {e}")
        
        if self.prev["right_eye"]["box"] is not None:
            r_box = self.prev["right_eye"]["box"]
            # 使用与左眼相同的固定尺寸
            fixed_width = 32  # 与左眼相同的宽度
            fixed_height = 20  # 与左眼相同的高度
            
            # 不做任何偏移
            y_offset = 0  # 保持居中位置
            
            # 重新计算右眼框的坐标
            r_cx = (r_box[0] + r_box[2]) // 2
            r_cy = (r_box[1] + r_box[3]) // 2
            new_r_box = (r_cx - fixed_width // 2, r_cy - fixed_height // 2 + y_offset, 
                         r_cx + fixed_width // 2, r_cy + fixed_height // 2 + y_offset)
            
            # 确保框不会超出图像边界
            h, w = frame.shape[:2]
            new_r_box = (max(0, new_r_box[0]), max(0, new_r_box[1]), 
                         min(w, new_r_box[2]), min(h, new_r_box[3]))
            
            # 更新框的坐标
            self.prev["right_eye"]["box"] = new_r_box
            
            # 绘制右眼框
            try:
                cv2.rectangle(frame, (new_r_box[0], new_r_box[1]), (new_r_box[2], new_r_box[3]), P.C_EYE, P.THICK)
            except Exception as e:
                self.log.warning(f"绘制右眼框失败: {e}")
            
            # 清除之前的绘制，然后重新绘制确保只显示一个框
            # 先清除图像上的所有旧框（简单方法：不重绘，只更新状态）
            # 移除之前的重新绘制代码，避免重叠绘制
            
        # 移除调试信息显示

        # 嘴
        if mouth_box is not None:
            # 优化EMA参数
            b = ema_box(self.prev["mouth"]["box"], mouth_box)
            self.prev["mouth"]["box"] = b
            self.prev["mouth"]["ttl"] = P.HOLD_TTL * 2  # 延长保持时间，确保框持续显示
            
            flags["mouth_open"] = self.sm["mouth_open"].update(bool(mouth_open_flag))
            try:
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), P.C_MOUTH, P.THICK)
                # 移除标签文字，只保留框
            except Exception as e:
                self.log.warning(f"绘制嘴巴框失败: {e}")
        elif self.prev["mouth"]["ttl"] > 0 and self.prev["mouth"]["box"] is not None:
            # 即使当前没有新检测，也保持框的显示，但稍微降低透明度
            self.prev["mouth"]["ttl"] -= 1
            b = self.prev["mouth"]["box"]
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), P.C_MOUTH, max(1, P.THICK - 1))

        if all(v["box"] is None or v["ttl"] == 0 for v in self.prev.values()):
            frame = self._draw_text(frame, "未检测到目标", (10, 120), P.C_HINT)

        # 使用状态锁确保状态更新的线程安全
        with self.state_lock:
            self.last_flags = flags.copy()
        return frame, flags.copy()  # 返回副本以避免外部修改

    # ------------ 计数与疲劳状态 + 语音播报（显式冷却 + 全局间隔 + 单帧仅一事件 + 成功才记日志） ------------
    def update_fatigue_status(self, flags):
        """更新疲劳状态，使用状态锁确保线程安全，支持眼镜优化"""
        with self.state_lock:
            now = time.time()
            # 初始化状态变量，确保所有执行路径都有定义
            status = "清醒"

            # 计数与时长累积
            # 左眼
            if flags["left_closed"]:
                self.left_closed_frames += 1
            else:
                self.left_closed_frames = 0

            # 右眼
            if flags["right_closed"]:
                self.right_closed_frames += 1
            else:
                self.right_closed_frames = 0
                
            # 获取眼镜检测状态
            is_wearing_glasses = getattr(self, 'is_wearing_glasses', False)
            
            # 确保flags中的状态被正确处理，添加防御性检查
            if "left_closed" not in flags:
                flags["left_closed"] = False
            if "right_closed" not in flags:
                flags["right_closed"] = False
            if "mouth_open" not in flags:
                flags["mouth_open"] = False

            # 优化的眨眼计数逻辑：提高检测准确性，支持单眼闭合检测
            left_closed = flags["left_closed"]
            right_closed = flags["right_closed"]
            any_closed = left_closed or right_closed  # 任一眼闭合
            both_closed = left_closed and right_closed  # 双眼闭合
            both_open = not left_closed and not right_closed
            
            # 改进的眨眼检测状态追踪
            if any_closed:
                # 任一眼闭合时开始计时
                if self.both_closed_start_time is None:
                    self.both_closed_start_time = now
                    self.closed_frames_count = 1
                    # 记录是单眼还是双眼闭合
                    self.was_both_closed = both_closed
                else:
                    # 继续闭合，增加帧数计数
                    self.closed_frames_count += 1
                    # 更新闭合类型
                    if both_closed:
                        self.was_both_closed = True
                    
                # 检查是否需要强制重置（防止系统一直卡在闭合状态）
                if self.both_closed_start_time is not None:
                    closing_duration = now - self.both_closed_start_time
                    # 超过2秒的闭合视为异常，重置状态
                    if closing_duration > 2.0:
                        # 强制重置状态
                        self.both_closed_start_time = None
                        self.closed_frames_count = 0
                        # 根据闭合类型和持续时间决定是否强制计数眨眼
                        # 对于超过2秒但不超过5秒的闭合，可以考虑计数一次
                        if now - self.last_blink_t >= 0.5 and closing_duration <= 5.0:
                            # 只有当确实检测到眼睛闭合而不是检测丢失时才计数
                            if hasattr(self, 'face_detected') and self.face_detected:
                                self.blink_count += 1
                                self.last_blink_t = now
                                self.log.info(f"强制计数眨眼: 计数={self.blink_count}, 闭合持续时间={closing_duration:.3f}s, 类型={'双眼' if getattr(self, 'was_both_closed', False) else '单眼'}")
                        # 重置标志
                        if hasattr(self, 'was_both_closed'):
                            delattr(self, 'was_both_closed')
            # 检测从闭合到睁开的转变
            elif not any_closed and self.both_closed_start_time is not None:
                closed_duration = now - self.both_closed_start_time
                was_both_closed = getattr(self, 'was_both_closed', False)
                
                # 优化的眨眼检测参数 - 根据单眼/双眼闭合调整阈值
                min_duration = 0.05 if was_both_closed else 0.08  # 降低最小持续时间，提高检测灵敏度
                max_duration = 0.5   # 降低最大持续时间，明确与长闭眼的边界
                min_frames = 2 if was_both_closed else 3  # 保持不变
                refract_blink = 0.3  # 300ms不应期保持不变
                
                # 眨眼检测条件 - 增加上下文感知
                # 考虑眼镜佩戴状态调整阈值
                is_wearing_glasses = getattr(self, 'is_wearing_glasses', False)
                if is_wearing_glasses:
                    min_duration = max(0.04, min_duration - 0.01)  # 戴眼镜时略微降低
                    min_frames = max(1, min_frames - 1)
                
                # 根据持续时间和帧数综合判断
                duration_match = min_duration <= closed_duration <= max_duration
                frames_match = self.closed_frames_count >= min_frames
                cooldown_ok = now - self.last_blink_t >= refract_blink
                
                # 优化的眨眼判定逻辑 - 降低条件严格性，提高正常眨眼检测率
                if cooldown_ok:  # 首先确保冷却期已过
                    # 对于双眼闭合，判定条件更宽松
                    if was_both_closed and frames_match and closed_duration >= min_duration:
                        self.blink_count += 1
                        self.last_blink_t = now
                        self.log.info(f"成功检测到眨眼(双眼): 计数={self.blink_count}, 持续时间={closed_duration:.3f}s, 帧数={self.closed_frames_count}")
                    # 对于单眼闭合，判定条件略微宽松
                    elif not was_both_closed and frames_match and closed_duration >= min_duration * 0.9:  # 略微放宽单眼检测
                        self.blink_count += 1
                        self.last_blink_t = now
                        self.log.info(f"成功检测到眨眼(单眼): 计数={self.blink_count}, 持续时间={closed_duration:.3f}s, 帧数={self.closed_frames_count}")
                
                # 重置状态
                self.both_closed_start_time = None
                self.closed_frames_count = 0
                # 重置was_both_closed而不是删除
                if hasattr(self, 'was_both_closed'):
                    self.was_both_closed = False
            
            # 智能强制眨眼检测：根据系统状态动态调整触发条件
            # 这是为了确保在眼睛检测存在问题的情况下，系统仍然能够有合理的眨眼计数
            # 但避免过度强制计数影响真实眨眼频率的准确性
            force_blink_interval = 15.0
            # 如果已检测到面部，增加强制计数的时间间隔
            if hasattr(self, 'face_detected') and self.face_detected:
                force_blink_interval = 20.0
            # 如果已经检测到多次眨眼，也可以适当增加间隔
            time_elapsed = now - self.time_start
            if time_elapsed > 60 and self.blink_count > 10:
                force_blink_interval = 25.0
                
            if (now - self.last_blink_t > force_blink_interval and 
                self.blink_sequence != "closing" and 
                time_elapsed > 10):  # 系统运行10秒后才开始强制计数
                # 增加面部检测状态检查，只有确实检测到面部但可能眨眼检测有问题时才计数
                if not hasattr(self, 'face_detected') or self.face_detected:
                    self.blink_count += 1
                    self.last_blink_t = now
                    self.log.info(f"智能强制眨眼计数: 长时间未检测到眨眼，计数增加到 {self.blink_count}, 面部检测状态={'已检测' if getattr(self, 'face_detected', True) else '未检测'}")
            
            # 优化的哈欠检测逻辑：确保只有明确的哈欠动作才计数
            if flags["mouth_open"]:
                # 只在连续检测到张嘴时增加计数
                if self.yawn_sequence != "opening":
                    self.yawn_sequence = "opening"
                    self.yawn_open_start_time = now
                    self.mouth_open_frames = 1  # 正确初始化帧数
                else:
                    self.mouth_open_frames += 1

                # 严格哈欠检测条件，确保是真实哈欠动作
                if self.yawn_open_start_time is not None:
                    yawn_duration = now - self.yawn_open_start_time
                    # 哈欠必须满足持续时间和帧数要求
                    if (yawn_duration >= P.YAWN_OPEN_MIN_SEC and
                            self.mouth_open_frames >= self.yawn_open_min_frames and
                            now - self.last_yawn_t >= P.YAWN_REFRACT):
                        self.yawn_count += 1
                        self.last_yawn_t = now
                        self.log.debug(f"哈欠计数+1: 持续时间={yawn_duration:.2f}s, 帧数={self.mouth_open_frames}, 当前计数={self.yawn_count}")
                        # 重置哈欠状态
                        self.mouth_open_frames = 0
                        self.yawn_sequence = "closed"
                        self.yawn_open_start_time = None
            else:
                # 当嘴巴闭合时，重置哈欠相关状态
                self.mouth_open_frames = 0
                self.yawn_sequence = "closed"
                self.yawn_open_start_time = None

            # 客观疲劳定义：简单科学的四级疲劳分级系统
            # 基于科学研究的驾驶员疲劳检测标准
            time_elapsed = now - self.time_start
            
            # 初始化长闭眼计数器（重要的客观疲劳指标）
            # 注意：这些属性已在reset_counters中初始化，但保留防御性检查
            if not hasattr(self, 'long_eye_close_count'):
                self.long_eye_close_count = 0
            if not hasattr(self, 'long_eye_close_start_time'):
                self.long_eye_close_start_time = None
            if not hasattr(self, 'last_long_close_time'):
                self.last_long_close_time = 0
            if not hasattr(self, 'fatigue_level'):
                self.fatigue_level = 0  # 0-清醒, 1-轻度疲劳, 2-中度疲劳, 3-严重疲劳
            
            # 增强的长闭眼检测逻辑 - 客观疲劳的核心指标
            # 1. 首先确定是否处于可能的长闭眼状态
            potential_long_close = any_closed
            # 2. 如果正在检测长闭眼，继续计时
            if potential_long_close:
                if self.long_eye_close_start_time is None:
                    self.long_eye_close_start_time = now
                    # 记录本次长闭眼开始时的状态
                    self.current_long_close_both_eyes = both_closed
                    self.current_long_close_frames = 1
                else:
                    # 继续累积帧数
                    self.current_long_close_frames += 1
                    # 更新是否为双眼闭合
                    if both_closed:
                        self.current_long_close_both_eyes = True
                
                # 3. 检查是否达到长闭眼阈值
                close_duration = now - self.long_eye_close_start_time
                # 长闭眼最小持续时间 - 明确与正常眨眼的边界，避免重叠
                min_long_close_duration = max(0.6, P.LONG_EYE_CLOSED_SEC)  # 确保至少0.6秒，与眨眼明确区分
                # 对于双眼闭合，可以适当调整但不低于眨眼的最大持续时间
                if getattr(self, 'current_long_close_both_eyes', False):
                    min_long_close_duration = max(0.6, min_long_close_duration * 0.9)  # 略微降低但保持与眨眼的边界
                
                # 4. 检测是否达到长闭眼标准，同时避免短时间内重复计数
                cooldown_period = 2.5  # 长闭眼计数的冷却期
                frames_threshold = int(min_long_close_duration * P.FPS * 0.9)  # 帧数阈值，略微提高要求
                
                # 长闭眼检测的额外保护：如果刚刚计数了眨眼，不应立即判定为长闭眼
                recently_blinked = now - self.last_blink_t < 0.5  # 0.5秒内有眨眼
                
                if (close_duration >= min_long_close_duration and 
                    now - self.last_long_close_time > cooldown_period and
                    getattr(self, 'current_long_close_frames', 0) >= frames_threshold and
                    not recently_blinked):  # 避免将刚计为眨眼的状态又计为长闭眼
                    # 验证这不是检测错误导致的误判
                    # 检查是否持续检测到面部
                    if not hasattr(self, 'face_detected') or self.face_detected:
                        self.long_eye_close_count += 1
                        self.last_long_close_time = now
                        # 根据闭合持续时间分类记录
                        close_type = "严重" if close_duration > 3.0 else "中度" if close_duration > 2.0 else "轻度"
                        self.log.info(f"检测到{close_type}长闭眼: 持续时间={close_duration:.2f}s, 帧数={getattr(self, 'current_long_close_frames', 0)}, " 
                                  f"类型={'双眼' if getattr(self, 'current_long_close_both_eyes', False) else '单眼'}, 总计数={self.long_eye_close_count}")
                        
                        # 重置当前长闭眼状态，准备检测下一次
                        self.long_eye_close_start_time = now  # 不立即重置，允许连续的长闭眼被分段检测
                        self.current_long_close_frames = 0
            else:
                # 眼睛睁开，重置长闭眼检测状态
                self.long_eye_close_start_time = None
                # 重置而不是删除属性，避免属性不存在的问题
                if hasattr(self, 'current_long_close_frames'):
                    self.current_long_close_frames = 0
                if hasattr(self, 'current_long_close_both_eyes'):
                    self.current_long_close_both_eyes = False
            
            # 重置状态标志
            self.is_fatigue = False
            
            # 客观疲劳判定标准（简单明确）
            if time_elapsed > 10:  # 系统运行10秒后开始判定
                # 1. 计算基本指标
                blink_rate = (self.blink_count / time_elapsed) * 60 if time_elapsed > 0 else 0
                
                # 2. 客观疲劳评分（0-100分）- 调整权重使判定更合理
                # 初始化评分更新时间变量
                if not hasattr(self, 'last_score_update_time'):
                    self.last_score_update_time = 0
                
                now = time.time()
                score_update_interval = 2.0  # 评分更新间隔：2秒
                
                # 周期性更新疲劳评分，避免评分变化过于频繁
                if now - self.last_score_update_time >= score_update_interval:
                    self.last_score_update_time = now
                    self.fatigue_score = 0  # 使用实例变量存储评分，方便返回
                    
                    # 基于科学研究的客观疲劳指标
                    # 重新平衡各指标权重，使评分更符合实际疲劳程度
                    # a. 眨眼频率异常 - 增加权重并细化区间
                    if blink_rate < 3:
                        self.fatigue_score += 10  # 极低眨眼频率（严重疲劳指标）
                    elif blink_rate < 6:
                        self.fatigue_score += 5   # 低眨眼频率
                    elif blink_rate > 35:
                        self.fatigue_score += 8   # 高频眨眼也表示疲劳
                    elif blink_rate > 25:
                        self.fatigue_score += 4   # 略高眨眼频率
                    
                    # b. 哈欠次数 - 调整权重并考虑累积效应
                    self.fatigue_score += self.yawn_count * 18  # 增加哈欠权重
                    
                    # c. 长闭眼检测 - 最严重的疲劳指标，提高权重
                    self.fatigue_score += self.long_eye_close_count * 30  # 增加长闭眼权重
                    
                    # d. 随时间累积的疲劳因素
                    # 驾驶时间越长，疲劳倾向越高
                    if time_elapsed > 300:  # 5分钟以上
                        self.fatigue_score += min(20, int(time_elapsed / 60))  # 最多加20分
                
                # 3. 客观疲劳状态判定 - 四级疲劳分级系统
                # 优化的疲劳等级判定：重新调整阈值和滞回区间，使评分与等级关系更合理
                # 提高系统对真实疲劳状态的识别能力，同时保持状态稳定性
                if hasattr(self, 'prev_fatigue_level'):
                    # 已有前一状态，应用优化的滞回逻辑
                    if self.prev_fatigue_level == 0:
                        # 从清醒到轻度疲劳的判定更敏感
                        if self.fatigue_score >= 40:  # 降低阈值，提高早期检测能力
                            self.is_fatigue = True
                            self.fatigue_level = 1
                        else:
                            self.fatigue_level = 0
                    elif self.prev_fatigue_level == 1:
                        # 轻度疲劳的滞回区间优化
                        if self.fatigue_score >= 65:  # 降低阈值，更合理的区间划分
                            self.is_fatigue = True
                            self.fatigue_level = 2
                        elif self.fatigue_score < 30:  # 调整回退阈值
                            self.fatigue_level = 0
                            self.is_fatigue = False
                        else:
                            # 保持轻度疲劳
                            self.is_fatigue = True
                            self.fatigue_level = 1
                    elif self.prev_fatigue_level == 2:
                        # 中度疲劳的滞回区间优化
                        if self.fatigue_score >= 85:  # 调整阈值
                            self.is_fatigue = True
                            self.fatigue_level = 3
                        elif self.fatigue_score < 50:  # 调整回退阈值
                            self.is_fatigue = True
                            self.fatigue_level = 1
                        else:
                            # 保持中度疲劳
                            self.is_fatigue = True
                            self.fatigue_level = 2
                    else:  # prev_fatigue_level == 3
                        # 严重疲劳状态的稳定性优化
                        if self.fatigue_score < 70:  # 调整回退阈值，保持适当的稳定性
                            self.is_fatigue = True
                            self.fatigue_level = 2
                        else:
                            # 保持严重疲劳
                            self.is_fatigue = True
                            self.fatigue_level = 3
                else:
                    # 首次判定，使用更合理的初始阈值
                    if self.fatigue_score >= 85:
                        self.is_fatigue = True
                        self.fatigue_level = 3
                    elif self.fatigue_score >= 65:
                        self.is_fatigue = True
                        self.fatigue_level = 2
                    elif self.fatigue_score >= 40:
                        self.is_fatigue = True
                        self.fatigue_level = 1
                    else:
                        self.fatigue_level = 0
                
                # 保存当前等级作为下次判定的参考
                self.prev_fatigue_level = self.fatigue_level
                
                # 转换为0-1范围的疲劳指数供系统使用
                self.fatigue_index = min(1.0, self.fatigue_score / 100)
                
                self.log.info(f"疲劳检测: 眨眼率={blink_rate:.1f}/min, 哈欠={self.yawn_count}, 长闭眼={self.long_eye_close_count}, 疲劳分数={self.fatigue_score}, 等级={self.fatigue_level})")
            else:
                # 初始阶段默认清醒
                self.fatigue_index = 0.0
                self.fatigue_level = 0

            # 周期复位
            if now - self.last_period_t > P.PERIOD_SEC:
                self.reset_counters()
                # 确保复位后状态正确同步
                self.both_closed_start_time = None
                self.blink_sequence = "open"
                self.yawn_sequence = "closed"
                self.yawn_open_start_time = None

            # 改进的疲劳判定逻辑：确保只有达到明确阈值才切换状态
            long_closed = (self.left_closed_frames >= self.long_eye_closed_frames) or \
                          (self.right_closed_frames >= self.long_eye_closed_frames)

            # 严格的疲劳判断标准：基于多维度指标和阈值
            fatigue = False
            
            # 1. 长闭眼直接判定为疲劳（危险情况）
            if long_closed:
                fatigue = True
            # 2. 疲劳指数达到阈值，同时结合哈欠次数作为辅助判断
            elif self.fatigue_index >= 0.8 and self.yawn_count >= 1:
                fatigue = True
            # 3. 哈欠次数过多也直接判定为疲劳
            elif self.yawn_count >= 3:
                fatigue = True

            # 状态切换需要稳定：只有在持续满足条件时才切换状态
            # 防止短暂波动导致状态频繁切换
            if fatigue and not self.prev_fatigue_status:
                # 从清醒切换到疲劳，需要确认是稳定状态
                if not hasattr(self, '_fatigue_candidate_start'):
                    self._fatigue_candidate_start = now
                    status = "清醒"  # 尚未确认，保持清醒状态
                elif now - self._fatigue_candidate_start >= 0.5:  # 至少持续0.5秒才确认疲劳
                    status = "疲劳"
                    self.prev_fatigue_status = True
                    self.fatigue_start_time = now
                    self._fatigue_candidate_start = None
                else:
                    status = "清醒"  # 尚未确认，保持清醒状态
            elif fatigue:
                # 已经是疲劳状态，保持不变
                status = "疲劳"
                self.prev_fatigue_status = True
                self._fatigue_candidate_start = None
                # 确保fatigue_start_time有值
                if self.fatigue_start_time is None:
                    self.fatigue_start_time = now
            else:
                # 清醒状态
                status = "清醒"
                self.prev_fatigue_status = False
                self.fatigue_start_time = None
                if hasattr(self, '_fatigue_candidate_start'):
                    delattr(self, '_fatigue_candidate_start')

        # ---------- 语音播报（按显式冷却 + 全局间隔 + 事件优先级） ----------
        try:
            did_speak = False
            if (now - self.voice_last_any) >= P.VOICE_GLOBAL_GAP:
                # 长闭眼优先
                if long_closed and (now - self.voice_last_longclose) >= P.VOICE_COOLDOWN_LONGCLOSE:
                    if speak_with_interval("检测到长时间闭眼，注意安全驾驶。", seconds=P.VOICE_COOLDOWN_LONGCLOSE, urgency='high'):
                        self.voice_last_longclose = now
                        self.voice_last_any = now
                        self.last_status_spoken = "疲劳"
                        did_speak = True
                        self.log.info(
                            f"[VOICE] 长闭眼播报 left_frames={self.left_closed_frames} right_frames={self.right_closed_frames}")
                # 哈欠其次
                elif (self.mouth_open_frames >= self.yawn_open_min_frames) and (
                        (now - self.voice_last_yawn) >= P.VOICE_COOLDOWN_YAWN):
                    if speak_with_interval("检测到哈欠，请注意休息。", seconds=P.VOICE_COOLDOWN_YAWN, urgency='normal'):
                        self.voice_last_yawn = now
                        self.voice_last_any = now
                        did_speak = True
                        self.log.info(f"[VOICE] 哈欠播报 mouth_open_frames={self.mouth_open_frames}")
                # 疲劳最后（避免与长闭眼/哈欠并发拥堵）
                elif (status == "疲劳") and ((now - self.voice_last_fatigue) >= P.VOICE_COOLDOWN_FATIGUE):
                    # 根据疲劳等级设置不同的提醒信息和紧急程度
                    if self.fatigue_level == 1:
                        msg = "请注意休息，您已出现轻度疲劳症状"
                        urgency = 'low'
                    elif self.fatigue_level == 2:
                        msg = "疲劳程度增加，建议适当休息"
                        urgency = 'normal'
                    elif self.fatigue_level == 3:
                        msg = "危险！您已处于严重疲劳状态，请立即停车休息！"
                        urgency = 'high'
                    else:
                        msg = "检测到疲劳驾驶，请立即休息。"
                        urgency = 'normal'
                    
                    if speak_with_interval(msg, seconds=P.VOICE_COOLDOWN_FATIGUE, urgency=urgency):
                        self.voice_last_fatigue = now
                        self.voice_last_any = now
                        self.last_status_spoken = "疲劳"
                        did_speak = True
                        self.log.info(
                            f"[VOICE] 疲劳播报 level={self.fatigue_level} blink={self.blink_count} yawn={self.yawn_count} long_closed={long_closed}")

            # 清醒提示：仅在从疲劳切换到清醒的瞬间提示一次（且本帧未播其他事件）
            if (status == "清醒") and (self.last_status_spoken == "疲劳") and not did_speak:
                if speak_with_interval("状态恢复正常。", seconds=4.0):
                    self.last_status_spoken = "清醒"
                    self.voice_last_any = now
                    self.log.info("[VOICE] 清醒播报（从疲劳恢复）")
        except Exception as e:
            self.log.debug(f"[VOICE] 播报异常: {e}")

        # 返回状态快照，避免外部直接访问内部状态
        # 使用状态锁确保返回数据的一致性和线程安全
        with self.state_lock:
            # 确保使用锁内计算的long_closed值，避免重复计算和线程安全问题
            # 构建完整的状态信息
            return {
                "blink": self.blink_count,         # 眨眼计数
                "yawn": self.yawn_count,           # 哈欠计数
                "time": int(time.time() - self.time_start),  # 检测时长（秒）
                "fatigue_index": round(self.fatigue_index, 2),  # 疲劳指数（0-1）
                "long_closed": long_closed,        # 是否长闭眼
                "fatigue_level": self.fatigue_level,  # 疲劳等级
                "fatigue_score": min(100, self.fatigue_score)  # 疲劳分数
            }

    # ------------ 外部接口 ------------
    def process_frame(self, frame):
        """处理一帧图像，返回处理后的帧和状态信息"""
        try:
            # 检测面部状态
            frame_det, flags = self.detect_facial_states(frame)
            
            # 更新疲劳状态
            stat = self.update_fatigue_status(flags)
            
            # 绘制状态信息（使用状态锁保护last_flags的访问）
            with self.state_lock:
                # 仅在show_text为True时（视频完整处理模式）显示文字
                if frame_det is not None and self.show_text:
                    # 获取疲劳等级文本和颜色
                    fatigue_levels = {0: '清醒', 1: '轻度疲劳', 2: '中度疲劳', 3: '严重疲劳'}
                    fatigue_colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 0, 255)}
                    
                    level_text = fatigue_levels.get(stat.get('fatigue_level', 0), '未知')
                    level_color = fatigue_colors.get(stat.get('fatigue_level', 0), (255, 255, 255))
                    
                    # 绘制疲劳等级
                    self._draw_text(frame_det, f'疲劳等级: {level_text}', (10, 30), level_color, font_scale=0.8, thickness=2)
                    
                    # 绘制疲劳分数
                    score_text = f'疲劳分数: {stat.get("fatigue_score", 0)}'
                    self._draw_text(frame_det, score_text, (10, 70), (255, 255, 0), font_scale=0.8, thickness=2)
                    
                    # 绘制检测统计信息
                    stats_text = f'眨眼: {stat.get("blink", 0)} 哈欠: {stat.get("yawn", 0)} 时长: {stat.get("time", 0)}s'
                    self._draw_text(frame_det, stats_text, (10, 110), (255, 255, 255), font_scale=0.7, thickness=1)
                    
                    # 如果是长闭眼，显示警告信息
                    if stat.get('long_closed', False):
                        self._draw_text(frame_det, '⚠️ 长闭眼警告!', (10, 150), (0, 0, 255), font_scale=0.9, thickness=2)
            
            # 验证状态信息完整性 - 作为最后一道防线
            # 确保stat始终是字典类型
            if not isinstance(stat, dict):
                self.log.error("状态信息不是字典类型")
                # 初始化为空字典
                stat = {}
                
            # 优化：确保返回的状态包含所有必要字段，特别是状态判断相关的字段
            required_fields = ["blink", "yawn", "time", "fatigue_index", "long_closed", "fatigue_level", "fatigue_score"]
            for field in required_fields:
                if field not in stat:
                    self.log.warning(f"状态信息缺少字段: {field}")
                    if field == "blink":
                        stat[field] = getattr(self, 'blink_count', 0)
                    elif field == "yawn":
                        stat[field] = getattr(self, 'yawn_count', 0)

                    elif field == "time":
                        stat[field] = int(time.time() - getattr(self, 'time_start', time.time()))
                    elif field == "fatigue_index":
                        stat[field] = getattr(self, 'fatigue_index', 0.0)
                    elif field == "long_closed":
                        stat[field] = False
                    elif field == "fatigue_level":
                        stat[field] = getattr(self, 'fatigue_level', 0)

                    elif field == "fatigue_score":
                        stat[field] = getattr(self, 'fatigue_score', 0)
            
            # 优化：如果检测到长闭眼，立即更新状态
            if stat.get("long_closed", False) and stat.get("fatigue_level", 0) < 3:
                stat["fatigue_level"] = 3
                self.log.info(f"长闭眼强制更新状态为严重疲劳")
            
            # 返回处理后的帧和状态副本
            return frame_det, stat.copy()
        except Exception as e:
            logging.getLogger(__name__).error(f"帧处理失败: {e}", exc_info=True)
            # 异常情况下返回安全的默认状态
            with self.state_lock:
                now = time.time()
                fallback = {
                    "blink": self.blink_count,
                    "yawn": self.yawn_count,
                    "time": int(now - self.time_start),
                    "fatigue_index": 0.0,
                    "long_closed": False,
                    "fatigue_level": 0,
                    "fatigue_score": 0
                }
            return frame, fallback
    
    def get_current_state(self):
        """获取当前系统状态的快照，用于外部组件查询"""
        with self.state_lock:
            return {
                "blink": self.blink_count,  # 眨眼次数
                "yawn": self.yawn_count,    # 哈欠次数
                "fatigue_level": self.fatigue_level,  # 疲劳等级
                "fatigue_score": min(100, self.fatigue_score),  # 疲劳分数，确保不超过100
                "fatigue_index": round(self.fatigue_index, 2)  # 疲劳指数，保留两位小数
            }