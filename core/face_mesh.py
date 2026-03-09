# path: core/face_mesh.py
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import time

try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    mp = None
    MP_OK = False

# 增强的Face Mesh索引定义 - 使用更完整的关键点集
LEFT_EYE_IDX = [33, 133, 159, 145, 160, 144, 158, 153, 147, 163, 143, 148]  # 扩展的左眼关键点
RIGHT_EYE_IDX = [362, 263, 386, 374, 385, 380, 384, 398, 382, 373, 387, 372]  # 扩展的右眼关键点
MOUTH_OUTER_IDX = [61, 291, 0, 267, 269, 270, 409, 405, 321, 375, 146, 91]  # 更完整的嘴巴外部轮廓
MOUTH_INNER_IDX = [13, 14, 78, 308, 310, 311, 312, 136, 268, 317, 318, 402, 403]  # 更完整的嘴巴内部轮廓
FACE_OUTLINE_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365]  # 部分人脸轮廓点

# 增加眼睛区域额外关键点以提高检测精度
LEFT_EYEBROW_IDX = [276, 283, 282, 295, 285]  # 左眉毛关键点
RIGHT_EYEBROW_IDX = [46, 53, 52, 65, 55]  # 右眉毛关键点

# 眼镜检测相关关键点 - 用于检测眼镜框
LEFT_GLASSES_IDX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  # 左眼区域及鼻梁关键点
RIGHT_GLASSES_IDX = [362, 398, 384, 385, 386, 387, 466, 263, 373, 374, 380, 381, 382, 362, 390, 249]  # 右眼区域及鼻梁关键点
BRIDGE_IDX = [168, 6, 197, 195, 5, 4, 1, 199, 200, 201]  # 鼻梁区域关键点

def _to_pixel(lmks, w: int, h: int) -> np.ndarray:
    """优化的坐标转换函数，增加边界检查和精度控制"""
    pts = []
    for l in lmks:
        x = int(np.clip(l.x * w, 0, w - 1))
        y = int(np.clip(l.y * h, 0, h - 1))
        pts.append([x, y])
    return np.asarray(pts, dtype=np.int32)

def _bbox_from_points(pts: np.ndarray, margin: float, w: int, h: int, region_type: str = 'general') -> Optional[Tuple[int, int, int, int]]:
    """增强的包围框生成函数，针对不同区域类型进行优化"""
    if pts is None or pts.size == 0:
        return None
    
    # 添加异常值过滤，提高鲁棒性
    if len(pts) > 3:  # 只有当点足够多时才进行过滤
        # 计算每个点到中心点的距离
        center = np.mean(pts, axis=0)
        distances = np.sqrt(np.sum((pts - center) ** 2, axis=1))
        # 过滤掉距离超过平均值2倍的点（可能是异常点）
        mean_dist = np.mean(distances)
        valid_pts = pts[distances < mean_dist * 2]
        if len(valid_pts) > 0:
            pts = valid_pts
    
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    
    # 根据区域类型调整margin
    if region_type == 'eye':
        # 眼睛区域：进一步增加水平边距，优化垂直边距，确保眼睛能被正确定位
        adjusted_margin = {'h': margin * 1.2, 'v': margin * 1.0}
        bw2 = bw * (1.0 + adjusted_margin['h'])
        bh2 = bh * (1.0 + adjusted_margin['v'])
    elif region_type == 'mouth':
        # 嘴巴区域：适度扩大
        adjusted_margin = {'h': margin * 0.9, 'v': margin * 1.1}
        bw2 = bw * (1.0 + adjusted_margin['h'])
        bh2 = bh * (1.0 + adjusted_margin['v'])
    else:
        # 通用区域
        bw2 = bw * (1.0 + margin)
        bh2 = bh * (1.0 + margin)
    
    # 计算新的边界框坐标
    x1n = int(np.clip(cx - bw2 / 2, 0, w - 1))
    y1n = int(np.clip(cy - bh2 / 2, 0, h - 1))
    x2n = int(np.clip(cx + bw2 / 2, 0, w - 1))
    y2n = int(np.clip(cy + bh2 / 2, 0, h - 1))
    
    # 确保框有最小尺寸
    min_size = 15 if region_type == 'eye' else 20
    if (x2n - x1n) < min_size:
        half_min = min_size // 2
        x1n = max(0, int(cx) - half_min)
        x2n = min(w - 1, int(cx) + half_min)
    if (y2n - y1n) < min_size:
        half_min = min_size // 2
        y1n = max(0, int(cy) - half_min)
        y2n = min(h - 1, int(cy) + half_min)
    
    # 最终有效性检查
    if x2n <= x1n or y2n <= y1n:
        return None
    
    return (x1n, y1n, x2n, y2n)

class FaceMeshHelper:
    """
    增强的Face Mesh助手类，提供更稳定、准确的面部特征检测
    返回：眼/嘴包围框、对应关键点子集、以及全脸468点像素坐标（all_pts）和置信度评分
    """
    def __init__(self, max_faces: int = 1):
        self.ready = False
        if not MP_OK:
            print("Warning: MediaPipe not available. Face mesh functionality will be disabled.")
            return
        
        # 增强的MediaPipe配置，提高检测灵敏度和稳定性
        self.mp_face_mesh = mp.solutions.face_mesh
        # 移除smooth_landmarks参数以兼容旧版本MediaPipe
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=0.4,  # 降低置信度阈值以提高检出率
            min_tracking_confidence=0.4    # 降低跟踪置信度以提高稳定性
        )
        
        # 状态管理变量
        self.ready = True
        self.last_face_detected = False
        self.detection_streak = 0
        self.non_detection_streak = 0
        self.max_non_detection = 3  # 连续未检测到人脸的最大帧数
        
        # 历史记录缓存，用于平滑和恢复
        self.prev_all_pts = None
        self.prev_left_eye_box = None
        self.prev_right_eye_box = None
        self.prev_mouth_box = None
        self.confidence_history = []
        self.max_history_size = 5
        self.last_success_time = 0
        
        # 自适应参数
        self.adaptive_conf_threshold = 0.4
        self.min_valid_points = 4  # 区域有效点的最小数量
        
        # 眼镜检测相关参数
        self.is_wearing_glasses = False  # 是否戴眼镜标记
        self.glasses_confidence = 0.0  # 眼镜检测置信度
        self.glasses_history = []  # 眼镜检测历史记录

    def close(self):
        """关闭并释放资源"""
        if self.ready and self.mesh:
            self.mesh.close()
            self.ready = False
        # 清除历史数据
        self.prev_all_pts = None
        self.prev_left_eye_box = None
        self.prev_right_eye_box = None
        self.prev_mouth_box = None
        self.confidence_history = []
    
    def _calculate_confidence(self, landmarks: np.ndarray, region_indices: List[int]) -> float:
        """计算特定区域的关键点检测置信度"""
        if landmarks is None or len(landmarks) == 0:
            return 0.0
        
        # 检查点是否都在图像合理范围内
        valid_points = 0
        total_points = len(region_indices)
        
        for idx in region_indices:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                # 允许点稍微超出边界（5像素）以提高鲁棒性
                if -5 <= x < self.w + 5 and -5 <= y < self.h + 5:
                    valid_points += 1
        
        # 计算有效点的比例作为置信度
        confidence = valid_points / total_points if total_points > 0 else 0.0
        
        # 额外检查：点不应过于集中
        if valid_points > 3:
            region_pts = landmarks[region_indices[:valid_points]]
            if np.std(region_pts[:, 0]) < 2 or np.std(region_pts[:, 1]) < 2:
                # 点过于集中，可能是错误检测
                confidence *= 0.5
        
        return confidence
    
    def _smooth_box(self, current_box: Tuple[int, int, int, int], previous_box: Tuple[int, int, int, int], alpha: float = 0.4) -> Tuple[int, int, int, int]:
        """增强的平滑包围框方法，更强的平滑效果以减少抖动"""
        if previous_box is None or current_box is None:
            return current_box
        
        # 降低当前框权重，增加历史框权重，提高稳定性
        x1 = int(alpha * current_box[0] + (1 - alpha) * previous_box[0])
        y1 = int(alpha * current_box[1] + (1 - alpha) * previous_box[1])
        x2 = int(alpha * current_box[2] + (1 - alpha) * previous_box[2])
        y2 = int(alpha * current_box[3] + (1 - alpha) * previous_box[3])
        
        # 确保平滑后的框有效
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.w - 1, x2)
        y2 = min(self.h - 1, y2)
        
        if x2 <= x1 or y2 <= y1:
            return previous_box
        
        return (x1, y1, x2, y2)
    
    def _validate_face_structure(self, all_pts: np.ndarray) -> bool:
        """验证人脸结构是否合理，避免错误检测"""
        if all_pts is None or len(all_pts) < max(FACE_OUTLINE_IDX):
            return False
        
        # 检查眼睛之间的距离是否合理
        left_eye_center = np.mean(all_pts[LEFT_EYE_IDX[:6]], axis=0)
        right_eye_center = np.mean(all_pts[RIGHT_EYE_IDX[:6]], axis=0)
        eye_distance = np.sqrt(np.sum((left_eye_center - right_eye_center) ** 2))
        
        # 检查眼睛到嘴巴的距离是否合理
        mouth_center = np.mean(all_pts[MOUTH_OUTER_IDX[:4]], axis=0)
        eye_mouth_dist = np.mean([
            np.sqrt(np.sum((left_eye_center - mouth_center) ** 2)),
            np.sqrt(np.sum((right_eye_center - mouth_center) ** 2))
        ])
        
        # 人脸比例检查：眼睛间距与眼睛到嘴巴距离的比例应在合理范围内
        if eye_distance > 0:
            ratio = eye_mouth_dist / eye_distance
            if not 1.0 < ratio < 3.0:  # 合理的人脸比例范围
                return False
        
        # 检查眼睛是否水平
        eye_y_diff = abs(left_eye_center[1] - right_eye_center[1])
        if eye_y_diff > eye_distance * 0.5:  # 眼睛不应过于倾斜
            return False
        
        return True

    def infer(self, frame_bgr: np.ndarray) -> Dict:
        """
        增强的推理方法，优化检测稳定性和准确性
        返回包含所有检测结果和置信度的字典
        """
        self.h, self.w = frame_bgr.shape[:2] if frame_bgr is not None else (0, 0)
        
        # 基本有效性检查
        if not self.ready or frame_bgr is None or frame_bgr.size == 0:
            return self._get_empty_result()
        
        # 转换为RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 自适应调整检测参数 - 更严格的阈值设置
        if len(self.confidence_history) > 0:
            avg_conf = sum(self.confidence_history) / len(self.confidence_history)
            # 根据历史置信度动态调整阈值 - 提高下限以增加稳定性
            if avg_conf < 0.7:
                self.adaptive_conf_threshold = max(0.35, self.adaptive_conf_threshold - 0.005)  # 提高下限并减慢调整速度
            else:
                self.adaptive_conf_threshold = min(0.5, self.adaptive_conf_threshold + 0.01)
        
        # 处理图像
        try:
            res = self.mesh.process(rgb)
            
            # 检查是否检测到人脸
            if not res.multi_face_landmarks:
                self.last_face_detected = False
                self.non_detection_streak += 1
                self.detection_streak = 0
                
                # 如果有最近的有效检测，可以尝试恢复
                current_time = time.time()
                if (self.prev_all_pts is not None and 
                    self.non_detection_streak < self.max_non_detection and
                    current_time - self.last_success_time < 1.0):  # 1秒内的结果可以恢复
                    
                    # 轻微调整历史框位置，模拟微小移动
                    adjusted_left_eye = self._adjust_box_for_recovery(self.prev_left_eye_box)
                    adjusted_right_eye = self._adjust_box_for_recovery(self.prev_right_eye_box)
                    adjusted_mouth = self._adjust_box_for_recovery(self.prev_mouth_box)
                    
                    return {
                        "left_eye_box": adjusted_left_eye,
                        "right_eye_box": adjusted_right_eye,
                        "mouth_box": adjusted_mouth,
                        "left_eye_pts": None,  # 恢复时不提供精确点
                        "right_eye_pts": None,
                        "mouth_pts": None,
                        "all_pts": self.prev_all_pts,
                        "confidence": 0.5  # 标记为恢复的低置信度结果
                    }
                
                # 返回空结果
                return self._get_empty_result()
            
            # 更新检测状态
            self.last_face_detected = True
            self.detection_streak += 1
            self.non_detection_streak = 0
            self.last_success_time = time.time()
            
            # 获取第一个人脸的关键点
            face_landmarks = res.multi_face_landmarks[0].landmark
            
            # 转换为像素坐标
            all_pts = _to_pixel(face_landmarks, self.w, self.h)
            
            # 增强的人脸结构验证 - 增加相似度检查
            if not self._validate_face_structure(all_pts):
                # 检查当前检测结果与历史结果的相似度
                if self.prev_all_pts is not None and self.detection_streak > 3:
                    # 计算与历史关键点的欧氏距离
                    if len(all_pts) > 0 and len(self.prev_all_pts) > 0:
                        # 只比较眼睛区域的关键点
                        min_len = min(len(all_pts), len(self.prev_all_pts))
                        eye_indices = set(LEFT_EYE_IDX[:6] + RIGHT_EYE_IDX[:6])
                        valid_indices = [i for i in eye_indices if i < min_len]
                        
                        if len(valid_indices) > 4:
                            current_eye_pts = all_pts[valid_indices]
                            prev_eye_pts = self.prev_all_pts[valid_indices]
                            
                            # 计算平均距离
                            avg_distance = np.mean(np.sqrt(np.sum((current_eye_pts - prev_eye_pts) ** 2, axis=1)))
                            
                            # 如果距离过大，可能是错误检测，使用历史数据
                            if avg_distance > 15:  # 过大的移动被视为异常
                                return {
                                    "left_eye_box": self.prev_left_eye_box,
                                    "right_eye_box": self.prev_right_eye_box,
                                    "mouth_box": self.prev_mouth_box,
                                    "left_eye_pts": None,
                                    "right_eye_pts": None,
                                    "mouth_pts": None,
                                    "all_pts": self.prev_all_pts,
                                    "confidence": 0.7  # 高置信度的历史结果，因为检测到异常移动
                                }
                # 如果结构不合理，尝试使用历史数据
                if self.prev_all_pts is not None:
                    return {
                        "left_eye_box": self.prev_left_eye_box,
                        "right_eye_box": self.prev_right_eye_box,
                        "mouth_box": self.prev_mouth_box,
                        "left_eye_pts": None,
                        "right_eye_pts": None,
                        "mouth_pts": None,
                        "all_pts": self.prev_all_pts,
                        "confidence": 0.6  # 中等置信度的历史结果
                    }
                return self._get_empty_result()
            
            # 计算各区域的置信度
            left_eye_conf = self._calculate_confidence(all_pts, LEFT_EYE_IDX)
            right_eye_conf = self._calculate_confidence(all_pts, RIGHT_EYE_IDX)
            mouth_conf = self._calculate_confidence(all_pts, MOUTH_OUTER_IDX)
            
            # 综合置信度
            overall_conf = (left_eye_conf + right_eye_conf + mouth_conf) / 3.0
            self.confidence_history.append(overall_conf)
            if len(self.confidence_history) > self.max_history_size:
                self.confidence_history.pop(0)
            
            # 提取关键点子集
            left_eye_pts = all_pts[LEFT_EYE_IDX, :] if len(all_pts) > max(LEFT_EYE_IDX) else None
            right_eye_pts = all_pts[RIGHT_EYE_IDX, :] if len(all_pts) > max(RIGHT_EYE_IDX) else None
            
            # 组合嘴巴的内外关键点
            if len(all_pts) > max(max(MOUTH_OUTER_IDX), max(MOUTH_INNER_IDX)):
                mouth_outer = all_pts[MOUTH_OUTER_IDX, :]
                mouth_inner = all_pts[MOUTH_INNER_IDX, :]
                mouth_pts = np.vstack([mouth_outer, mouth_inner])
            else:
                mouth_pts = None
            
            # 生成包围框 - 使用区域特定的参数
            left_eye_box = None
            right_eye_box = None
            mouth_box = None
            
            # 只在关键点足够且置信度足够时生成框
            if left_eye_pts is not None and left_eye_conf > self.adaptive_conf_threshold:
                left_eye_box = _bbox_from_points(left_eye_pts, margin=0.3, w=self.w, h=self.h, region_type='eye')
            
            if right_eye_pts is not None and right_eye_conf > self.adaptive_conf_threshold:
                right_eye_box = _bbox_from_points(right_eye_pts, margin=0.3, w=self.w, h=self.h, region_type='eye')
            
            if mouth_pts is not None and mouth_conf > self.adaptive_conf_threshold:
                mouth_box = _bbox_from_points(mouth_pts, margin=0.35, w=self.w, h=self.h, region_type='mouth')
            
            # 使用历史框进行平滑，提高稳定性
            if self.prev_left_eye_box is not None and left_eye_box is not None:
                left_eye_box = self._smooth_box(left_eye_box, self.prev_left_eye_box)
            if self.prev_right_eye_box is not None and right_eye_box is not None:
                right_eye_box = self._smooth_box(right_eye_box, self.prev_right_eye_box)
            if self.prev_mouth_box is not None and mouth_box is not None:
                mouth_box = self._smooth_box(mouth_box, self.prev_mouth_box)
            
            # 更新历史数据
            self.prev_all_pts = all_pts
            self.prev_left_eye_box = left_eye_box
            self.prev_right_eye_box = right_eye_box
            self.prev_mouth_box = mouth_box
            
            # 返回结果
            return {
                "left_eye_box": left_eye_box,
                "right_eye_box": right_eye_box,
                "mouth_box": mouth_box,
                "left_eye_pts": left_eye_pts,
                "right_eye_pts": right_eye_pts,
                "mouth_pts": mouth_pts,
                "all_pts": all_pts,
                "confidence": overall_conf,
                "left_eye_conf": left_eye_conf,
                "right_eye_conf": right_eye_conf,
                "mouth_conf": mouth_conf
            }
            
        except Exception as e:
            # 错误处理
            print(f"FaceMesh inference error: {str(e)}")
            self.last_face_detected = False
            self.non_detection_streak += 1
            self.detection_streak = 0
            
            # 尝试返回历史数据
            if self.prev_all_pts is not None and self.non_detection_streak < 2:
                return {
                    "left_eye_box": self.prev_left_eye_box,
                    "right_eye_box": self.prev_right_eye_box,
                    "mouth_box": self.prev_mouth_box,
                    "left_eye_pts": None,
                    "right_eye_pts": None,
                    "mouth_pts": None,
                    "all_pts": self.prev_all_pts,
                    "confidence": 0.4  # 低置信度的错误恢复
                }
            
            # 否则返回空结果
            return self._get_empty_result()
    
    def _get_empty_result(self) -> Dict:
        """返回空结果字典"""
        return {
            "left_eye_box": None, "right_eye_box": None, "mouth_box": None,
            "left_eye_pts": None, "right_eye_pts": None, "mouth_pts": None,
            "all_pts": None,
            "confidence": 0.0
        }
    
    def _adjust_box_for_recovery(self, box: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """稳定的框位置恢复方法，移除随机偏移以避免抖动"""
        if box is None:
            return None
        
        # 移除随机偏移，保持框位置稳定
        # 只进行边界检查，不添加随机偏移
        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(self.w - 1, box[2])
        y2 = min(self.h - 1, box[3])
        
        return (x1, y1, x2, y2)
    
    def _detect_glasses(self, all_pts: np.ndarray) -> Tuple[bool, float]:
        """
        检测用户是否佩戴眼镜
        使用鼻梁和眼睛区域的关键点几何特征进行分析
        """
        if all_pts is None:
            return False, 0.0
        
        # 确保所有需要的关键点都在范围内
        max_idx = max(max(LEFT_GLASSES_IDX), max(RIGHT_GLASSES_IDX), max(BRIDGE_IDX))
        if len(all_pts) <= max_idx:
            return False, 0.0
        
        # 获取眼镜检测区域的关键点
        left_glasses_pts = all_pts[LEFT_GLASSES_IDX]
        right_glasses_pts = all_pts[RIGHT_GLASSES_IDX]
        bridge_pts = all_pts[BRIDGE_IDX]
        
        # 计算鼻梁区域的特征 - 眼镜通常会在这个区域产生边缘
        confidence = 0.0
        glasses_detected = False
        
        try:
            # 检查眼睛区域和鼻梁之间的几何关系
            # 1. 计算左右眼镜区域的高度差异
            left_height = np.max(left_glasses_pts[:, 1]) - np.min(left_glasses_pts[:, 1])
            right_height = np.max(right_glasses_pts[:, 1]) - np.min(right_glasses_pts[:, 1])
            height_ratio = left_height / right_height if right_height > 0 else 0
            
            # 2. 计算鼻梁区域的特征
            bridge_top = np.mean(bridge_pts[:3], axis=0)
            bridge_bottom = np.mean(bridge_pts[-3:], axis=0)
            bridge_length = np.sqrt(np.sum((bridge_top - bridge_bottom) **2))
            
            # 3. 检查眼睛外侧点到鼻梁的距离
            left_outside = all_pts[7]  # 左眼外侧点
            right_outside = all_pts[249]  # 右眼外侧点
            eye_to_bridge_dist = np.sqrt(np.sum((left_outside - right_outside) **2))
            
            # 计算置信度分数（简化版）
            if 0.8 < height_ratio < 1.2:  # 左右眼镜区域高度相似
                confidence += 0.3
            
            if bridge_length > 10:  # 鼻梁区域有足够长度
                confidence += 0.2
            
            # 检查关键点的一致性 - 眼镜会影响关键点分布
            if len(left_glasses_pts) > 0 and len(right_glasses_pts) > 0:
                left_std_x = np.std(left_glasses_pts[:, 0])
                right_std_x = np.std(right_glasses_pts[:, 0])
                
                # 如果x方向分布更广泛，可能有眼镜
                if left_std_x > 5 and right_std_x > 5:
                    confidence += 0.3
            
            # 根据置信度判断是否戴眼镜
            glasses_detected = confidence > 0.5
            
            # 更新历史记录
            self.glasses_history.append(confidence)
            if len(self.glasses_history) > 5:
                self.glasses_history.pop(0)
                
            # 使用历史平均进行平滑
            avg_confidence = sum(self.glasses_history) / len(self.glasses_history)
            self.is_wearing_glasses = avg_confidence > 0.5
            self.glasses_confidence = avg_confidence
            
            return self.is_wearing_glasses, avg_confidence
            
        except Exception as e:
            # 出错时返回默认值
            return False, 0.0
    
    def adjust_parameters_for_glasses(self) -> None:
        """
        根据是否佩戴眼镜调整检测参数
        当检测到眼镜时，修改相关参数以提高检测精度
        """
        if self.is_wearing_glasses:
            # 调整眼睛区域的参数，适应眼镜佩戴情况
            self.adaptive_conf_threshold = 0.35  # 略微降低阈值
            # 可以在这里添加更多针对眼镜的参数调整
        else:
            # 恢复默认参数
            self.adaptive_conf_threshold = 0.4
