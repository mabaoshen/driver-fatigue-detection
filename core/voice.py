# path: core/voice.py
from __future__ import annotations

import threading
import queue
import time
import logging

log = logging.getLogger(__name__)

# 尝试导入两种后端：pyttsx3 与 win32com SAPI
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import win32com.client  # 作为稳定后备
except Exception:
    win32com = None

class _TTSWorker:
    def __init__(self, rate=160, volume=1.0, voice_name=None):
        self.q = queue.Queue()
        self.stop_flag = False
        self._last_text = None
        self._last_time = 0.0
        self._rate = int(rate)
        self._volume = float(volume)
        self._voice_name = voice_name
        self._ready = True  # 线程就绪标志
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        log.info(f"[VOICE] 语音线程就绪 rate={self._rate} volume={self._volume} voice={self._voice_name or 'default'}")

    # 每次播报都新建并销毁引擎，避免复用导致静音
    def _speak_once_pyttsx3(self, text: str, urgency='normal') -> bool:
        if pyttsx3 is None:
            return False
        try:
            eng = pyttsx3.init(driverName="sapi5")
            # 设置属性
            try:
                # 根据紧急程度调整语速
                rate = self._rate
                if urgency == 'high':
                    rate = int(rate * 0.9)  # 降低语速10%
                elif urgency == 'low':
                    rate = int(rate * 1.1)  # 提高语速10%
                
                eng.setProperty("rate", rate)
                eng.setProperty("volume", self._volume)
            except Exception:
                pass
            # 选择语音
            if self._voice_name:
                try:
                    voices = eng.getProperty("voices")
                    for v in voices:
                        name = getattr(v, "name", "") or ""
                        if self._voice_name in name:
                            eng.setProperty("voice", v.id)
                            break
                except Exception:
                    pass
            log.debug(f"[VOICE] pyttsx3 播报: {text} (urgency: {urgency})")
            eng.say(text)
            eng.runAndWait()
            # 结束并释放
            try:
                eng.stop()
            except Exception:
                pass
            return True
        except Exception as e:
            log.warning(f"[VOICE] pyttsx3 播报失败: {e}", exc_info=False)
            return False

    def _speak_once_sapi(self, text: str, urgency='normal') -> bool:
        # 直接调用原生 SAPI 作为稳健后备
        if win32com is None:
            return False
        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            # 设置语速与音量（SAPI: Rate -10~10，Volume 0~100）
            try:
                # 根据紧急程度调整语速
                rate = 2  # 默认值
                if urgency == 'high':
                    rate = 1  # 降低语速
                elif urgency == 'low':
                    rate = 3  # 提高语速
                
                voice.Rate = rate
                voice.Volume = int(max(0, min(100, self._volume * 100)))
            except Exception:
                pass
            # 设置中文语音
            if self._voice_name:
                try:
                    token_cat = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
                    token_cat.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices", False)
                    tokens = voice.GetVoices()
                    for i in range(tokens.Count):
                        tok = tokens.Item(i)
                        name = tok.GetAttribute("Name")
                        if name and self._voice_name in name:
                            voice.Voice = tok
                            break
                except Exception:
                    pass
            log.debug(f"[VOICE] SAPI 播报: {text} (urgency: {urgency})")
            voice.Speak(text)
            return True
        except Exception as e:
            log.error(f"[VOICE] SAPI 播报失败: {e}", exc_info=False)
            return False

    def _speak_once(self, text: str, urgency='normal') -> bool:
        # 优先 pyttsx3，失败则回退 SAPI
        ok = self._speak_once_pyttsx3(text, urgency)
        if ok:
            log.debug(f"[VOICE] 播报完成: {text}")
            return True
        ok2 = self._speak_once_sapi(text, urgency)
        if ok2:
            log.debug(f"[VOICE] 播报完成(回退SAPI): {text}")
            return True
        log.debug(f"[VOICE] 播报失败(两后端均失败): {text}")
        return False

    def _loop(self):
        while not self.stop_flag:
            try:
                text, urgency = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if self._ready and text:
                    self._speak_once(text, urgency)
                else:
                    log.debug("[VOICE] 线程未就绪，丢弃播报。")
            except Exception as e:
                log.error(f"[VOICE] 线程播报异常: {e}", exc_info=True)

    def speak(self, text: str, urgency='normal', dedup_interval=2.0) -> bool:
        """
        返回 True 表示本次文本已入队播报；False 表示被去重、队列防积压或线程未就绪。
        队列防积压：若队列里已有待播文本，则直接丢弃当前请求（保持"一次只播一个"）。
        
        Args:
            text: 要播报的文本
            urgency: 紧急程度，可选值：'low', 'normal', 'high'
            dedup_interval: 重复文本去重间隔（秒）
        """
        if not text or not self._ready:
            return False
        # 队列防积压：只允许同时存在一个待播文本
        try:
            if not self.q.empty():
                return False
        except Exception:
            pass
        now = time.time()
        # 相邻重复文本在 dedup_interval 秒内不再播报
        if self._last_text == text and (now - self._last_time) < dedup_interval:
            return False
        self._last_text = text
        self._last_time = now
        try:
            self.q.put((text, urgency))
            return True
        except Exception:
            return False

    def shutdown(self):
        self.stop_flag = True

_worker = None
_default_voice_name = None  # 当前语音名称

def _get_worker():
    global _worker
    if _worker is None:
        _worker = _TTSWorker(rate=160, volume=1.0, voice_name=_default_voice_name)
    return _worker

def speak(text: str, urgency='normal') -> bool:
    try:
        return _get_worker().speak(text, urgency=urgency)
    except Exception:
        return False

def shutdown():
    try:
        if _worker:
            _worker.shutdown()
    except Exception:
        pass

def speak_with_interval(text: str, seconds: float, urgency='normal') -> bool:
    """
    公开的限频播报接口：相同文本在 seconds 秒内只播报一次。
    返回 True 表示实际入队播报。
    
    Args:
        text: 要播报的文本
        seconds: 去重间隔（秒）
        urgency: 紧急程度，可选值：'low', 'normal', 'high'
    """
    try:
        return _get_worker().speak(text, urgency=urgency, dedup_interval=float(seconds))
    except Exception:
        return False

def set_voice(voice_name: str | None):
    """
    设置中文语音名称（需要系统安装对应语音包）。
    如: voice_name="Microsoft Huihui Desktop - Chinese (Simplified)"
    """
    global _worker, _default_voice_name
    _default_voice_name = voice_name
    try:
        # 重建 worker 以应用新的 voice_name（后台线程内每次播报创建引擎）
        if _worker:
            _worker.shutdown()
        _worker = _TTSWorker(rate=160, volume=1.0, voice_name=_default_voice_name)
    except Exception:
        pass
