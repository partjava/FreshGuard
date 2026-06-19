# -*- coding: utf-8 -*-
"""
AI 对话模块
- 调用本地 ollama qwen2.5vl:7b 进行图片+文字对话
- faster-whisper 语音识别（按住说话）
- edge-tts 语音合成（朗读回复）
"""
import base64
import json
import threading
import tempfile
import os
import asyncio
import urllib.request
import urllib.error
import cv2
import numpy as np
import pyaudio
import wave

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"
WHISPER_MODEL = "small"
_whisper_model = None  # 缓存，避免每次重新加载

SYSTEM_PROMPT = (
    "你是一个智能水果质检机器人的AI助手。"
    "你只负责检测以下5种水果：苹果(apple)、香蕉(banana)、橙子(orange)、葡萄(grape)、芒果(mango)。"
    "当用户提供图片时，图片中只可能出现这5种水果之一，请直接判断它的新鲜程度、外观特征和建议，不要说成其他食物或蔬菜。"
    "当用户提问时，请根据你的知识和对话历史回答，不要强行要求图片。"
    "回答用简短的中文，控制在60字以内，不要使用Markdown格式，不要用**加粗**或其他符号。"
)


def frame_to_base64(frame) -> str:
    """OpenCV帧转base64字符串，压缩到640x480再发"""
    frame = cv2.resize(frame, (640, 480))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def ask_ollama(messages: list) -> str:
    """调用ollama API，返回回复文字"""
    # 转换消息格式：把 image_url 转成 ollama 原生的 images 字段
    ollama_messages = []
    for msg in messages:
        if isinstance(msg["content"], list):
            text = ""
            images = []
            for part in msg["content"]:
                if part["type"] == "text":
                    text = part["text"]
                elif part["type"] == "image_url":
                    # 提取 base64 数据
                    url = part["image_url"]["url"]
                    b64 = url.split(",", 1)[1] if "," in url else url
                    images.append(b64)
            new_msg = {"role": msg["role"], "content": text}
            if images:
                new_msg["images"] = images
            ollama_messages.append(new_msg)
        else:
            ollama_messages.append(msg)

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": ollama_messages,
        "stream": False
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            text = result["message"]["content"].strip()
            # 去掉Markdown格式符号
            import re
            text = re.sub(r'\*+', '', text)
            text = re.sub(r'#+\s*', '', text)
            return text
    except Exception as e:
        print(f"[Ollama] 连接失败详情: {type(e).__name__}: {e}")
        return f"AI连接失败：{e}"


def record_audio(stop_event, rate=16000) -> str:
    """录制音频直到 stop_event 被设置，返回临时wav文件路径"""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1,
                     rate=rate, input=True, frames_per_buffer=1024)
    frames = []
    while not stop_event.is_set():
        frames.append(stream.read(1024, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    pa.terminate()

    if not frames:
        return ""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return tmp.name


def transcribe(wav_path: str) -> str:
    """faster-whisper 语音识别，返回简体中文"""
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
        if _whisper_model is None:
            print("[Whisper] 模型未就绪，等待加载...")
            _whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
        segments, _ = _whisper_model.transcribe(wav_path, language="zh")
        text = "".join(s.text for s in segments).strip()
        # 繁体转简体
        try:
            import opencc
            cc = opencc.OpenCC('t2s')
            text = cc.convert(text)
        except Exception:
            pass
        return text
    except Exception as e:
        print(f"[Whisper] 识别出错: {e}")
        return ""
    finally:
        try:
            os.unlink(wav_path)
        except:
            pass


def speak(text: str):
    """edge-tts 朗读文字（异步，后台线程）"""
    def _run():
        try:
            import edge_tts
            tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tmp.close()

            async def _tts():
                comm = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
                await comm.save(tmp.name)

            asyncio.run(_tts())

            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                import time; import time; time.sleep(0.1)
            pygame.mixer.quit()
            os.unlink(tmp.name)
        except Exception as e:
            print(f"[TTS] {e}")

    threading.Thread(target=_run, daemon=True).start()


class AIChat:
    """管理多轮对话历史，支持带图片的首次分析"""

    def __init__(self):
        self.history = []
        self._lock = threading.Lock()
        # 启动时后台预加载 whisper 模型 + 预热 ollama
        threading.Thread(target=self._preload_whisper, daemon=True).start()
        threading.Thread(target=self._preheat_ollama, daemon=True).start()

    def _preheat_ollama(self):
        """预热 ollama，让模型提前加载到显存"""
        print(f"[Ollama] 预热模型 {OLLAMA_MODEL}...")
        ask_ollama([{"role": "user", "content": "你好"}])
        print(f"[Ollama] 模型预热完成，AI已就绪")

    def _preload_whisper(self):
        global _whisper_model
        if _whisper_model is None:
            print("[Whisper] 预加载模型...")
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
            print("[Whisper] 模型预加载完成，语音识别已就绪")

    def reset(self):
        with self._lock:
            self.history = []

    def analyze_frame(self, frame, on_reply, fruit_name="水果"):
        """带图片分析（到达时自动调用），on_reply(text) 回调"""
        def _run():
            img_b64 = frame_to_base64(frame)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"图片中是一个{fruit_name}，请分析这个{fruit_name}的新鲜程度。",
                    "images": [img_b64]
                }
            ]
            reply = ask_ollama(messages)
            with self._lock:
                self.history.append({"role": "user", "content": f"请分析这个{fruit_name}的新鲜程度。"})
                self.history.append({"role": "assistant", "content": reply})
            on_reply(reply)
            speak(reply)

        threading.Thread(target=_run, daemon=True).start()

    def chat(self, text: str, on_reply):
        """纯文字追问，on_reply(text) 回调"""
        def _run():
            msg = {"role": "user", "content": text}
            with self._lock:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history + [msg]
            reply = ask_ollama(messages)
            with self._lock:
                self.history.append(msg)
                self.history.append({"role": "assistant", "content": reply})
            on_reply(reply)
            speak(reply)

        threading.Thread(target=_run, daemon=True).start()

    def chat_with_frame(self, text: str, frame, on_reply):
        """发送时附带当前画面，但只有问题涉及图片内容时才真正带图"""
        # 判断是否需要看图
        visual_keywords = ["图片", "看", "这个", "新鲜", "腐烂", "什么水果", "颜色", "外观", "分析", "检测", "好不好", "坏了"]
        need_image = any(kw in text for kw in visual_keywords)

        def _run():
            with self._lock:
                history = list(self.history)
            if need_image and frame is not None:
                img_b64 = frame_to_base64(frame)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
                    {"role": "user", "content": text, "images": [img_b64]}
                ]
            else:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
                    {"role": "user", "content": text}
                ]
            reply = ask_ollama(messages)
            with self._lock:
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": reply})
            on_reply(reply)
            speak(reply)

        threading.Thread(target=_run, daemon=True).start()
