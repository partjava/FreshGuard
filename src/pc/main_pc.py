# -*- coding: utf-8 -*-
"""
水果追踪 GUI 界面 (main_pc.py)
只负责：画面显示 / 按钮控制 / 状态展示
追踪逻辑全部委托给 FruitController
"""
import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fruit_controller import FruitController, COLOR_RANGES
from detector import FruitDetector
from ai_chat import AIChat

MODEL_PATH = 'runs/fruit_quality_max3/weights/best.pt'
CENTER_DEAD_ZONE = 60
CANVAS_W = 1280
CANVAS_H = 720
AI_OVERLAY_H = 160  # 画面底部AI回复区域高度


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("水果质量检测追踪系统")
        self.root.resizable(False, False)

        # 创建控制器（注入 GUI 回调）
        self.controller = FruitController(
            on_frame=self._on_frame,
            on_status=self._set_status,
            on_result=self._set_result,
        )

        self._build_ui()

        # 启动网络连接
        self.controller.receiver.start()
        self.controller.commander.start()

        # AI 对话
        self.ai = AIChat()
        self._ai_text = ""
        self.controller._on_ai = self.trigger_ai_analysis
        # 摄像头连接后预热 ollama
        self.controller.receiver._on_connected = self.ai._preheat_ollama

        # 启动画面刷新（仅用于空闲时的预览框）
        self._last_frame = None
        self._update_frame()

    def _build_ui(self):
        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H, bg='black')
        self.canvas.grid(row=0, column=0, rowspan=6, padx=10, pady=(10,0))

        # AI 输入框区域（画面正下方）
        ai_bar = tk.Frame(self.root, bg='#1e1e1e')
        ai_bar.grid(row=6, column=0, sticky='ew', padx=10, pady=(0,10))

        tk.Label(ai_bar, text="🤖", bg='#1e1e1e', fg='white',
                 font=('Arial', 14)).pack(side='left', padx=6)
        self.ai_input = tk.Entry(ai_bar, font=('Arial', 12), bg='#2d2d2d', fg='white',
                                 insertbackground='white', relief='flat')
        self.ai_input.pack(side='left', fill='x', expand=True, ipady=6, padx=4)
        self.ai_input.bind('<Return>', lambda e: self._send_ai())

        tk.Button(ai_bar, text='发送', font=('Arial', 11), bg='#4CAF50', fg='white',
                  relief='flat', padx=10, command=self._send_ai).pack(side='left', padx=4)
        self.btn_voice = tk.Button(ai_bar, text='🎤 按住说话', font=('Arial', 11),
                                   bg='#2196F3', fg='white', relief='flat', padx=10)
        self.btn_voice.pack(side='left', padx=4)
        self.btn_voice.bind('<ButtonPress-1>',   self._voice_start)
        self.btn_voice.bind('<ButtonRelease-1>', self._voice_stop)
        self._voice_recording = False

        panel = tk.Frame(self.root, padx=10)
        panel.grid(row=0, column=1, sticky='n', pady=10)

        tk.Label(panel, text="目标水果", font=('Arial', 11)).pack(anchor='w')
        self.fruit_var = tk.StringVar(value="apples")
        tk.Entry(panel, textvariable=self.fruit_var, font=('Arial', 12), width=12).pack(fill='x', pady=4)

        self.btn_start = tk.Button(panel, text="▶ 开始追踪", font=('Arial', 11, 'bold'),
                                   bg='#4CAF50', fg='white', command=self._start_tracking,
                                   width=12, height=1)
        self.btn_start.pack(fill='x', pady=4)

        self.btn_stop = tk.Button(panel, text="■ 停止", font=('Arial', 11, 'bold'),
                                  bg='#f44336', fg='white', command=self._stop_tracking,
                                  width=12, state='disabled')
        self.btn_stop.pack(fill='x', pady=4)

        self.btn_next = tk.Button(panel, text="⟳ 换目标", font=('Arial', 11),
                                  bg='#2196F3', fg='white', command=self._next_target,
                                  width=12, state='disabled')
        self.btn_next.pack(fill='x', pady=4)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(panel, text="状态", font=('Arial', 11)).pack(anchor='w')
        self.lbl_status = tk.Label(panel, text="就绪", font=('Arial', 10),
                                   fg='gray', wraplength=180, justify='left')
        self.lbl_status.pack(anchor='w', pady=2)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(panel, text="检测结果", font=('Arial', 11)).pack(anchor='w')
        self.lbl_result = tk.Label(panel, text="—", font=('Arial', 14, 'bold'),
                                   fg='gray', wraplength=180, justify='left')
        self.lbl_result.pack(anchor='w', pady=4)

        self.btn_redetect = tk.Button(panel, text="🔄 重新检测", font=('Arial', 10),
                                      bg='#FF9800', fg='white', command=self._redetect,
                                      width=12)
        self.btn_redetect.pack(fill='x', pady=4)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(panel, text="手动控制（按住移动）", font=('Arial', 11)).pack(anchor='w')
        btn_style = dict(font=('Arial', 12), width=3, height=1, bg='#607D8B', fg='white')

        row0 = tk.Frame(panel); row0.pack()
        row1 = tk.Frame(panel); row1.pack()
        row2 = tk.Frame(panel); row2.pack()

        self._make_ctrl_btn(row0, '↖', 'left',     **btn_style).pack(side='left', padx=2, pady=2)
        self._make_ctrl_btn(row0, '↑', 'forward',  **btn_style).pack(side='left', padx=2, pady=2)
        self._make_ctrl_btn(row0, '↗', 'right',    **btn_style).pack(side='left', padx=2, pady=2)
        self._make_ctrl_btn(row1, '←', 'left',     **btn_style).pack(side='left', padx=2, pady=2)
        tk.Button(row1, text='⏹', **btn_style,
                  command=lambda: self.controller.commander.send('stop')).pack(side='left', padx=2, pady=2)
        self._make_ctrl_btn(row1, '→', 'right',    **btn_style).pack(side='left', padx=2, pady=2)
        self._make_ctrl_btn(row2, '↙', 'left',     **btn_style).pack(side='left', padx=2, pady=2)
        self._make_ctrl_btn(row2, '↓', 'backward', **btn_style).pack(side='left', padx=2, pady=2)
        self._make_ctrl_btn(row2, '↘', 'right',    **btn_style).pack(side='left', padx=2, pady=2)

    def _set_status(self, msg, color='black'):
        self.root.after(0, lambda: self.lbl_status.config(text=msg, fg=color))

    def _set_result(self, msg, color='black'):
        self.root.after(0, lambda: self.lbl_result.config(text=msg, fg=color))

    def _make_ctrl_btn(self, parent, text, cmd, **kwargs):
        """创建按住发指令、松开停止的控制按钮"""
        btn = tk.Button(parent, text=text, **kwargs)
        btn.bind('<ButtonPress-1>',   lambda e: self.controller.commander.send(cmd))
        btn.bind('<ButtonRelease-1>', lambda e: self.controller.commander.send('stop'))
        return btn

    def _redetect(self):
        """对当前帧重新跑5次YOLO投票，更新检测结果"""
        name = self.fruit_var.get().strip().lower()
        classes = self.controller.detector.get_target_classes(name)
        if not classes:
            self._set_result("找不到类别", 'red')
            return
        self._set_result("检测中...", 'gray')
        def _run():
            votes = []
            for _ in range(5):
                f = self.controller.receiver.get_frame()
                if f is not None:
                    b, l, c, _ = self.controller.detector.detect(f, classes)
                    if b is not None and l is not None:
                        votes.append((l, c))
            if votes:
                best = max(votes, key=lambda x: x[1])
                label, conf = best
                is_fresh = label.startswith('fresh') or label.endswith('_fresh')
                quality = "✅ 新鲜 (Fresh)" if is_fresh else "❌ 腐烂 (Rotten)"
                self._set_result(f"{quality}\n置信度 {conf:.2f}",
                             '#2e7d32' if is_fresh else '#c62828')
            else:
                self._set_result("未检测到目标", 'orange')
        threading.Thread(target=_run, daemon=True).start()

    def _on_frame(self, frame):
        """追踪线程回调：把已经画好框的帧存起来，等 _update_frame 显示"""
        if frame is None:
            self._last_frame = None
            return
        self._last_frame = frame.copy()

    def _start_tracking(self):
        name = self.fruit_var.get().strip().lower()
        classes = self.controller.detector.get_target_classes(name)
        if not classes:
            self._set_status(f"找不到类别: {name}", 'red')
            return
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.btn_next.config(state='normal')
        self._set_result("—", 'gray')
        threading.Thread(
            target=self.controller.start_tracking,
            args=(name, classes),
            daemon=True
        ).start()

    def _stop_tracking(self):
        self.controller.stop_tracking()
        self._last_frame = None  # 清空旧帧，防止画面卡在上一次的搜索结果上
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.btn_next.config(state='disabled')
        self._set_status("已停止", 'gray')

    def _next_target(self):
        self.controller.tracking = False
        self.controller.running = False
        self.controller.commander.send('stop')
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        self.btn_next.config(state='disabled')
        self._set_status("请输入新目标后点击开始", 'gray')

    def _update_frame(self):
        """30fps UI 刷新：如果追踪中用追踪线程画好的帧，否则画三色预览"""
        if self.controller.running and self._last_frame is not None:
            # 追踪中：直接显示追踪线程处理过的帧
            display = self._last_frame
        else:
            # 空闲或停车后：取新帧画三色预览
            display = self.controller.receiver.get_frame()
            if display is not None:
                display = display.copy()
                self._draw_preview(display)

        if display is not None:
            # 缩放到 canvas 尺寸
            display = cv2.resize(display, (CANVAS_W, CANVAS_H))
            cx = CANVAS_W // 2
            # 画中心线和死区线
            cv2.line(display, (cx, 0), (cx, CANVAS_H), (128, 128, 128), 1)
            cv2.line(display, (cx - CENTER_DEAD_ZONE, 0),
                     (cx - CENTER_DEAD_ZONE, CANVAS_H), (0, 200, 255), 1)
            cv2.line(display, (cx + CENTER_DEAD_ZONE, 0),
                     (cx + CENTER_DEAD_ZONE, CANVAS_H), (0, 200, 255), 1)

            # AI 回复叠加在画面底部
            if self._ai_text:
                self._draw_ai_overlay(display, self._ai_text)

            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor='nw', image=imgtk)

        self.root.after(33, self._update_frame)

    def _draw_preview(self, frame):
        """空闲时把三种水果的颜色块都框出来"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        target = self.fruit_var.get().strip().lower()

        PREVIEW = {
            'apples':  ([(np.array([0, 60, 100]),   np.array([10, 255, 255])),
                         (np.array([165, 60, 100]), np.array([179, 255, 255]))],
                        (0, 80, 220), 'Apple'),
            'banana':  ([(np.array([20, 100, 120]), np.array([35, 255, 255]))],
                        (0, 220, 220), 'Banana'),
            'oranges': ([(np.array([5, 120, 100]),  np.array([20, 255, 255]))],
                        (0, 140, 255), 'Orange'),
        }

        for fruit_name, (ranges, color, label) in PREVIEW.items():
            if not target or (fruit_name not in target and target not in fruit_name):
                continue

            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                area_ratio = cv2.contourArea(cnt) / (frame.shape[0] * frame.shape[1])
                if area_ratio >= 0.005:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{label} {area_ratio*100:.1f}%",
                                (x, max(14, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_ai_overlay(self, frame, text: str):
        """在画面底部画半透明黑底+白字的AI回复，用PIL支持中文"""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - AI_OVERLAY_H), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 转PIL画中文
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
            font_small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
        except:
            font = ImageFont.load_default()
            font_small = font

        draw.text((10, h - AI_OVERLAY_H + 8), "AI:", font=font, fill=(0, 200, 255))

        # 自动换行
        max_w = w - 60
        line = ""
        y = h - AI_OVERLAY_H + 36
        for ch in text:
            line += ch
            bbox = draw.textbbox((0, 0), line, font=font_small)
            if bbox[2] > max_w:
                draw.text((10, y), line[:-1], font=font_small, fill=(255, 255, 255))
                y += 26
                line = ch
                if y > h - 8:
                    break
        if line:
            draw.text((10, y), line, font=font_small, fill=(255, 255, 255))

        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _set_ai_text(self, text: str):
        self._ai_text = text

    def _send_ai(self):
        text = self.ai_input.get().strip()
        if not text:
            return
        self.ai_input.delete(0, tk.END)
        # 发送时抓最新一帧（不用旧的追踪帧）
        frame = self.controller.receiver.get_frame()
        self._ai_text = f"你: {text}\nAI: 思考中..."
        if frame is not None:
            self.ai.chat_with_frame(text, frame, lambda r: self.root.after(0, lambda: self._set_ai_text(f"你: {text}\nAI: {r}")))
        else:
            self.ai.chat(text, lambda r: self.root.after(0, lambda: self._set_ai_text(f"你: {text}\nAI: {r}")))

    def _voice_start(self, event):
        self._voice_recording = True
        self._voice_stop_event = threading.Event()
        self.btn_voice.config(bg='#f44336', text='🔴 录音中...')
        self._voice_thread = threading.Thread(target=self._do_record, daemon=True)
        self._voice_thread.start()

    def _voice_stop(self, event):
        self._voice_recording = False
        if hasattr(self, '_voice_stop_event'):
            self._voice_stop_event.set()
        self.btn_voice.config(bg='#2196F3', text='🎤 按住说话')

    def _do_record(self):
        from ai_chat import record_audio, transcribe
        print("[Voice] 开始录音...")
        wav = record_audio(self._voice_stop_event)
        print(f"[Voice] 录音结束，文件: {wav}")
        if not wav:
            print("[Voice] 录音文件为空")
            return
        print("[Voice] 开始识别...")
        text = transcribe(wav)
        print(f"[Voice] 识别结果: '{text}'")
        if text:
            self.root.after(0, lambda: self.ai_input.insert(0, text))
            # 只填到输入框，不自动发送，让用户确认后再按发送
        else:
            self.root.after(0, lambda: self._set_status("未识别到语音，请重试", 'orange'))

    def trigger_ai_analysis(self, frame):
        """到达时自动触发AI分析（由追踪逻辑调用）"""
        self.ai.reset()
        fruit_name = self.fruit_var.get().strip().lower()
        # 中文名映射
        name_map = {'apples': '苹果', 'apple': '苹果', 'banana': '香蕉', 'bananas': '香蕉',
                    'oranges': '橙子', 'orange': '橙子', 'grape': '葡萄', 'grapes': '葡萄',
                    'mango': '芒果', 'mangos': '芒果'}
        fruit_cn = name_map.get(fruit_name, fruit_name)
        self._ai_text = "🤖 AI 分析中..."
        self.ai.analyze_frame(
            frame,
            lambda r: self.root.after(0, lambda: self._set_ai_text(r)),
            fruit_name=fruit_cn
        )


if __name__ == '__main__':
    print("🚀 正在启动智能水果质检系统...")
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    app.controller.stop_tracking()
