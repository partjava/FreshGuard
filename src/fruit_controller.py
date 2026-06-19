# -*- coding: utf-8 -*-
"""
水果追踪控制器
两阶段策略：
  1. 远距离：HSV 颜色检测，追踪靠近
  2. 近距离：切换 YOLO 精确判断 fresh/rotten
扫描：原地转圈找颜色目标，找到后追踪

支持 GUI 模式：传入回调函数 on_frame / on_status / on_result
"""
import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from detector import FruitDetector
from video_receiver import VideoReceiver
from commander import Commander
from pid import PositionalPID

MODEL_PATH = 'runs/fruit_quality_max3/weights/best.pt'
TURN_45_TIME = 0.3      # 转45度时间（秒，调慢）
SCAN_WAIT_TIME = 1.2    # 转完后等待时间
CENTER_X = 320          # 画面中心x
ARRIVE_AREA = 0.08      # YOLO bbox 面积阈值，超过则停车判断
CENTER_DEAD_ZONE = 30   # 转向死区（像素），越小=对准越精确
LOST_FRAMES = 5         # 颜色丢失多少帧后启用记忆补偿

# 各水果颜色到达阈值（面积占比），水果越小阈值越小，保证停车距离一致
ARRIVE_COLOR = {
    'apples':  0.06,
    'apple':   0.06,
    'banana':  0.06,
    'bananas': 0.06,
    'oranges': 0.06,
    'orange':  0.06,
    'grape':   0.05,
    'grapes':  0.05,
    'mango':   0.06,
    'mangos':  0.06,
}

# 各水果的 HSV 颜色追踪范围
COLOR_RANGES = {
    # 橙子：橙红色，H范围收窄避免与香蕉重叠，S提高过滤淡色物体
    'oranges': [(np.array([5, 120, 100]), np.array([20, 255, 255]))],
    'orange':  [(np.array([5, 120, 100]), np.array([20, 255, 255]))],

    # 香蕉：明黄色，S和V提高以过滤墙壁、灯光等淡黄色干扰
    'banana':  [(np.array([20, 100, 120]), np.array([35, 255, 255]))],
    'bananas': [(np.array([20, 100, 120]), np.array([35, 255, 255]))],

    # 苹果：红色跨越HSV边界，需要两个范围。S提高到60过滤粉红色地板反光
    'apples':  [
        (np.array([0, 60, 100]), np.array([10, 255, 255])),     # 偏红端 (0-10度)
        (np.array([165, 60, 100]), np.array([179, 255, 255]))   # 紫红端 (165-179度)
    ],
    'apple':   [
        (np.array([0, 60, 100]), np.array([10, 255, 255])),
        (np.array([165, 60, 100]), np.array([179, 255, 255]))
    ],

    # 葡萄：黄绿色，H在35-65之间
    'grape':   [(np.array([35, 60, 80]), np.array([65, 255, 255]))],
    'grapes':  [(np.array([35, 60, 80]), np.array([65, 255, 255]))],

    # 芒果：偏纯绿色，H在40-80之间，饱和度高
    'mango':   [(np.array([40, 80, 60]), np.array([80, 255, 255]))],
    'mangos':  [(np.array([40, 80, 60]), np.array([80, 255, 255]))],
}

SCAN_TURN_SPEED = 0.25  # 扫描时旋转速度（调慢，防止转过头）
FOLLOW_SPEED = 0.45     # 正常追踪速度（调慢）
SLOW_SPEED = 0.25       # 接近水果时减速
TURN_GAIN = 0.6         # 转向增益


class FruitController:
    def __init__(self, on_frame=None, on_status=None, on_result=None):
        """
        on_frame(frame)  : 每帧图像回调（用于 GUI 显示）
        on_status(msg, color) : 状态文字回调
        on_result(msg, color) : 结果文字回调
        """
        self.detector = FruitDetector(MODEL_PATH)
        self.receiver = VideoReceiver()
        self.commander = Commander()
        self.turn_pid = PositionalPID(0.15, 0, 0.05)

        self._on_frame = on_frame
        self._on_status = on_status
        self._on_result = on_result
        self._on_ai = None  # AI分析回调，由GUI注入

        # GUI 模式下通过这两个变量控制追踪循环
        self.running = False
        self.tracking = False

    # ── 回调辅助 ──────────────────────────────────────────────────
    def _show(self, frame):
        if self._on_frame:
            self._on_frame(frame)
        else:
            cv2.imshow("Fruit Detection", frame)

    def _status(self, msg, color='black'):
        if self._on_status:
            self._on_status(msg, color)
        else:
            print(f"[STATUS] {msg}")

    def _result(self, msg, color='black'):
        if self._on_result:
            self._on_result(msg, color)
        else:
            print(f"[RESULT] {msg}")

    def _trigger_ai(self, frame):
        """触发AI图片分析"""
        if self._on_ai:
            self._on_ai(frame)

    # ── CLI 入口（无界面模式）────────────────────────────────────
    def run(self):
        self.receiver.start()
        self.commander.start()
        time.sleep(2)

        self.running = True
        while self.running:
            target_input, target_classes = self._ask_target()
            print("🔍 开始扫描...")
            self.tracking = True
            self.start_tracking(target_input, target_classes)

    # ── GUI 调用入口 ─────────────────────────────────────────────
    def start_tracking(self, target_input, target_classes):
        """由 GUI 调用，在后台线程中运行追踪逻辑"""
        self.running = True
        self.tracking = True

        found = self._scan_yolo(target_classes, target_input)
        if not found:
            self._status("扫描未找到目标，持续旋转搜索中...", 'orange')

        self.turn_pid = PositionalPID(0.15, 0, 0.05)
        lost_color_count = 0
        cx_color_last = None
        smooth_box = None  # EMA平滑后的bbox
        EMA_ALPHA = 0.4    # 平滑系数，越小越平滑

        while self.running and self.tracking:
            frame = self.receiver.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
                
            frame = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            # 1. 颜色检测 (远距离追踪)
            cx_color, area_color = self._detect_color(frame, target_input)

            # 颜色短期记忆，防止单帧丢失导致逻辑抖动
            if cx_color is not None:
                cx_color_last = cx_color
                lost_color_count = 0
            elif cx_color_last is not None and lost_color_count < LOST_FRAMES:
                lost_color_count += 1
                cx_color = cx_color_last

            # 2. 每帧只跑一次 YOLO
            box, label, conf, _ = self.detector.detect(frame, target_classes)

            # EMA平滑bbox，减少抖动
            if box is not None:
                if smooth_box is None:
                    smooth_box = box.copy()
                else:
                    smooth_box = EMA_ALPHA * box + (1 - EMA_ALPHA) * smooth_box
                box = smooth_box
            else:
                smooth_box = None

            # ================= 到达判断：YOLO bbox>=12% 且居中 =================
            arrive_thresh = ARRIVE_COLOR.get(target_input, 0.10)
            color_close = (area_color >= arrive_thresh and cx_color is not None)

            yolo_arrived = box is not None and (
                (box[2] - box[0]) * (box[3] - box[1]) / (frame_w * frame_h) >= ARRIVE_AREA
                and abs((box[0] + box[2]) / 2 - frame_w / 2) < CENTER_DEAD_ZONE * 2
            )

            if yolo_arrived or color_close:
                self.commander.send('stop')
                time.sleep(0.5)  # 等车停稳
                # 到达后连续检测5帧取置信度最高的结果
                votes = []
                for _ in range(8):
                    f = self._get_stable_frame()
                    if f is not None:
                        b, l, c, _ = self.detector.detect(f, target_classes)
                        if b is not None and l is not None:
                            votes.append((l, c))
                
                if votes:
                    # 投票：取出现次数最多的类别，票数相同时取置信度更高的
                    from collections import Counter
                    label_counts = Counter(l for l, c in votes)
                    best_label = label_counts.most_common(1)[0][0]
                    best_conf = max(c for l, c in votes if l == best_label)
                    label, conf = best_label, best_conf
                    is_fresh = label.startswith('fresh') or label.endswith('_fresh')
                    quality = "✅ 新鲜 (Fresh)" if is_fresh else "❌ 腐烂 (Rotten)"
                    self._result(f"{quality}\n置信度 {conf:.2f}",
                                 '#2e7d32' if is_fresh else '#c62828')
                    self._status("已到达！" + quality,
                                 '#2e7d32' if is_fresh else '#c62828')
                    if box is not None:
                        self._draw_yolo(frame, box, label, conf, target_input, "YOLO")
                else:
                    self._result("已到达（YOLO未识别，请手动判断）", 'orange')
                    self._status("已到达，YOLO未识别", 'orange')
                self._show(frame)
                # 停稳后重新抓一帧，裁剪出水果区域给AI分析，避免AI看到旁边其他水果
                import time as _time; _time.sleep(0.3)
                clean_frame = self._get_stable_frame()
                if clean_frame is not None:
                    # 用YOLO重新检测一次，拿到最新的bbox裁剪
                    cb, cl, cc, _ = self.detector.detect(clean_frame, target_classes)
                    if cb is not None:
                        x1c, y1c, x2c, y2c = map(int, cb)
                        # 稍微扩大一点裁剪区域，留点边距
                        pad = 20
                        h_f, w_f = clean_frame.shape[:2]
                        x1c = max(0, x1c - pad)
                        y1c = max(0, y1c - pad)
                        x2c = min(w_f, x2c + pad)
                        y2c = min(h_f, y2c + pad)
                        ai_frame = clean_frame[y1c:y2c, x1c:x2c]
                    else:
                        ai_frame = clean_frame
                    self._trigger_ai(ai_frame)
                else:
                    self._trigger_ai(frame)
                self._on_frame(None)  # 通知GUI清空旧帧，恢复实时画面
                self.tracking = False
                break

            # ================= 转向追踪 =================
            if box is not None:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                bbox_area = (box[2] - box[0]) * (box[3] - box[1]) / (frame_w * frame_h)
                self._move_toward(cx, frame_w)
                self._status(f"YOLO追踪 {label} ({conf:.2f})", 'blue')
                self._draw_yolo(frame, box, label, conf, target_input, "YOLO")
                cv2.putText(frame, f"Color:{area_color*100:.1f}% YOLO:{bbox_area*100:.1f}%",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            elif cx_color is not None:
                speed = SLOW_SPEED if area_color > 0.05 else FOLLOW_SPEED
                self._move_toward(cx_color, frame_w, speed)
                phase = "Slow" if area_color > 0.05 else "Full"
                self._status(f"颜色追踪 ({area_color*100:.1f}%) {phase}", '#FF9800')
                self._draw_yolo(frame, None, None, 0, target_input, "Color")
                cv2.putText(frame, f"Color {area_color*100:.1f}% {phase}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                # 颜色和YOLO都没检测到，转一小步再停下来检测
                self.commander.send('left')
                time.sleep(0.3)
                self.commander.send('stop')
                time.sleep(0.5)
                self._status("旋转搜索目标...", 'orange')
                self._draw_yolo(frame, None, None, 0, target_input, "Search")

            self._show(frame)

            # CLI 模式下处理按键
            if self._on_frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.commander.send('stop')
                    cv2.destroyAllWindows()
                    self.running = False
                    return
                elif key == ord('n'):
                    self.commander.send('stop')
                    self.tracking = False
                    break

        self.commander.send('stop')

    def stop_tracking(self):
        """由 GUI 调用，停止追踪"""
        self.tracking = False
        self.running = False
        for _ in range(3):
            self.commander.send('stop')
            time.sleep(0.05)

    # ── 内部工具 ──────────────────────────────────────────────────
    def _detect_color(self, frame, target_name):
        """HSV 颜色检测，返回 (cx, area_ratio) 或 (None, 0)"""
        # 支持 orange/oranges, apple/apples 等模糊匹配
        if target_name not in COLOR_RANGES:
            target_name = target_name + 's'
        if target_name not in COLOR_RANGES:
            return None, 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for (lower, upper) in COLOR_RANGES[target_name]:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        frame_area = frame.shape[0] * frame.shape[1]
        area_ratio = area / frame_area
        if area_ratio < 0.002:
            return None, 0
            
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 165, 255), 2)
        return cx, area_ratio

    def _move_toward(self, cx, frame_w, speed=None):
        """根据目标 cx 控制转向（使用位置式PID平滑控制）"""
        if speed is None:
            speed = FOLLOW_SPEED
        
        # 计算偏差：目标在画面中心，当前位置是 cx
        # center > 0 表示目标在右边，需要右转
        # center < 0 表示目标在左边，需要左转
        center = (cx - frame_w / 2) / (frame_w / 2)
        
        if abs(cx - frame_w / 2) < CENTER_DEAD_ZONE:
            # 在死区内，直行
            self.commander.send(f'motors:{speed:.2f}:{speed:.2f}')
        else:
            # 使用 PID 平滑转向
            self.turn_pid.SystemOutput = center
            self.turn_pid.SetStepSignal(0)
            self.turn_pid.SetInertiaTime(0.2, 0.1)
            
            # PID 输出的转向量（用 PidOutput，不是 SystemOutput）
            turn = self.turn_pid.PidOutput
            
            # 差速转向：左轮减速，右轮加速 = 左转
            speed_l = max(0.0, min(1.0, speed - TURN_GAIN * turn))
            speed_r = max(0.0, min(1.0, speed + TURN_GAIN * turn))
            self.commander.send(f'motors:{speed_l:.2f}:{speed_r:.2f}')

    def _scan_yolo(self, target_classes, target_name) -> bool:
        """扫描8个方向，优先用颜色检测找目标水果"""
        for i in range(8):
            if not self.tracking:
                self.commander.send('stop')
                return False
            self._status(f"扫描方向 {i+1}/8 ...", 'blue')
            hits = 0
            for _ in range(5):
                if not self.tracking:
                    self.commander.send('stop')
                    return False
                frame = self._get_stable_frame()
                if frame is not None:
                    cx, area = self._detect_color(frame, target_name)
                    if cx is not None:
                        hits += 1
                        cv2.putText(frame, "Scan Hit (Color)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        box, label, _, _ = self.detector.detect(frame, target_classes)
                        if box is not None:
                            hits += 1
                            cv2.putText(frame, "Scan Hit (YOLO)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Scanning {i+1}/8...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    self._show(frame)
                time.sleep(0.2)
            if hits >= 2:
                self.commander.send('stop')
                self._status(f"第{i+1}方向发现目标！", 'green')
                return True
            if not self.tracking:
                self.commander.send('stop')
                return False
            self.commander.send('left')
            time.sleep(TURN_45_TIME)
            self.commander.send('stop')
            time.sleep(SCAN_WAIT_TIME)
        return False

    def _get_stable_frame(self):
        for _ in range(10):
            frame = self.receiver.get_frame()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    def _ask_target(self):
        while True:
            name = input("🎯 请输入水果名（如 apples, banana, oranges）：").strip().lower()
            classes = self.detector.get_target_classes(name)
            if classes:
                return name, classes
            print(f"❌ 找不到 '{name}'，请重新输入")

    def _draw_yolo(self, frame, box, label, conf, target_input, mode):
        if box is not None and label is not None:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if (label.startswith('fresh') or label.endswith('_fresh')) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1-8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"[{mode}] {target_input}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.line(frame, (CENTER_X, 0), (CENTER_X, frame.shape[0]), (128, 128, 128), 1)