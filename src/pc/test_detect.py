"""
测试入口：只做检测显示，不控制小车
用于验证模型能否识别摄像头画面中的水果
"""
import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from detector import FruitDetector
from video_receiver import VideoReceiver

MODEL_PATH = 'runs/fruit_quality_max3/weights/best.pt'

if __name__ == '__main__':
    detector = FruitDetector(MODEL_PATH)
    receiver = VideoReceiver()
    receiver.start()

    print("✅ 模型加载完成，等待画面...")
    # 检测所有类别（不过滤目标）
    all_classes = detector.names

    while True:
        frame = receiver.get_frame()
        if frame is None:
            continue

        box, label, conf, _ = detector.detect(frame, all_classes)
        if label:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if label.startswith('fresh') else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, "Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No fruit detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Fruit Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
