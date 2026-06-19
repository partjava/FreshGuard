# -*- coding: utf-8 -*-
"""
专门用于获取水果真实颜色范围的标定工具。
不再依赖 YOLO！直接在屏幕中心画一个固定准心框。
只要把苹果放在中间的框内，就能实时显示真实的 HSV 范围。
"""
import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from video_receiver import VideoReceiver

def main():
    print("🚀 启动颜色标定工具 (无脑取色模式)...")
    receiver = VideoReceiver()
    receiver.start()
    
    time.sleep(2)  # 等待摄像头稳定
    print("✅ 摄像头已连接！请把水果完全放到屏幕中央的蓝色方框内...")
    
    while True:
        frame = receiver.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
            
        display = frame.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 屏幕尺寸
        h, w = frame.shape[:2]
        
        # 在屏幕正中央画一个 100x100 的固定准心框 (你可以调整大小)
        box_size = 100
        cx, cy = w // 2, h // 2
        x1, x2 = cx - box_size // 2, cx + box_size // 2
        y1, y2 = cy - box_size // 2, cy + box_size // 2
        
        # 画出准心框
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display, "Put Fruit HERE", (x1 - 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 提取准心框内的 HSV 图像块
        core_hsv = hsv_frame[y1:y2, x1:x2]
        
        if core_hsv.size > 0:
            # 计算这块区域的 H, S, V 平均值和范围
            h_mean = int(np.mean(core_hsv[:,:,0]))
            s_mean = int(np.mean(core_hsv[:,:,1]))
            v_mean = int(np.mean(core_hsv[:,:,2]))
            
            # 使用更鲁棒的分位数来计算范围，避免被极个别噪点干扰（取10%到90%的值）
            h_min, h_max = int(np.percentile(core_hsv[:,:,0], 10)), int(np.percentile(core_hsv[:,:,0], 90))
            s_min, s_max = int(np.percentile(core_hsv[:,:,1], 10)), int(np.percentile(core_hsv[:,:,1], 90))
            v_min, v_max = int(np.percentile(core_hsv[:,:,2], 10)), int(np.percentile(core_hsv[:,:,2], 90))
            
            # 若跨越红色边界 (一半在0附近，一半在180附近，粉红经常这样)
            if h_max - h_min > 90:
                h_mean_text = f"H: Red Edge ({h_min}~{h_max})"
            else:
                h_mean_text = f"H: {h_mean} ({h_min}~{h_max})"
                
            info1 = f"Range H: {h_min} - {h_max}"
            info2 = f"Range S: {s_min} - {s_max}"
            info3 = f"Range V: {v_min} - {v_max}"
            
            # 画在屏幕上，绿色大字，非常清晰
            cv2.putText(display, "Live HSV Sampling:", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, info1, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, info2, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, info3, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 命令行也打一份防丢
            print(f"采样中... H:{h_mean}({h_min}-{h_max}) S:{s_min}-{s_max} V:{v_min}-{v_max}", end='\r')
            
        cv2.imshow("Color Calibration Tool (No YOLO)", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    receiver.running = False
    print("\n🛑 工具已退出！")

if __name__ == '__main__':
    main()
