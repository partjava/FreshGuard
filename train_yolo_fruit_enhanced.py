"""
YOLO11 模型训练脚本（水果质量检测增强版-4070优化版）
针对RTX 4070 8G显存优化，解决显存吃满、训练变慢问题
训练完成后自动将最优权重复制到 models/fruit_quality_enhanced.pt
"""
import shutil
import os
from ultralytics import YOLO

if __name__ == '__main__':
    # 配置：YOLO11m 中等模型（4070显存适配）
    model = YOLO('yolo11m.pt')

    # 核心训练配置（4070 8G显存优化版）
    results = model.train(
        # 基础配置（显存优化：batch=8，稳定不卡）
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,                   # 4070 8G显存安全值，改回8，利用率拉满
        device=0,
        project='runs',
        name='fruit_quality_enhanced',
        patience=20,               # 早停机制，避免过拟合
        save=True,
        workers=0,                 # Windows下保持0，避免多进程bug

        # 核心增强（保留关键项，去掉冗余，提速+显存优化）
        degrees=180,               # 0~180°随机旋转，打乱位置
        translate=0.3,             # 0~30%画面平移，解决位置集中
        scale=0.6,                 # 0.6~1.4倍缩放，解决大小集中
        hsv_h=0.02,                # 色调增强，适配不同光照
        hsv_s=0.7,                 # 饱和度增强
        hsv_v=0.5,                 # 亮度增强
        flipud=0.5,                # 上下翻转
        fliplr=0.5,                # 左右翻转
        mosaic=1.0,                # 4图拼接，强制位置多样化（核心）
        # 可选：去掉以下参数进一步提速，不影响核心效果
        # shear=10,
        # perspective=0.002,
        # copy_paste=0.2,
        # mixup=0.1,
    )

    # 保存增强版最优模型（增加异常捕获）
    os.makedirs('models', exist_ok=True)
    target_path = 'models/fruit_quality_enhanced.pt'
    try:
        best_pt = os.path.join(results.save_dir, 'weights', 'best.pt')
        shutil.copy(best_pt, target_path)
        print(f"\n✅ 训练完成！增强版最优模型已保存至：{target_path}")
        print(f"📁 训练日志路径：{results.save_dir}")
    except FileNotFoundError:
        print(f"\n⚠️  未找到best.pt文件，可能训练未正常完成！")
        print(f"📁 训练日志路径：{results.save_dir}")