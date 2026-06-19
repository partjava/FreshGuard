import os
from pathlib import Path
from ultralytics import YOLO
import cv2

def generate_labels():
    dataset_dir = Path("c:/Users/34970/Desktop/challenge_cup_fruit/dataset")
    
    # 类别映射：使用六个类别
    classes_map = {
        'freshapples': 0,
        'freshbanana': 1,
        'freshoranges': 2,
        'rottenapples': 3,
        'rottenbanana': 4,
        'rottenoranges': 5
    }
    
    # COCO数据集中水果的类别ID (香蕉46, 苹果47, 橘子/橙子49)
    valid_coco_classes = [46, 47, 49]

    # 加载预训练模型来获取目标的bounding box
    model_path = "c:/Users/34970/Desktop/challenge_cup_fruit/yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}，请确保路径正确")
        # Fallback to downloading
        model = YOLO("yolo11n.pt") 
    else:
        model = YOLO(model_path)
    
    print("开始使用YOLOv11提取图像中的水果边框并生成标签...")
    
    splits = ['train', 'test']
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir():
                continue
                
            # 判断好坏类别
            folder_name = class_folder.name.lower()
            if folder_name in classes_map:
                label_id = classes_map[folder_name]
            else:
                print(f"跳过未知类别文件夹: {class_folder}")
                continue
                
            print(f"正在处理: {class_folder}")
            
            # 遍历文件夹内图片
            for img_file in class_folder.glob('*.*'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                    
                # 运行YOLO检测
                results = model.predict(source=str(img_file), verbose=False, save=False)
                
                label_file = img_file.with_suffix('.txt')
                box_found = False
                
                with open(label_file, 'w') as f:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # 过滤只保留水果的预测框
                            if int(box.cls[0].item()) in valid_coco_classes:
                                # 获取归一化的x_center, y_center, width, height
                                b = box.xywhn[0].tolist()
                                line = f"{label_id} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n"
                                f.write(line)
                                box_found = True
                                
                    # 如果未检测到水果，或者置信度过低，默认给一个包含全图的边框
                    # 或者给一个居中且占画面较大比例的边框作为退路
                    if not box_found:
                        # 默认标签格式：class_id 0.5 0.5 0.8 0.8 (居中，占80%的区域)
                        f.write(f"{label_id} 0.500000 0.500000 0.800000 0.800000\n")
                        
    print("全部标签生成完成！标签文件已保存在对应的图片同级目录下。")
    print("建议：请检查YOLOv11是否能够识别这两种类别，创建对应的 data.yaml 用于后续训练。")

def create_yaml():
    yaml_path = Path("c:/Users/34970/Desktop/challenge_cup_fruit/dataset/data.yaml")
    yaml_content = """
path: c:/Users/34970/Desktop/challenge_cup_fruit/dataset
train: train
val: test 

# Classes
names:
  0: freshapples
  1: freshbanana
  2: freshoranges
  3: rottenapples
  4: rottenbanana
  5: rottenoranges
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    print(f"已生成数据集配置文件: {yaml_path}")

if __name__ == '__main__':
    generate_labels()
    create_yaml()
