"""
水果检测模块
功能：加载 YOLO 模型，对画面进行推理
"""
from ultralytics import YOLO
from config import CONF_THRESHOLD


class FruitDetector:
    def __init__(self, model_path: str, device=0):
        """加载 YOLO 模型，默认使用GPU推理 (device=0)"""
        self.model = YOLO(model_path)
        self.model.to(device)
        self.names = self.model.names  # {cls_id: 'freshapple', ...}

    def get_supported_fruits(self) -> list:
        """返回模型支持的水果名列表"""
        # 新数据集类别名直接是 fresh/rotten，旧数据集是 freshorange 等
        fruits = set()
        for name in self.names.values():
            for prefix in ('fresh', 'rotten'):
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    fruits.add(suffix if suffix else name)
        return sorted(fruits)

    def get_target_classes(self, fruit_name: str) -> dict:
        """
        根据水果名返回匹配的类别 ID 字典
        兼容两种数据集：
          - 新数据集：类别名是 'fresh'/'rotten'，输入任意水果名都返回全部类别
          - 旧数据集：类别名是 'freshorange' 等，按水果名过滤
        """
        fruit_name = fruit_name.strip().lower()
        # 检查是否是纯 fresh/rotten 类型的数据集
        all_names = list(self.names.values())
        if all(n in ('fresh', 'rotten') for n in all_names):
            # 新数据集：直接返回全部类别（fresh + rotten）
            return dict(self.names)
        # 旧数据集：按水果名匹配（支持模糊匹配：banana/bananas, orange/oranges, apple/apples）
        aliases = [fruit_name, fruit_name + 's', fruit_name.rstrip('s')]
        return {cid: n for cid, n in self.names.items()
                if any(alias in n.lower() for alias in aliases)}

    def detect(self, frame, target_classes: dict):
        """
        对单帧图像推理，只返回目标水果中置信度最高的结果
        返回: (box, label, conf, None) 或 (None, None, 0, None)
        """
        results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return None, None, 0.0, None

        # 筛选目标水果的框
        target_indices = [i for i in range(len(boxes))
                          if int(boxes.cls[i]) in target_classes]

        if not target_indices:
            return None, None, 0.0, None

        # 取置信度最高的框
        best_i = max(target_indices, key=lambda i: float(boxes.conf[i]))
        box = boxes.xyxy[best_i].cpu().numpy()
        cls_id = int(boxes.cls[best_i])
        conf = float(boxes.conf[best_i])
        label = self.names[cls_id]

        # 长宽比过滤：苹果/橙子近似正方形，垃圾桶/篮子宽高比差异大
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        ratio = w / h if h > 0 else 99
        # banana 可以是横向，其他水果限制在 0.4~2.5
        if 'banana' not in label and not (0.4 < ratio < 2.5):
            return None, None, 0.0, None

        return box, label, conf, None
