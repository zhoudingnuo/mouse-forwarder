"""
test_capture_detect.py - 截图采集卡画面并调用 ONNX 模型检测

用法: python test_capture_detect.py
输出:
  - capture_test.jpg        (带检测框的画面)
  - capture_test_raw.jpg    (原始画面)
  - 控制台打印检测结果
"""

import sys
import os
import time
import numpy as np
import cv2

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'pc_app', 'backend'))

MODEL_PATH = os.path.join(ROOT, 'valorant.onnx')


def find_capture_card():
    """查找 MS2130 采集卡 (跳过 index 0 的内置摄像头)"""
    print("Searching for MS2130 capture card...")
    for idx in [1, 2, 8, 3, 4, 5, 6, 7, 9]:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                continue
            # 尝试 720p - MS2130 支持
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(3):
                cap.read()
                time.sleep(0.02)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            time.sleep(0.1)
            if w >= 1280:
                print(f"  Found MS2130 at index {idx} ({w}x{h})")
                return idx
            print(f"  Camera {idx}: {w}x{h} (not MS2130)")
        except Exception as e:
            print(f"  Camera {idx} error: {e}")
    # 回退
    print("No MS2130 found, trying index 1")
    return 1


def capture_frame(idx):
    """从采集卡抓取一帧"""
    print(f"\nOpening camera {idx}...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {idx}")

    # 尝试 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 等几帧稳定
    for _ in range(10):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {w}x{h}")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to read frame")
    print(f"Captured frame: {frame.shape}")
    return frame


def detect(frame, model_path):
    """调用 ONNX 模型检测"""
    print(f"\nLoading model: {model_path}")
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available]
    session = ort.InferenceSession(model_path, providers=preferred)
    print(f"Providers: {preferred}")

    # 预处理 (与 object_detector.py 一致)
    INPUT_SIZE = 256
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0).astype(np.float32)

    # 推理
    t0 = time.perf_counter()
    outputs = session.run(['output0'], {'images': batch})
    t1 = time.perf_counter()
    print(f"Inference time: {(t1 - t0) * 1000:.1f}ms")

    output = outputs[0]  # [1, 9, 1344]
    print(f"Output shape: {output.shape}")

    # 后处理 (与 object_detector.py 一致)
    scores = np.squeeze(output, axis=0)  # [9, 1344]
    cx = scores[0, :]
    cy = scores[1, :]
    w = scores[2, :]
    h = scores[3, :]
    conf = scores[4, :]
    cls_scores = scores[5:9, :]

    # sigmoid
    conf = 1.0 / (1.0 + np.exp(-conf))
    cls_ids = np.argmax(cls_scores, axis=0)
    cls_conf = 1.0 / (1.0 + np.exp(-np.max(cls_scores, axis=0)))
    final_conf = conf * cls_conf

    # 置信度阈值
    THRESH = 0.45
    mask = final_conf >= THRESH
    if not np.any(mask):
        print("No detections above threshold")
        return []

    cx = cx[mask]; cy = cy[mask]; w = w[mask]; h = h[mask]
    final_conf = final_conf[mask]; cls_ids = cls_ids[mask]

    # 映射回原始尺寸
    orig_h, orig_w = frame.shape[:2]
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE
    cx_abs = cx * scale_x
    cy_abs = cy * scale_y
    w_abs = w * scale_x
    h_abs = h * scale_y
    x_abs = cx_abs - w_abs / 2
    y_abs = cy_abs - h_abs / 2

    CLASS_NAMES = ['head', 'body', 'weapon', 'unknown']
    COLORS = [
        (0, 255, 0),    # head - 绿
        (0, 200, 255),  # body - 橙
        (255, 100, 0),  # weapon - 蓝
        (200, 200, 200),
    ]

    detections = []
    for i in range(len(cx_abs)):
        cls_name = CLASS_NAMES[cls_ids[i]] if cls_ids[i] < len(CLASS_NAMES) else 'unknown'
        detections.append({
            'class_id': int(cls_ids[i]),
            'class_name': cls_name,
            'confidence': float(final_conf[i]),
            'x': float(x_abs[i]),
            'y': float(y_abs[i]),
            'w': float(w_abs[i]),
            'h': float(h_abs[i]),
            'cx': float(cx_abs[i]),
            'cy': float(cy_abs[i]),
        })

    # NMS
    keep = []
    order = np.argsort(-final_conf)
    boxes = np.stack([x_abs, y_abs, x_abs + w_abs, y_abs + h_abs], axis=1)
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        iw = np.maximum(0, xx2 - xx1)
        ih = np.maximum(0, yy2 - yy1)
        inter = iw * ih
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-7)
        order = order[1:][iou < 0.45]

    return [detections[i] for i in keep]


def draw(frame, detections):
    """在帧上绘制检测框"""
    CLASS_NAMES = ['head', 'body', 'weapon', 'unknown']
    COLORS = [
        (0, 255, 0),    # head - 绿
        (0, 200, 255),  # body - 橙
        (255, 100, 0),  # weapon - 蓝
        (200, 200, 200),
    ]
    annotated = frame.copy()
    for d in detections:
        color = COLORS[d['class_id']] if d['class_id'] < len(COLORS) else (200, 200, 200)
        x1 = int(d['x']); y1 = int(d['y'])
        x2 = int(d['x'] + d['w']); y2 = int(d['y'] + d['h'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    # 1. 找采集卡
    idx = find_capture_card()

    # 2. 抓一帧
    frame = capture_frame(idx)

    # 保存原始帧
    raw_path = os.path.join(ROOT, 'capture_test_raw.jpg')
    cv2.imwrite(raw_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nSaved raw frame: {raw_path}")

    # 3. 调用模型检测
    detections = detect(frame, MODEL_PATH)

    print(f"\n{'='*60}")
    print(f"Detected {len(detections)} object(s):")
    print(f"{'='*60}")
    for i, d in enumerate(detections):
        print(f"  [{i+1}] {d['class_name']:8s} conf={d['confidence']:.3f}  "
              f"box=({d['x']:.0f},{d['y']:.0f},{d['w']:.0f},{d['h']:.0f})  "
              f"center=({d['cx']:.0f},{d['cy']:.0f})")
    if not detections:
        print("  (no detections)")

    # 4. 绘制框图并保存
    annotated = draw(frame, detections)
    out_path = os.path.join(ROOT, 'capture_test.jpg')
    cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nSaved annotated frame: {out_path}")
    print(f"\nDone. Open these files to view:")
    print(f"  {raw_path}")
    print(f"  {out_path}")


if __name__ == '__main__':
    main()