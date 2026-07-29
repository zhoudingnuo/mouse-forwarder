"""
batch_detect.py - 批量视频检测脚本

输出:
  1. 带检测框的视频 (output_detected.mp4)
  2. 有检测的帧保存为单图 (frame_XXXX.jpg)

用法: python batch_detect.py <视频路径> [输出目录]
"""

import os, sys, time
import numpy as np
import cv2
import onnxruntime as ort

# ── 配置 ────────────────────────────────────────
CONF_THRESH = 0.25
IOU_THRESH = 0.45
INPUT_SIZE = 256          # 裁剪尺寸 = 模型输入尺寸 (不 resize)
MODEL_PATH = 'valorant.onnx'
PROCESS_MINUTES = 5
FPS_SAMPLE = 1

CLASS_NAMES = ['body', 'head', 'teammate', 'breakable', 'dodge']
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


def nms_cpu(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        mask = iou <= iou_thresh
        order = order[1:][mask]
    return keep


def main():
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
    else:
        video_path = r"C:\Users\AnlangZ\Desktop\train\2026-07-30-01-35-26.mkv"

    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'batch_output')

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    print(f"[INFO] Loading model: {model_path}")
    available = ort.get_available_providers()
    providers = [p for p in ['DmlExecutionProvider', 'CPUExecutionProvider'] if p in available]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    opts.enable_cpu_mem_arena = False
    session = ort.InferenceSession(model_path, opts, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"[INFO] Providers: {providers}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {w0}x{h0}, {fps:.2f}fps, {total_frames} frames")

    max_frames = min(int(PROCESS_MINUTES * 60 * fps), total_frames)
    half = INPUT_SIZE // 2

    # 输出视频 (XVID codec)
    video_out_path = os.path.join(output_dir, 'output_detected.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(video_out_path, fourcc, 1.0, (w0, h0))

    print(f"[INFO] Processing first {PROCESS_MINUTES}min (1 fps, center {INPUT_SIZE}x{INPUT_SIZE} crop)")
    print(f"[INFO] Video output: {video_out_path}")
    print(f"[INFO] Frames with detections saved as JPG in: {output_dir}\n")

    start = time.time()
    t_infer_list = []
    processed = 0
    frame_idx = 0
    det_frames = 0
    last_log = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = frame_idx / fps
        # 只处理整秒的帧
        if abs(current_sec - round(current_sec)) > 0.01:
            frame_idx += 1
            continue

        h0f, w0f = frame.shape[:2]
        x_start = max(0, w0f // 2 - half)
        y_start = max(0, h0f // 2 - half)
        x_end = min(w0f, x_start + INPUT_SIZE)
        y_end = min(h0f, y_start + INPUT_SIZE)

        # ---- 预处理: 中心裁剪 256×256 (不 resize) ----
        img_crop = frame[y_start:y_end, x_start:x_end]
        ch, cw = img_crop.shape[:2]
        if ch != INPUT_SIZE or cw != INPUT_SIZE:
            img_crop = cv2.copyMakeBorder(img_crop, 0, INPUT_SIZE-ch, 0, INPUT_SIZE-cw,
                                           cv2.BORDER_CONSTANT, value=(0,0,0))
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(img_rgb, (2, 0, 1))[np.newaxis, :]

        # ---- 推理 ----
        t0 = time.perf_counter()
        out = session.run([session.get_outputs()[0].name], {input_name: inp})[0][0]
        infer_ms = (time.perf_counter() - t0) * 1000
        t_infer_list.append(infer_ms)

        # ---- 解析 ----
        cx = out[0]; cy = out[1]; w = out[2]; h = out[3]
        class_scores = np.max(out[4:], axis=0)
        class_ids = np.argmax(out[4:], axis=0)
        mask = class_scores >= CONF_THRESH

        boxes_xyxy = np.empty((0, 4))
        scores_keep = np.empty(0)
        ids_keep = np.empty(0, dtype=int)

        if mask.sum() > 0:
            cx_m, cy_m, w_m, h_m = cx[mask], cy[mask], w[mask], h[mask]
            scores_keep = class_scores[mask]
            ids_keep = class_ids[mask]

            # 1:1 映射: 256 空间坐标 = 像素坐标 (无 resize)
            cx_img = cx_m + x_start
            cy_img = cy_m + y_start
            w_img = w_m   # 像素尺寸
            h_img = h_m

            x1 = (cx_img - w_img / 2).clip(0, w0f)
            y1 = (cy_img - h_img / 2).clip(0, h0f)
            x2 = (cx_img + w_img / 2).clip(0, w0f)
            y2 = (cy_img + h_img / 2).clip(0, h0f)
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = nms_cpu(boxes_xyxy, scores_keep, IOU_THRESH)

        # ---- 画框 ----
        annotated = frame.copy()
        for idx in keep:
            bx1, by1, bx2, by2 = boxes_xyxy[idx].astype(int)
            cls_id = ids_keep[idx]
            conf = scores_keep[idx]
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 3)
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (bx1, max(by1 - 20, 0)), (bx1 + tw, max(by1 - 5, 0)), color, -1)
            cv2.putText(annotated, label, (bx1, max(by1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 裁剪区域边框
        cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)
        info = f"#{processed} t={current_sec:.0f}s det={len(keep)} infer={infer_ms:.1f}ms"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 写入视频 (每帧)
        out_writer.write(annotated)

        # 有检测的帧保存单图
        if len(keep) > 0:
            det_frames += 1
            cv2.imwrite(os.path.join(output_dir, f"frame_{processed:04d}_t{current_sec:.0f}s_{len(keep)}det.jpg"),
                        annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        processed += 1
        if processed % 30 == 0:
            avg_ms = np.mean(t_infer_list)
            print(f"[{processed}/~{PROCESS_MINUTES*60}] t={current_sec:.0f}s "
                  f"det={len(keep)} | infer={infer_ms:.1f}ms avg={avg_ms:.1f}ms")

        frame_idx += 1

    cap.release()
    out_writer.release()

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Done! {processed} frames in {elapsed:.1f}s ({processed/elapsed:.1f}fps)")
    if t_infer_list:
        print(f"Inference: avg={np.mean(t_infer_list):.1f}ms max={max(t_infer_list):.1f}ms")
    print(f"Frames with detections: {det_frames}")
    print(f"Video: {video_out_path}")
    print(f"Single frames: {output_dir}")


if __name__ == '__main__':
    main()
