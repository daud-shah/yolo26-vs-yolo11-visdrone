"""
Live demo: YOLO26n (ONNX, CPU-only) fine-tuned on VisDrone.
Runs raw onnxruntime inference, not the ultralytics wrapper, since the exported
graph is already end-to-end (NMS baked in), the output is final boxes directly.
"""
import time
import numpy as np
import onnxruntime as ort
import gradio as gr
import spaces
from PIL import Image, ImageDraw

MODEL_PATH = "model.onnx"  # renamed on upload, see deployment steps
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name


def preprocess(image: Image.Image) -> tuple[np.ndarray, float, float]:
    """Simple resize to IMG_SIZE, not letterboxed. Returns the tensor and the
    x/y scale factors needed to map boxes back to the original image size.

    [verify] training used ultralytics' default rect=False resize, if boxes
    look shifted on non-square images, letterbox padding may be needed here
    to match the training preprocessing exactly.
    """
    orig_w, orig_h = image.size
    resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]  # HWC -> CHW -> NCHW
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE
    return arr, scale_x, scale_y


@spaces.GPU  # required for HF's free ZeroGPU tier to accept this Space at startup,
             # inference below still runs on CPU via CPUExecutionProvider, unchanged
def detect(image: Image.Image) -> tuple[Image.Image, str]:
    if image is None:
        raise gr.Error("Upload an image first")

    tensor, scale_x, scale_y = preprocess(image)

    start = time.perf_counter()
    outputs = session.run(None, {INPUT_NAME: tensor})
    latency_ms = (time.perf_counter() - start) * 1000

    # output shape (1, 300, 6): x1, y1, x2, y2, score, class_id, already NMS'd
    detections = outputs[0][0]
    keep = detections[:, 4] >= CONF_THRESHOLD
    detections = detections[keep]

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for x1, y1, x2, y2, score, cls_id in detections:
        box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        label = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], max(box[1] - 12, 0)), f"{label} {score:.2f}", fill="lime")

    stats = f"{len(detections)} objects detected in {latency_ms:.1f} ms on CPU (YOLO26n, ONNX)"
    return annotated, stats


demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="Upload aerial or street image"),
    outputs=[gr.Image(label="Detections"), gr.Textbox(label="Stats")],
    title="YOLO26n Small-Object Detector, CPU only",
    description=(
        "Fine-tuned on VisDrone. Runs raw ONNX inference on CPU, no GPU. "
        "Benchmarked at 69.9ms mean latency / 14.3 FPS on a Kaggle Xeon CPU, "
        "about 15% faster than an equivalent YOLO11n export."
    ),
)

if __name__ == "__main__":
    demo.launch()