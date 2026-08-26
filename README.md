# YOLO26n vs YOLO11n: Small-Object Detection Benchmark on VisDrone

A head-to-head benchmark of Ultralytics' YOLO26n against YOLO11n, fine-tuned on the same aerial-imagery dataset under identical conditions, then exported to ONNX and deployed as a live CPU-only demo.

**Live demo:** https://daudshah-yolo26-visdrone-demo.hf.space

## Why this comparison

YOLO26 (Ultralytics, January 2026) makes two specific, testable claims: native end-to-end NMS-free inference, and training changes (Small-Target-Aware Label Assignment, Progressive Loss Balancing) aimed at small and distant objects. VisDrone is dominated by exactly that kind of object, tiny, distant pedestrians and vehicles seen from a drone, so it's a reasonable place to check whether those claims hold up outside a vendor benchmark.

## Method

Both models were fine-tuned from their pretrained nano checkpoints on the same VisDrone2019-DET split, 30 epochs, same image size (640), same optimizer selection (`auto`, which picked AdamW for both), same seed. The only variable between the two runs is model architecture. Full training logs and configs are in the phase notebooks.

## Results

| | mAP50-95 | CPU latency (mean) | FPS |
|---|---|---|---|
| YOLO26n | 0.145 | 69.9 ms | 14.3 |
| YOLO11n | 0.158 | 82.4 ms | 12.1 |

CPU latency measured with ONNX Runtime, `CPUExecutionProvider`, on a Kaggle Xeon CPU, 100 runs after 5 warmup runs.

**Neither model wins outright.** YOLO11n is more accurate at this epoch budget. YOLO26n is about 15% faster on CPU and gets about 18% more FPS. That's a real accuracy/speed trade-off, not a clean win for the newer model, and it's worth saying plainly: this doesn't match Ultralytics' own published claim of up to 43% faster CPU inference, likely because that figure was measured on different hardware, a different opset, or with OpenVINO, none of which this benchmark used. The 15% here is what was actually measured on this setup, not a restated vendor number.

One architectural difference is visible directly in the training logs, not just claimed: YOLO26n's loss columns show `box_loss cls_loss l1_loss`, no `dfl_loss`, confirming Distribution Focal Loss removal. YOLO11n's columns include `dfl_loss` as expected.

## Repo structure

```
notebooks/
  phase1_setup_sanity_check.ipynb   # environment check, 1-epoch sanity run on 10% of data
  phase2_full_training.ipynb        # full 30-epoch training, both models
  phase3_export_benchmark.ipynb     # ONNX export, CPU latency benchmark
  phase4a_test_before_deploy.ipynb  # local inference test before deployment
app/
  app.py                            # Gradio demo, raw ONNX Runtime inference
  requirements.txt
```

## Reproducing this

Dataset: VisDrone2019-DET, auto-downloaded by Ultralytics' built-in `VisDrone.yaml` config, no manual download needed.

```bash
pip install ultralytics onnxruntime gradio

yolo train model=yolo26n.pt data=VisDrone.yaml epochs=30 imgsz=640 device=0
yolo train model=yolo11n.pt data=VisDrone.yaml epochs=30 imgsz=640 device=0
```

Full commands with path handling, ONNX export, and the benchmark script are in the numbered notebooks.

## Deployment notes

The demo runs on Hugging Face Spaces' ZeroGPU tier (free CPU Basic Spaces are no longer available for new Gradio Spaces as of mid-2026, so `@spaces.GPU` is required for HF to accept the Space at startup). Inference itself still runs on CPU via `onnxruntime`'s `CPUExecutionProvider`, unchanged from the benchmark above, the decorator is a hosting requirement, not a change to what was measured.

## License

MIT
