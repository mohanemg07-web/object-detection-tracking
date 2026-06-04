# notebooks/

GPU notebooks meant to run on a **free Kaggle/Colab T4**.

- `train_kaggle.ipynb` — fine-tune YOLOv8m on VisDrone-DET with MLflow
  logging to DagsHub. (Added in Phase 3.)
- `tensorrt_int8.ipynb` — build + calibrate a TensorRT INT8 engine and
  benchmark it against ONNX-INT8. GPU-only. (Added in Phase 4.)

These notebooks mirror the standalone scripts in `src/` so every
GPU-dependent step is reproducible without a local GPU.
