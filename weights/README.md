# weights/

Model artifacts live here. All weight files (`*.pt`, `*.onnx`, `*.engine`)
are **gitignored** — only this README is tracked.

Expected files after running the pipeline:

| File                 | Produced by                          | Used by                    |
|----------------------|--------------------------------------|----------------------------|
| `best.pt`            | training (Colab T4)                  | ONNX export, GPU tracking  |
| `best.onnx`          | `src/optimization/export_onnx.py`    | INT8 quant, benchmark      |
| `best.int8.onnx`     | `src/optimization/quantize_int8.py`  | **CPU demo**, benchmark    |
| `best.int8.engine`   | `src/optimization/build_tensorrt.py` (GPU) | benchmark (GPU)      |

Drop the trained `best.pt` here after the Colab run, then proceed with
Phase 4 (optimization).
