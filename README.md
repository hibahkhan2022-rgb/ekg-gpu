# Realtime EKG Anomaly Detection

# ✅ Project Roadmap — Realtime EKG Anomaly Detector (CUDA/MPS/ONNX)

This checklist keeps the project on track for the next 8 weeks.

---

## Week 1 — Scaffold + Data In
- [X] Create repo scaffold
- [ ] `io.py`: download + load PhysioNet dataset, segment windows, train/val/test split.
- [ ] `dsp_cpu.py`: bandpass FIR + moving average filter.
- [ ] `01_explore_data.ipynb`: plot raw + filtered signals.
- [ ] README: add project goal + empty latency table.

---

## Week 2 — MPS Path + Tiny Model
- [ ] `dsp_mps.py`: reimplement filters with PyTorch ops on MPS.
- [ ] `model.py`: define lightweight 1D-CNN or BiLSTM (<100k params).
- [ ] `02_training.ipynb`: train on MPS backend, save best checkpoint.
- [ ] Verify baseline accuracy ≥ 85–90%.

---

## Week 3 — Unified Inference + API
- [ ] `inference.py`: `predict(window, backend="cpu"|"mps"|"cuda")`.
- [ ] `app.py`: FastAPI `/predict` endpoint.
- [ ] `client.py`: test script for sending windows.
- [ ] Prove round-trip latency on MPS < 250 ms (initial target).
- [ ] README: add “How to run locally” section.

---

## Week 4 — Benchmarks v1 (Mac)
- [ ] `benchmarks.py`: measure filter, model, and end-to-end latency (CPU vs MPS).
- [ ] Add results (table/plots) to README.
- [ ] `02_training.ipynb`: confusion matrix for accuracy.
- [ ] Draft ablation plan (e.g., window size or filter taps).

---

## Week 5 — CUDA Pass (NVIDIA GPU)
- [ ] Run `benchmarks.py` with `backend="cuda"` on Colab or RTX box.
- [ ] `cuda/cupy_fir.py`: implement FIR filter with CuPy/Numba.
- [ ] Optional: `stft_kernel.cu` experiment.
- [ ] Add CUDA numbers to latency table.

---

## Week 6 — Interop (ONNX + Core ML)
- [ ] `export_onnx.py`: export to ONNX, run test inference with `onnxruntime`.
- [ ] `export_coreml.py`: export `.mlmodel`.
- [ ] README: add instructions for ONNX + Core ML.

---

## Week 7 — Polish + Ablation + Demo
- [ ] Run 1 ablation (e.g., FIR taps or model width) → plot latency vs accuracy.
- [ ] Tighten FastAPI latency (<150 ms on MPS).
- [ ] Record 2–3 min demo (benchmarks + one API call).
- [ ] Add architecture diagram to README.
- [ ] Near-final README polish.

---

## Week 8 — Ship + Conference Ready
- [ ] Final latency table (CPU, MPS, CUDA).
- [ ] Repo passes fresh clone test (`pip install -r requirements.txt`, run server, run benchmarks).
- [ ] Résumé bullet updated with results.
- [ ] Optional: Publish LinkedIn/blog post with demo + insights.

---

## 🚩 Success Bars
- End-to-end latency (MPS): ≤ 150 ms per 1-sec window (target 100 ms).
- CUDA: ≥ 3× faster vs CPU baseline.
- ONNX: loads + runs inference.
- Core ML: exports successfully.
- FastAPI: 100 sequential requests stable.
