# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A YOLO (Ultralytics) object detector for 5 Thai chocolate snack brands: `beng-beng`, `bon o bon`, `kalpa`, `milky snack`, `sumo`. The pipeline goes: video frames, then Label Studio annotation, then YOLO dataset, then training, then inference on an image, a video, or a webcam. There is no git repo, no test suite and no linter. Code comments and console messages are in Thai. Keep that style when you edit the existing scripts.

## Environment

- Windows, Python 3.13, NVIDIA GPU with CUDA 12.8. The virtualenv is in `env/`. Activate it with `env\Scripts\Activate.ps1` (PowerShell), or call `env/Scripts/python.exe` directly.
- To install: `pip install -r requirements.txt`. The file pins `torch==2.11.0+cu128`, `ultralytics==8.4.115` and `label-studio==1.23.0`, and pulls from the PyTorch cu128 index.
- Training and camera inference hard-code `device=0` (GPU).

## Pipeline (numbered scripts, run from the repo root)

All paths are relative to the repo root, so run the scripts from there.

1. `python 01-export_dataset.py [--json FILE] [--images-dir frame/images] [--output-dir dataset] [--train-split 0.8] [--seed 42]`
   Converts a Label Studio JSON export into a YOLO detect dataset.
   - If `--json` is omitted, the script needs exactly **one** `*.json` in the repo root (currently `project-4-at-...json`). Adding a second JSON makes it exit with an error.
   - Source images come from `frame/images/`, which holds frames extracted from `train_chocolate_treats.mp4` (no extraction script is included). Images are matched by basename, taken from the task's `data.image` URL (`?d=` local-files paths are handled).
   - Label Studio rotated rectangles become the **axis-aligned bounding box of the rotated corners**. The rotation is computed in pixel space because width and height percentages differ on non-square images.
   - Class IDs come from the **sorted** set of labels. Adding a new label can shift the existing IDs, which makes old weights and labels incompatible.
   - The script **wipes** `dataset/{images,labels}/{train,val}` on every run. It then regenerates `dataset/data.yaml` with an absolute `path:` and writes `dataset/classes.txt`.
2. `python 02-train.py` fine-tunes `yolo26s.pt` on `dataset/data.yaml`: up to 200 epochs with `patience=60`, imgsz 640, batch 16, `optimizer="MuSGD"`, `cos_lr=True`, `close_mosaic=20`, with extra rotation/shear/perspective/mixup augmentation because the training footage was shot from a single angle. Ultralytics writes each run to a new folder, `runs/detect/train-N/`.
3. `python 03-test_image.py` predicts on `test.jpg` (conf 0.5, save).
4. `python 04-test_video.py` predicts on `test_chocolate_treats_video.mp4`, then saves and shows the result.
5. `python 05-test-camera.py` runs real-time webcam detection with OpenCV DirectShow at 1280x720 MJPG. Press `q` to quit.

The inference scripts (03–05) hard-code the weights path `runs/detect/train-3/weights/best.pt`. After a new training run, update that path in all three scripts.

The Ultralytics CLI works too, for example `yolo detect predict model=runs/detect/train-3/weights/best.pt source=test.jpg conf=0.5`.

## Current model

`runs/detect/train-3` (yolo26s, stopped early at epoch 148; `best.pt` is from epoch 88) was trained on 228 images: 147 frames from `train_chocolate_treats.mp4` plus 81 close-up, multi-angle frames (`bonobon_*.jpg`) from `bon o bon.mp4.MOV`. On the 46-image val split it scores P 0.962, R 0.959, mAP50 0.971 and mAP50-95 0.614, and every class is above 0.9 in P, R and mAP50. It detects bon o bon in 93% of the frames of the bon o bon clip and all 5 brands on `test.jpg`. The previous `train-v2` scores mAP50 0.970 / mAP50-95 0.599 on the same split, but bon o bon recall is only 0.717 and it detects bon o bon in 38% of the clip's frames. The val images come from the same videos as the train images, so these numbers are optimistic.

The new JSON export (`project-4-at-2026-10-04-...json`) sits beside the old one, so `01-export_dataset.py` needs `--json`. The bon o bon images were drag-and-drop uploads (`/data/upload/...` with an 8-hex prefix), which `resolve_image_file()` strips when matching files in `frame/images/`.
