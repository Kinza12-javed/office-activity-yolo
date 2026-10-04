# Office Activity Detector (Working / Notworking)

A YOLO26n model fine-tuned with transfer learning (COCO pretrained) to draw a box around each
person in an office frame and label them Working or Notworking.

## Demo
[LinkedIn post link or video link]

## Dataset
Employee Activity Detection by Kushal Workspace, Roboflow Universe, version 7, CC BY 4.0.
https://universe.roboflow.com/kushal-workspace/employee-activity-detection
1,688 images, 2 classes. Split: train 1,178 / valid 333 / test 177.

## Method
- Ultralytics YOLO26n on Google Colab (T4 GPU), image size 640
- Round 1: first 10 layers frozen, early stopping (patience 10), best epoch 47
- Round 2: from Round 1 best, all layers unfrozen, AdamW lr0=0.001, patience 10, best epoch 31
- Demo: COCO person detector + tracking + label smoothing, so each person gets one stable box

## Results

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Round 1 (valid) | 0.959 | 0.927 | 0.962 | 0.675 |
| Round 2 (valid) | 0.961 | 0.949 | 0.969 | 0.695 |
| Round 2 (test) | 0.956 | 0.949 | 0.972 | 0.674 |

## Limitations
- Scores are on the same dataset. On a new office video the model made mistakes
  (for example, labeling a person at work as Notworking).
- The training data looks like it comes from a single camera and has no empty background images,
  so boxes can appear on empty spots in new scenes.
- The model only sees body pose; it cannot know whether someone is actually working.
- This is an activity-detection demo, not a tool to score anyone's performance.

## Files
- office_activity_yolo26n.pt: final model
- training.ipynb: Colab notebook

## Credits
Dataset: Kushal Workspace (CC BY 4.0). Model: Ultralytics YOLO.
