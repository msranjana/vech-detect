# Crowd person-detection accuracy dataset

Used by `quick_accuracy_crowd.py` (not `quick_accuracy.py`).

- `frames/` — `.jpg` or `.png` images  
- `labels/` — one `.txt` per image, YOLO format, **class `0` = person**

Panic / stampede **behavior** accuracy is not covered here; that needs time-segment labels on video.
