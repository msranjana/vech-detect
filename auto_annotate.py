from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_DIR = Path("accuracy_test/frames")
LABEL_DIR = Path("accuracy_test/labels")
PREVIEW_DIR = Path("accuracy_test/preview")

MODEL_NAME = "yolo11l.pt"

# COCO classes:
# car = 2
# motorcycle = 3
# bus = 5
# truck = 7
VEHICLE_CLASSES = {
    2: 0,  # car -> our class 0
    3: 1,  # motorcycle -> our class 1
    5: 2,  # bus -> our class 2
    7: 3,  # truck -> our class 3
}

CONFIDENCE = 0.25
IMAGE_SIZE = 1280


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

LABEL_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading model:", MODEL_NAME)

model = YOLO(MODEL_NAME)

print("Model loaded.")


# --------------------------------------------------
# PROCESS 100 FRAMES
# --------------------------------------------------

images = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

print("Images found:", len(images))

for index, image_path in enumerate(images, start=1):

    print(f"[{index}/{len(images)}] {image_path.name}")

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        classes=[2, 3, 5, 7],
        device="cpu",
        verbose=False,
        save=False,
    )

    result = results[0]

    # YOLO label file
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    lines = []

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            # Convert COCO class to our class
            if cls not in VEHICLE_CLASSES:
                continue

            new_class = VEHICLE_CLASSES[cls]

            # xywh normalized
            x, y, w, h = boxes.xywhn[i].tolist()

            lines.append(
                f"{new_class} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

    # Save YOLO annotation
    label_path.write_text(
        "\n".join(lines) + ("\n" if lines else "")
    )

    # Save visual preview
    result.save(
        filename=str(PREVIEW_DIR / image_path.name)
    )

print("\n===================================")
print("AUTO ANNOTATION COMPLETE")
print("===================================")
print("Images :", len(images))
print("Labels :", LABEL_DIR)
print("Preview:", PREVIEW_DIR)
