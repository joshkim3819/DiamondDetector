import torch
import cv2
import matplotlib.pyplot as plt
import argparse

# Detecting diamonds from images since the images could be of diamond rings or necklaces
def detect_diamond(image_path, conf_threshold=0.5, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading the YOLOv5 small model (yolov5s) from Torch Hub with no changes in the pre-trained COCO weights
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device).eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv5 model: {e}")

    # Load image from the provided path using OpenCV in BGR format
    # Changing from BGR to RGB since YOLOv5 works with RGB
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No Image Found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLOv5 model on the image to detect objects, which includes bounding boxes, confidence scores, and semantic labeling
    results = model(img_rgb)
    detections = results.xyxy[0].cpu().numpy()

    # Filtering Labels based on confidence levels and semantic labels
    diamond_boxes = []
    target_labels = ['diamonds']
    for *box, conf, cls in detections:
        label = model.names[int(cls)].lower()
        if label in target_labels and conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            diamond_boxes.append([x1, y1, x2, y2])

            # Drawing and Labeling the diamonds
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_rgb, label.capitalize(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Visualize the image with boxes and labels using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Detected {', '.join(target_labels)} Areas")
    plt.show()

    # Return the list of detected bounding boxes
    return diamond_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects with YOLOv5")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")  # Required image path
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")  # Optional threshold
    args = parser.parse_args()  # Parse the arguments

    # Print the detected bounding boxes, which is what we want
    boxes = detect_diamond(args.image, conf_threshold=args.conf)
    print("Detected Boxes:", boxes)
