import torch
import cv2
import matplotlib.pyplot as plt
import argparse

#Going to image and predetermining the label threshold
def detect_diamond(image_path, conf_threshold=0.50):
    
    #Loading the model (suggestions from YOLOv5 Paper)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    #Grabbing Image if available, then changing to RGB for model
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No Image Found: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    results = model(img_rgb)
    detections = results.xyxy[0].cpu().numpy()
    
    #Creates boxes around areas of "diamonds" + Labelling
    diamond_boxes = []
    for *box, conf, cls in detections:
        label = model.names[int(cls)].lower()
        if label in ['diamond', 'jewelry'] and conf >= conf_threshold:
            box = list(map(int, box))
            diamond_boxes.append(box)

            x1, y1, x2, y2 = box
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_rgb, "Diamonds", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    #Showing the output
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Detected Diamond Area")
    plt.show()
    
    return diamond_boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting Diamonds with YOLOv5")
    parser.add_argument("--image", type=str, required=True, help="Path to feed image")
    args = parser.parse_args()
    
    boxes = detect_diamond(args.image)
    print("Detected Diamond Area:", boxes)
