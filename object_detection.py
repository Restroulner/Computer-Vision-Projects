import cv2
import numpy as np

class ObjectDetector:
    """
    A simple object detector using a pre-trained model (e.g., MobileNet SSD).
    This is a conceptual implementation for demonstration purposes.
    """
    def __init__(self, model_path, config_path, labels_path, confidence_threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        with open(labels_path, 'r') as f:
            self.labels = f.read().strip().split('\n')
        self.confidence_threshold = confidence_threshold

    def detect(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{self.labels[idx]}: {confidence:.2f}"
                results.append({
                    "label": label,
                    "box": (startX, startY, endX, endY),
                    "confidence": confidence
                })
        return results

    def draw_detections(self, image, detections):
        for det in detections:
            (startX, startY, endX, endY) = det["box"]
            label = det["label"]
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

if __name__ == "__main__":
    # Dummy paths for demonstration (replace with actual model files)
    dummy_model_path = "dummy_model.caffemodel"
    dummy_config_path = "dummy_config.prototxt"
    dummy_labels_path = "dummy_labels.txt"
    dummy_image_path = "dummy_test_image.jpg"

    # Create dummy files
    with open(dummy_model_path, "w") as f: f.write("dummy model content")
    with open(dummy_config_path, "w") as f: f.write("dummy config content")
    with open(dummy_labels_path, "w") as f: f.write("background\nperson\ncar\ncat\ndog")
    dummy_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.imwrite(dummy_image_path, dummy_image)

    print("Starting object detection example...")
    try:
        detector = ObjectDetector(dummy_model_path, dummy_config_path, dummy_labels_path)
        image = cv2.imread(dummy_image_path)
        
        # Simulate a detection
        # In a real scenario, detections would come from the model
        simulated_detections = [
            {"label": "person: 0.95", "box": (50, 50, 150, 250), "confidence": 0.95},
            {"label": "car: 0.88", "box": (300, 200, 500, 350), "confidence": 0.88}
        ]

        output_image = detector.draw_detections(image.copy(), simulated_detections)
        cv2.imwrite("output_detections.jpg", output_image)
        print("Object detection example finished. Output saved to output_detections.jpg")

    except Exception as e:
        print(f"Error during object detection example: {e}")
    finally:
        # Clean up dummy files
        os.remove(dummy_model_path)
        os.remove(dummy_config_path)
        os.remove(dummy_labels_path)
        os.remove(dummy_image_path)
        if os.path.exists("output_detections.jpg"):
            os.remove("output_detections.jpg")
