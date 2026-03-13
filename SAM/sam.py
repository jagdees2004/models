import os
# Suppress TF and OpenMP warnings before imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

def get_generator():
    try:
        from ultralytics import YOLO
        # Provide the name of the YOLOv8 segmentation weights file.
        # It'll download automatically if it doesn't exist locally, classifying objects like 'dog', 'person', etc.
        model = YOLO('yolov8n-seg.pt')
        return model
    except ImportError:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

def generate_masks_from_image(generator, image):
    # Resize the image for even faster performance if needed
    image.thumbnail((640, 640))
    
    # Run the model
    results = generator(image, device='cpu', retina_masks=True, conf=0.4, iou=0.9)
    
    # Extract the natively plotted image with all overlapping masks colored correctly
    if not results:
        import numpy as np
        return np.array(image), []
        
    # results[0].plot() returns BGR numpy array
    annotated_image_bgr = results[0].plot()
    
    # Convert bool/float tensor to RGB numpy for displaying
    annotated_image_rgb = annotated_image_bgr[:, :, ::-1]
    
    # Extract the class names
    detected_classes = []
    if results[0].boxes:
        names_dict = results[0].names
        # Get the class IDs from the boxes
        class_ids = results[0].boxes.cls.cpu().numpy()
        detected_classes = [names_dict[int(c)] for c in class_ids]
    
    # Remove duplicates but preserve order for a clean list
    unique_classes = list(dict.fromkeys(detected_classes))
    
    return annotated_image_rgb, unique_classes

def main():
    print("Loading FastSAM (Fast Segment Anything Model)...\n")
    
    try:
        generator = get_generator()
    except Exception as e:
        print(f"  ⚠ Error initializing model: {e}")
        return

    print('Type an image path to generate masks. Type "quit" to exit.\n')

    while True:
        try:
            user_input = input("Image Path ► ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
            
        # Strip quotes if dragged and dropped in terminal
        image_path = user_input.strip('"\'')
        
        if not os.path.isfile(image_path):
            print(f"  ⚠ Could not find image: {image_path}\n")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            annotated_img, detected_classes = generate_masks_from_image(generator, image)
            
            if detected_classes:
                print(f"AI  ► Successfully generated segmentation masks for: {', '.join(detected_classes)}\n")
            else:
                print("AI  ► Successfully generated segmentation masks, but no specific objects were recognized.\n")
            
            # Displaying the image with masks
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 10))
            plt.imshow(annotated_img)
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"  ⚠ Error processing image: {e}\n")


if __name__ == "__main__":
    main()
