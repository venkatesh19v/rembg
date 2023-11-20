from PIL import Image
import cv2
import os
from rembg import remove
import random

def main():
    os.makedirs('py_org', exist_ok=True)
    os.makedirs('py_masked', exist_ok=True)
    cv2_img = capture_image()
    if cv2_img is not None:
        local_filename = 'py_org/captured_image.jpg'
        cv2.imwrite(local_filename, cv2_img)
        print(f"Image saved as {local_filename}")
        selected_background =   select_background()

        if selected_background:
            background_file = selected_background
            threshold = 50 
            subject_file = local_filename
            output_file = 'py_masked/' + os.path.basename(subject_file)
            with open(output_file, 'wb') as f:
                subject_img = open(subject_file, 'rb').read()
                subject = remove(subject_img, alpha_matting=True, alpha_matting_foreground_threshold=threshold)
                f.write(subject)

            background_img = Image.open(background_file)
            subject_img = Image.open(output_file)

            background_img = background_img.resize(subject_img.size)
            merged_image = Image.alpha_composite(background_img.convert("RGBA"), subject_img)
            merged_image_path = 'py_masked/background.png'
            merged_image.save(merged_image_path, format='PNG')

            print("Merged image saved as", merged_image_path)
        else:
            print("Please select a background.")
    else:
        print("Please capture an image.")

def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        return None

def select_background():
    background_images = [
        "/home/tenet/Downloads/Rev_BG/image-background-removal-and-replacement-using-ml-and-ai/final rembg/Seashore.jpg",
        "/home/tenet/Downloads/Rev_BG/image-background-removal-and-replacement-using-ml-and-ai/final rembg/forest.jpg",
        "/home/tenet/Downloads/Rev_BG/image-background-removal-and-replacement-using-ml-and-ai/final rembg/Space.jpg",
        "/home/tenet/Downloads/Rev_BG/image-background-removal-and-replacement-using-ml-and-ai/final rembg/flowerland.jpg",
        "/home/tenet/Downloads/Rev_BG/image-background-removal-and-replacement-using-ml-and-ai/final rembg/Himalaiya.jpg",
        "/home/tenet/Downloads/Rev_BG/image-background-removal-and-replacement-using-ml-and-ai/final rembg/landscape.jpeg"
    ]
    return random.choice(background_images)
if __name__ == "__main__":
    main()
