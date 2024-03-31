import cv2
import os
from background_removal import SelfiSegmentation

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Initialize SelfiSegmentation for background removal
segmentor = SelfiSegmentation()

# Load custom background images
bg_images_dir = "images"
bg_images = [cv2.imread(os.path.join(bg_images_dir, filename)) for filename in os.listdir(bg_images_dir)]

index_img = 0

while True:
    # Capture a frame from the webcam
    success, img = cap.read()
    if not success:
        break

    # Remove the original background and replace it with a custom background image
    img_out = segmentor.removeBG(img, bg_images[index_img], cutThreshold=0.1)

    # Stack the original image and the image with background removed side by side
    img_stacked = cv2.hconcat([img, img_out])

    # Display the stacked images
    cv2.imshow("Virtual Background", img_stacked)

    # Wait for a key press
    key = cv2.waitKey(1)

    # Change background image on 'a' or 'd' key press
    if key == ord('a'):
        index_img = (index_img - 1) % len(bg_images)
    elif key == ord('d'):
        index_img = (index_img + 1) % len(bg_images)
    # Exit loop on 'q' key press
    elif key == ord('q'):
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
