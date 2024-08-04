from roboflow import Roboflow
import cv2
import os

rf = Roboflow(api_key="regiQVSqqhJsWVAUxUSS")
project = rf.workspace().project("parasite-detection-x5vzb")
model = project.version(4).model
# Input folder containing images
input_folder = r'D:\PROJECT\confusion matrix\normal'

# Output folder for saving detection results
output_folder = r'D:\PROJECT\confusion matrix\result-normal'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png','.tiff')):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Save the image with all bounding boxes
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.jpg")
        # visualize your prediction
        model.predict(r"D:\PROJECT\confusion matrix\normal\20170728_201531.tiff", confidence=40, overlap=30).save(
            output_path)

# Close any remaining windows
cv2.destroyAllWindows()
