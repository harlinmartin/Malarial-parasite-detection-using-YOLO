from PIL import Image
from roboflow import Roboflow

# Initialize Roboflow client
rf = Roboflow(api_key="regiQVSqqhJsWVAUxUSS")
l
# Load the image and resize it
image_path = r"D:\PROJECT\confusion matrix\parasite\20170724_163653.jpg"
image = Image.open(image_path)
# Resize the image to a smaller size (e.g., 50% of the original dimensions)
resized_image = image.resize((int(image.width * 0.5), int(image.height * 0.5)))

# Save the resized image to a temporary file
resized_image_path = "resized_image.jpg"
resized_image.save(resized_image_path)

# Load Roboflow project and model
project = rf.workspace().project("parasite-detection-x5vzb")
model = project.version(4).model

# Predict on the resized image
prediction = model.predict(resized_image_path, confidence=40, overlap=30)

# Save the prediction visualization
prediction.save("prediction.jpg")
