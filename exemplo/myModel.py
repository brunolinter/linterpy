import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Load the trained model
loaded_model = load_model('vgg16_trained_model.h5')

# Function to preprocess new images
def preprocess_image(image_path, target_size=(224, 224), color_scale_factor=1./255):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_scaled = img_array * color_scale_factor
    img_batch = np.expand_dims(img_array_scaled, axis=0)
    return img_batch

# The paths to the new images to test
test_image_paths = ['path/to/test/image1.jpg', 'path/to/test/image2.jpg']

for path in test_image_paths:
    # Preprocess the image
    preprocessed_image = preprocess_image(path)
    # Predict
    predictions = loaded_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=-1)
    print(f"Predicted class for {path}: {predicted_class}")
