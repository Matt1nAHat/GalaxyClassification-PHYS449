import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib  # For model persistence

# Load your data
# Replace 'your_data.csv' with the actual CSV file containing your data
data = pd.read_csv('your_data.csv')

# Load and preprocess your images
# Replace 'load_and_preprocess_image' with your actual image loading and preprocessing function
def load_and_preprocess_image(image_path):
    # Your image loading and preprocessing code here
    # For example, using OpenCV
    # Make sure to adapt this according to your actual image loading and preprocessing needs
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    return processed_image.flatten()

# Assuming you have a column 'filename' in your CSV with the file paths
image_paths = data['filename'].tolist()
X = np.array([load_and_preprocess_image(path) for path in image_paths])

# Encode the class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['class'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an Extra Trees classifier
model = ExtraTreesClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model for future use
joblib.dump(model, 'galaxy_classifier_model.pkl')
