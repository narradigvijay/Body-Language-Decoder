Real-Time Body Language Classifier 🕺🤖
A production-grade machine learning application that utilizes Computer Vision and Classification Algorithms to detect and categorize human body language and emotions in real-time.

🌟 Overview
This project implements a full end-to-end machine learning pipeline. It uses MediaPipe Holistic to extract 543 unique landmarks (Face, Hands, and Pose) and feeds them into a Scikit-learn pipeline to perform real-time classification.

The system can distinguish between several emotional states, including:

Happy (Smiling, open posture)

Sad (Slumped shoulders, frowning)

Victorious (Arms raised, V-signs)

Angry (Tense posture, furrowed brow)

Normal (Neutral baseline state)

🛠️ Technical Stack
Computer Vision: MediaPipe (Holistic API)

Machine Learning: Scikit-learn

Data Processing: Pandas, NumPy

Interface: OpenCV

Evaluation: Seaborn, Matplotlib

📈 Performance & Evaluation
The model was evaluated using 10-Fold Cross-Validation to ensure the accuracy was robust and not a result of overfitting.

Best Model: Logistic Regression (within a StandardScaler Pipeline)

Average Accuracy: ~96%

Evaluation Metrics: * Precision/Recall: Balanced across all classes.

Confusion Matrix: High diagonal density with minimal confusion between similar classes (e.g., Sad vs. Normal).

🚀 How It Works
1. Feature Extraction
Using MediaPipe Holistic, we capture a total of 1,629 data points (x, y, z, and visibility) per frame. This includes:

Face Mesh: 468 landmarks.

Pose: 33 landmarks.

Hands: 21 landmarks per hand.

2. Model Training
The landmarks are flattened into a CSV format and trained using a Scikit-learn pipeline. The pipeline includes a StandardScaler to normalize the coordinate data, ensuring that the distance from the camera doesn't break the model.

3. Real-Time Inference
The trained model (.pkl file) is loaded into a live OpenCV loop. Every frame from the webcam is processed, landmarks are extracted, and the model predicts the emotion with an associated probability score.

📂 Project Structure
Bash
├── .ipynb_checkpoints/
├── coords.csv              # The raw landmark dataset
├── body_language.pkl      # The final trained model
├── Body_Language_Decoder.ipynb  # Main development notebook
└── README.md              # Project documentation
🔧 Installation & Usage
Clone the repo:

Bash
git clone https://github.com/YOUR_USERNAME/Body-Language-Classifier.git
Install dependencies:

Bash
pip install mediapipe opencv-python pandas scikit-learn
Run the Notebook: Open Body_Language_Decoder.ipynb and run the cells to start the live webcam feed.

📝 References
Bazarevsky et al. (2020). BlazePose: On-device Real-time Body Pose Tracking.

Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python.
