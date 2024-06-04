# Sign Language Recognition

## Introduction

Sign language recognition involves converting sign language gestures into text or speech. This project focuses on implementing American Sign Language (ASL) translation using landmark detection to convert ASL gestures into text by analyzing hand movements through detected landmark points.

The dataset includes diverse hand orientations, lighting conditions, and backgrounds. It comprises 36 hand gestures, each with 150 samples, captured using a webcam.

## Methodology

### Dataset Creation & Preprocessing
- **Dataset**: Collected a dataset of 36 alphanumeric sign language gestures, each with 150 images, totaling 5,400 data points.
- **Data Collection**: Utilized webcam images to capture hand landmarks.
- **Preprocessing**: Processed images to ensure consistency.

### Image Processing & Landmark Detection
- **Tools**: Employed MediaPipe Hands and OpenCV for precise hand landmark detection.
- **Image Conversion**: Processed images to RGB format and extracted x and y coordinates for feature extraction.

### Labeling & Data Serialization
- **Labeling**: Assigned labels based on gesture directories.
- **Serialization**: Organized and serialized the processed data into a pickle file for efficient storage and retrieval.

### Model Training & Evaluation
- **Classifier**: Utilized a Random Forest classifier for training and testing.
- **Feature Extraction**: Extracted relevant features from landmark data.
- **Performance Metrics**: Assessed model performance using accuracy metrics, achieving a validation accuracy of approximately 93%.

## Project Architecture

1. **Input**: Frame from video
2. **Detection**: 
   - Origin Transform
   - Calculate Distance
   - New Coordinates + Distances + Handedness
3. **Machine Learning Model**: Random Forest Classifier
4. **Output**: Predicted Class

## Results & Conclusion

The project achieved a validation accuracy of approximately 93%, indicating that the Random Forest classifier demonstrated robustness in recognizing sign language gestures. The use of OpenCV and MediaPipe facilitated accurate identification of hand gestures.

## Team Members

- **Apurva Kamble** - [GitHub](https://github.com/apurva-1403)
- **Ayushi Agarwal** - [GitHub](https://github.com/agarwalayushi2102)
- **Shwetha Gajula** - [GitHub](https://github.com/Shwetha1011)

## References

- [Sign Language Recognition with Mediapipe Hands](https://techcrunch.com/2019/08/19/this-hand-tracking-algorithm-could-lead-to-sign-language-recognition/)
- [IEEE Research Paper](https://ieeexplore.ieee.org/document/9908995)

## Repository Links

- [GitHub Repository](https://github.com/apurva-1403/Sign-Language-Recognition)

---

Thank you for exploring the Sign Language Recognition project!
