
# CNN Model for Plastic Waste Classification


## Overview 
This project focuses on building a Convolutional Neural Network (CNN) model to classify images of plastic waste into different categories. The goal is to contribute to efficient waste management by automating the segregation process through deep learning technologies.



## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Weekly Progress](#weekly-progress)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Future Scope](#future-scope)


## Project Description
Plastic pollution poses a significant environmental challenge, and efficient waste segregation plays a crucial role in addressing this issue. This project uses a CNN-based approach to classify plastic waste into various categories, facilitating automated waste management and recycling processes.
##  Dataset

The dataset used in this project is the Waste Classification Data by Sashaank Sekar, available on Kaggle. It contains labeled images divided into two primary categories: Organic and Recyclable.

### Dataset Details 
- Total Images: 25,077
  * Training Data: 22,564 images (85%)
  * Test Data: 2,513 images (15%)
- Classes: Organic and Recyclable
- Purpose: To assist in automating waste segregation processes using machine learning.

### Dataset Link 
Access the dataset here: Waste Classification Data.     
**Note :** Please adhere to the dataset licensing and usage guidelines.

## Approach
The project followed these steps:

1.  **Problem Understanding:** Researched plastic waste segregation challenges and opportunities for automation using deep learning.

2. **Data Preparation:** Processed the dataset by resizing, normalizing, and augmenting images to improve model performance.

3. **Model Design:** Developed a CNN with layers for feature extraction, pooling, and classification using ReLU and Softmax activations.

4. **Training:** Trained the model using the Adam optimizer and categorical crossentropy loss with hyperparameter tuning.

5. **Evaluation:** Evaluated the model using test data and performance metrics like accuracy and F1 score.

6. **Future Enhancements:** Explored deeper architectures, transfer learning, and integration with real-world waste management systems.
## Model Architecture

The CNN architecture includes :
![image](https://github.com/user-attachments/assets/d81ce8a1-a3e0-4bf7-bedb-e6c550a5a3b4)

- **Convolutional Layers:** Extract features from the images.
- **Pooling Layers:** Reduce the dimensionality of feature maps.
- **Fully Connected Layers**: Perform the final classification.
- **Activation Functions**: ReLU for hidden layers and Softmax for output layers.
## Weekly Progress
This section will be updated with weekly progress and links to corresponding Jupyter Notebooks.

### Week 1 : Dataset Preparation and Initial Setup
- **Activities:**
  - Imported required libraries and set up the project environment.
  - Explored and cleaned the dataset.
  - Visualized sample data.

### Week 2 : TBD
- **Activities:**
  - Train the Model: Feed the dataset into the model, adjust hyperparameters, and optimize using techniques like backpropagation.
  - Evaluate Performance: Assess the model using metrics such as accuracy, precision, recall, and F1-score on a validation/test dataset.
  - Fine-tune & Optimize: Adjust hyperparameters, use regularization, or apply techniques like transfer learning to improve performance.
  - Make Predictions: Use the trained model to generate predictions on unseen data and analyze the results.

### Week 3 : TBD
*Details will be added here*


## How to Run

1. Clone the repository:
*     git clone https://github.com/Danishh07/CNN-PlasticWasteClassifier [Your Repository Link]  
*     cd CNN-PlasticClassifier [Your Repository Name]

2. Install the required dependencies:
*     pip install -r requirements.txt

3. Train the model:
  *     python train.py  

4. Perform inference:
 *     python predict.py --image_path /path/to/image.jpg  
## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
