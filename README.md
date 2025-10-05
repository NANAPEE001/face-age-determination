This project implements a deep learning model that predicts a person’s age from facial images using TensorFlow and MobileNetV2. 
The model leverages transfer learning and applies data augmentation techniques such as random flipping, rotation, and zoom to improve accuracy and generalization. 
Images are preprocessed by resizing to 224x224 pixels and normalized to match MobileNetV2’s input format. The dataset is organized with face images grouped in folders named by age labels.
Training is performed in two stages: first with a frozen base model, then fine-tuning all layers for improved performance. 
Early stopping and model checkpointing are used to avoid overfitting and save the best weights. 
Evaluation is done on a validation set using Mean Absolute Error (MAE) and Mean Squared Error (MSE). 
The project includes scripts to train the model and make age predictions on new images. 
This repository demonstrates a complete pipeline for image-based regression tasks in deep learning
