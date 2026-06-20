# Dream drive spot the ball AI
An offline-first Computer Vision pipeline that predicts "Spot the Ball" coordinates by modeling human-judge decision heuristics using a Hybrid CNN-Pose Estimation network in PyTorch.

## The Core System Design Pivot
In traditional aports analytics, machine learning models are trained to calculate phycial ball tragectories using the laws of physics. However in "Spot the Ball" style competitions (such as Dream Drive), the winning  coordinates are not determined by where the physical ball was.

Instead, a panel of professional sports experts analyses the image after the ball has been digitally erased, and votes where they believe the center of the ball should be. This changes the approach/ engineering strategy needed to predict the coordinates at the center of the ball.

### The Engineering Strategy
This architecture does not attempt to model physics. Instead it treats the problem as a Continuous Coordinate Regression task, training an artificial neural network to learn , model, and mimic the cognitive biases and consensus behavious of human judges.

## Ther Pipeline Architecture
The system is designed with two evolutionary approaches to solve coordinates convergence.

### Approach 1: Geometric Heuristic Tracking (Rule-Driven)
It utilises powe estimation models (YOLOv8-Pose/ MediaPipe) to extract keypoint nodes from the players (eyes, shoulders, kicking legs). It calculate s the spital vectors to find the mthematical point of closest convergence. 

### Approach 2: End-to-End Coordinate Regression (Data Driven)
Passes the normalised images into a deep learning Convolutional Neural Network (CNN) backbone *ResNet). The standard classification layer is replaced with a custom regression head that outputs continuous values bounded between 0.0 and 1.0 via a sigmoid activation function.

## Data Challenges
>- Data Contrains: Because the historical winning profiles are scarce, training is initialised with a highly constrained  baseline.
>- Transfer Learning: To prevent overfitting, the model employs a pre-trained  ResNet backbone initialised with ImageNet weights, allowing the network to leverage robust edge and text detection premitive shapes immediately.
>- Regularisation: High-ratio Dropout layers (0, 3) an dmEan Squared Error (MSE)/ Huber Loss are used to maintain generalisation during training phases.

## The Results
The first evolution approach results are mixed and inconsistent, It seems the model is struggling to pick up and generalise patterns in the previous competition winning corrdinates.

The second evolution approach will introduces:
1. Pose estimator (YOLOv8 or MediaPipe)
2. Gaze or body vector extraction - calculating vectors from a players eyes to the ball, the shoulder to estimate where the ball might be.
3. Intersection calculator - calculate the closest point of convergence of these vectors