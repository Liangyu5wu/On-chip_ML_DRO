# On-Chip Machine Learning for Dual-Readout Calorimeters

Welcome to the repository for my work on on-chip machine learning for dual-readout calorimeters. 

This repository contains the code developed for various subtasks associated with the project. 

To construct and train the ML model, you can use either of the following scripts:

1. **Testmodel.py**: This script is used for standard model training.

2. **Testmodel_QAT.py**: This script is used for quantization-aware training (QAT) of the model, which helps in training the model with quantization in mind to improve efficiency without losing much accuracy.


To assess the resource utilization of the trained model, use the **hls4ml_ana.py** script. This script provides insights into the resource usage of the model, typically useful when deploying the model on hardware such as FPGAs.
