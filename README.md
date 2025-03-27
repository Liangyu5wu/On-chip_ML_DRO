# On-Chip Machine Learning for Dual-Readout Calorimeters

Welcome to the repository for my work on on-chip machine learning for dual-readout calorimeters.

## Contents

This repository contains the code developed for various subtasks associated with the project.

## Model Training

To construct and train the ML model, you can use either of the following scripts:

- **Testmodel.py**: For standard model training
- **Testmodel_QAT.py**: For quantization-aware training (QAT), which optimizes the model for quantization without significant accuracy loss

## Resource Analysis

- **hls4ml_ana.py**: Analyzes resource utilization of the trained model, particularly useful for FPGA deployment

## Data Processing

### Source Code
- **h5_builder/DSB.C**: Original analysis code from [Calvision_DESY_SDL](https://github.com/ledovsk/Calvision_DESY_SDL/tree/main)
- **Synthetic_wf/generator.C**: My iterated version derived from DSB.C, used to synthesize waveforms with specific Scintillation & Cherenkov components

### Data Sources
- Cherenkov SPR data: [SPR_SDL.root](https://github.com/ledovsk/Calvision_DESY_SDL/blob/main/SPR_SDL.root)
- CalVision Electron beam test data: [outfile_LG.root](https://cernbox.cern.ch/remote.php/dav/public-files/MvmLcYjgsUb2CeA/run_200/outfile_LG.root)
