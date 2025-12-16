# AI for Gas Distribution Mapping

A evaluation of modern AI methods (diffusion models and Fourier neural operators) against classical approaches for reconstructing continuous gas concentration fields from sparse sensor measurements.

## Overview

This repository accompanies our paper **"When does Modern AI Add Value to Gas Distribution Mapping?"** which evaluates whether recent AI advances provide genuine practical value over classical methods for gas distribution mapping (GDM).

We compare:
- **FNO** (Fourier Neural Operator) - operator learning approach
- **Diffusion Models** - generative AI for uncertainty quantification
- **KDM+V** (Kernel DM+V) - classical statistical smoothing
- **DARES** - physics-informed method

## Requirements
```
torch
scipy
scikit-learn
neuralop
```
