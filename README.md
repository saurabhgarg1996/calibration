# Calibration Library

This repository contains library code to measure the calibration error of models from paper [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). We implement temperature scaling and vector scaling motivated from code [here](https://github.com/gpleiss/temperature_scaling). Importantly, we do not need TempScaling and VectorScaling modules to be wrapper around pytorch model. Instead, we implement calibration techniques simply on numpy probability and label arrays.  

## Installation


```pip install git+https://github.com/saurabhgarg1996/calibration.git``` 

or 

```python
git clone https://github.com/saurabhgarg1996/calibration 
cd calibration
pip install .
```

The calibration library requires python 3.6 or higher. 

## Overview

Measuring the calibration error of a model is as simple as:


```python
import calibration as cal
calibration_error = cal.ece_loss(model_logits, labels)
```

Recalibrating a model is very simple as well. Recalibration requires a small labeled dataset, on which we train a recalibrator:

```python
calibrator = cal.TempScaling(bias=False) # cal.VectorScaling(num_label=<num_classes>, bias=True)
calibrator.fit(model_logits, labels)
```

Now whenever the model outputs a prediction, we pass it through the calibrator to produce better probabilities.

```python
calibrated_logits = calibrator.calibrate(test_logits)
calibrated_probs = calibrator.add_softmax(test_logits)
```

## Calibartion Approaches

This library implements the following calibration approaches:

(i) Temperature Scaling as `cal.TempScaling(bias=False)`

(ii) Bias corrected Temperature Scaling as `cal.TempScaling(bias=True, num_label=<num_classes>)`

(iii) Vector Scaling as `cal.VectorScaling(num_label=<num_classes>, bias=True)`

(iv) Matrix Scaling as `cal.MatrixScaling(num_label=<num_classes>, bias=True)`

## Questions, bugs, and contributions

Please feel free to ask us questions, submit bug reports, or contribute push requests.

