# Calibration Library

This repository contains library code to measure the calibration error of models. We implement temperature scaling and vector scaling motivated from code [here](https://github.com/gpleiss/temperature_scaling). 

## Installation

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
calibration_error = cal.ece_loss(model_probs, labels)
```

Recalibrating a model is very simple as well. Recalibration requires a small labeled dataset, on which we train a recalibrator:

```python
calibrator = cal.TempScaling(bias=False)
calibrator.fit(model_probs, labels)
```

Now whenever the model outputs a prediction, we pass it through the calibrator to produce better probabilities.

```python
calibrated_probs = cal.calibrate(test_probs)
```


## Questions, bugs, and contributions

Please feel free to ask us questions, submit bug reports, or contribute push requests.

