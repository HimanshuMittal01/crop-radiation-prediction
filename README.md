# Predict net radiation for crops

## Install dependancies
Will add all steps soon

```bash
# For regression spline
pip install --editable git+https://github.com/scikit-learn-contrib/py-earth@v0.2dev#egg=pyearth
```

## How to run

```bash
# Train
$ python train.py --config "config/config.json" 
```

## Pending updates
- [x] Create ML framework for ultra rapid experimentation
- [x] Add DNN model
- [ ] Add more flexibiltiy in hyperparameters of deep learning models
- [ ] Replace json with yaml
- [ ] Create a parser for custom optimize fn
- [ ] Add speed for xgboost training