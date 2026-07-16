Minimal Python package built from a fork/clone of XGBoost with diffusion estimators:

- XGBDDPMRegressor
- XGBDDPMClassifier
- XGBDiffusionRegressor

The above estimators allow jointly noising the input & output according to a diffusion noise schedule. Here we allow marginalizing over different noise seeds by duplicating the training set and/or refreshing the noise seed (and thus histogram bin assignments) over boosting rounds. 

We're not aware of any other use cases for jointly noising the input and output of XGBoost while refreshing the seed across rounds, but we're open to suggestions and collaborations. Let us know through a GitHub issue.
