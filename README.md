# GalaxyClassification-PHYS449
 Final project for PHYS 449 - ML models for galaxy classification based on the paper by Moonzarin Reza

### Members:
- Matthew Charbonneau - 20822976
- Jakob Devey - 20854189
- Tom Harding - 20825392
- Aaron Herschberger - 20778142

# Run instructions
To train Extra Trees on existing post-PCA files with default hyperparameters:
### main.py --ET=True

To train the ANN on existing post-PCA files with default hyperparameters:
### main.py --ANN=True

To create feature vectors from images:
### featureVector < [input_folder] > PCA.py -o inputs.txt

# Dependancies
### This project utilizes the following libraries:
- numpy
- SDSS
- csv
- sklearn
- pandas
- io

# Citations
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. (https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
