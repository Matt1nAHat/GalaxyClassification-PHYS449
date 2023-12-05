# GalaxyClassification-PHYS449
This is a final group project for PHYS 449 - Two machine learning models that aim to classify SDSS objects into one of four categories (Stars, Spirals, Ellipticals, Mergers) based on a paper by Moonzarin Reza.

Use of AI tools, namely copilot, were used to assist in writing code for this project. 

### Members:
- Matthew Charbonneau - 20822976
- Jakob Devey - 20854189
- Tom Harding - 20825392
- Aaron Herschberger - 20778142

## Run instructions
To train Extra Trees on existing post-PCA files with default hyperparameters:
```sh
python main.py --ET True
```

To train the ANN on existing post-PCA files with default hyperparameters:
```sh
python main.py --ANN True
```

To preprocess a list of objects to use for the models (fetch the features and perform PCA):
```sh
python main.py --PROCESS True --OBJ_LIST "test.csv" 
```
Where OBJ_LIST is the file of objects you want to process
Additional parameters include the feature vector and PCA list file names which default to "featuresList.txt" and "PCAList.txt" respectively.


## Dependancies
### This project utilizes the following libraries:
- torch
- numpy
- SDSS
- csv
- sklearn
- pandas
- io
- matplotlib

# Citations
Paper: Moonzarin Reza, Galaxy morphology classification using automated machine learning, Astronomy and Computing, Volume 37, 2021, 100492, ISSN 2213-1337, https://doi.org/10.1016/j.ascom.2021.100492.

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. (https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
