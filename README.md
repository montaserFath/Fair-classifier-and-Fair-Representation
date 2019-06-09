# Fair Classifier and Fair Representation
The main objective is:
- Build a fair classifier which no-one will be able to predict the sensitive attribute from the model. 
- Build a Fair representation of the data.
## [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
-  A set of reasonably clean records was extracted.
-  Prediction task is to determine whether a person makes over 50K a year. 
-  Sensitive Attribute: Gender male or female. 
-  Features: 113 Features, like age, Gender, race, education level, work class, hours per week, country , etc.
-  Number of training samples: 32,561 samples and the Number of testing samples: 16,281 samples.

## Data Visualization:
- The data-set is Unbalanced, umber of males is more larger than the number of females.

![gender_hist](https://github.com/montaserFath/Fair-classifier-and-Fair-Representation/blob/master/Results/gender_hist.png)

- The data-set is Unbalanced, number of people who has less than 50K per year is more larger than the number people who get more thank 50K per Year.

![income_hist](https://github.com/montaserFath/Fair-classifier-and-Fair-Representation/blob/master/Results/income_hist.png)

## Accuracy Functions:

- Accuracy (A):


**<div style="text-align:center"><img src="https://latex.codecogs.com/gif.latex?A=\frac{1}{n}\sum_{i=1}^{n}1[\hat{Y}=Y]"/>**

- Reweighted accuracy (R) of a classifier as the mean accuracy normalized by the size of the two group:


**<div style="text-align:center"><img src="https://latex.codecogs.com/gif.latex?R=\frac{1}{2}\bigg[\frac{1}{n_{A}=0}\sum_{i=1}^{n}1[\hat{Y}=Y,A=0]+\frac{1}{n_{A}=1}\sum_{i=1}^{n}1[\hat{Y}=Y,A=1]\bigg]"/>**

- Fairness metric DP , which measures the demographic parity (DP):


**<div style="text-align:center"><img src="https://latex.codecogs.com/gif.latex?DP=\bigg|\frac{1}{n_{A}=0}\sum_{i=1}^{n}\hat{Y}(1-A)-\frac{1}{n_{A}=1}\sum_{i=1}^{n}\hat{Y}(A)\bigg|"/>**

# Part One:Classification

##  Most correlated Features to Income (Y) and Gender (A):
- I used to methods to calculate crrelation between features and Income (Y) and crrelation between features and Income (Y):
- 1- pearson: the Pearson correlation coefficient measures the linear relationship between two datasets (x,y). Pearson’s correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.

Top correlated Features to Income (Y) using Pearson  |  Top correlated Features to Gender (A) using Pearson
:-------------------------:|:-------------------------:
![](https://github.com/montaserFath/Fair-classifier-and-Fair-Representation/blob/master/Results/top_features_y.png)  |  ![](https://github.com/montaserFath/Fair-classifier-and-Fair-Representation/blob/master/Results/top_features_a.png)

- 2- Tress: Use tress to return the feature importance (the higher, the more important the feature).

Top correlated Features to Income (Y) using Tress      |  Top correlated Features to Gender (A) using Tress 
:-------------------------:|:-------------------------:
![](https://github.com/montaserFath/Fair-classifier-and-Fair-Representation/blob/master/Results/top_features_y_tress.png)  |  ![](https://github.com/montaserFath/Fair-classifier-and-Fair-Representation/blob/master/Results/top_features_a_tress.png)


## Binary Classifier:

I used the following methods to train a Binary Classifier to predict Income (Y) and Gender (A):

- **Binary Logistic Regression.**
- **Random Forest.**
- **Linear Support Vector machine (SVM).**
- **Kernal SVC.**

#### Results:

- Compare different methods to predict **Income (Y)**

**Method**  | Accuracy (A) | Reweighted accuracy (R)  |  DP Accuracy
------------- | ------------- | ------------- | -------------
**Binary Logistic Regression**  | 79.5%  | 81.92% |  **0.04**
**Random Forest**  | 85.79% | 87.6% |  0.16 
**Linear SVM**  | 85.65%  | 87.48% |  0.17 
**SVC**  | **86.59%** |**88.29%**|  0.16 

- Compare different methods to predict **Gender (A)** 

**Method**  | Accuracy (A) | Reweighted accuracy (R)  |  DP Accuracy
------------- | ------------- | ------------- | -------------
**Binary Logistic Regression**  | **66.7%**  | **50%** |  **0.00**
**Random Forest**  | 84.47% | 81.91% |  0.64 
**Linear SVM**  | 84.18% | 83.46% |  0.67 
**SVC**  |  82.19% | 81.71% |  0.63 


# Bart Two: Representation Learning

## pre-processing 

- In this section, we study the effect of pre-processing in representation, so Data-set split into two part: Male data (A=1) and Females data (A=0), After that, we calculate the mean and stander deviation for each part and normalize each part. 

- Train the model after normalized each group (males and females) topredict  Gender  (A)

**Method**  | Accuracy (A) | Reweighted accuracy (R)  |  DP Accuracy
------------- | ------------- | ------------- | -------------
**Binary Logistic Regression**  | 63.41%  | **48.57%** |  0.03
**Random Forest**  | 66.7% | 50% |  **0.00**
**Linear SVM**  | **37.74%** | 51.61% |  0.03
**SVC**  |  66.7% | 50% | **0.00**

## Nural Network

- I used Neural Network as a classifier, So, I used MMD loss with Binary cross-entropy as Loss function as following:

Total Loss = Binary cross entropy loss + alpha * MMD Loss

- Where alpha is a hyperparameter decided how much MMD loss waited comparing to Binary cross entropy loss

**Alpha**  | Accuracy (A) | Reweighted accuracy (R)  |  DP Accuracy
------------- | ------------- | ------------- | -------------
**alpha=0.0**  | **85.6%**  | 86.84% |  0.19
**alpha=0.1**  | 85.14%  | 86.97% |  0.18
**alpha=0.2**  | 85.39%  | 87.15% |  0.18
**alpha=0.5**  | **85.6%**  | **87.38%** |  0.16
**alpha=0.7**  | 85.36%  | 87.14% |  0.18
**alpha=1.0**  | 85.25%  | 87.03% |  0.19
**alpha=10.0**  | 83.66%  | 84.66% |  0.06
**alpha=100.0**  | 79.06%  | 81.7% |  **0.04**

# Installing
Install torchvision
```
pip install torchvision
```
Install pandas libary
```
pip install pandas
```
Install sklearn libary
```
pip install sklearn
```

Install scipy libary
```
pip install scipy
```
