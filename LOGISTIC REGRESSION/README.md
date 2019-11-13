# INSURANCE CLAIM PREDICTION

Now in this project, we are going to predict the Insurance claim using logistic regression. This dataset contains information on the insurance claim. each observation is different policyholder with various features like the age of the person, the gender of the policyholder, body mass index, providing an understanding of the body, number of children of the policyholder, smoking state of the policyholder and individual medical costs billed by health insurance.

![Alt txt](https://rajputhimanshu.files.wordpress.com/2018/03/linear_vs_logistic_regression.jpg)

## About the Dataset:
The dataset has details of 1338 Insurance claim with the following 8 features <br/>
age , sex, bmi, children, smoker, region, charges, insuranceclaim <br/>

## Things done in this project:
-- load the dataset and see how it looks like. Additionally, split it into train and test set.<br/>
-- plot the box plot to check for the outlier <br/>
-- check the pair_plot for feature vs feature. This tells us which features are highly correlated with the other feature and help us predict it better logistic regression model.<br/>
-- check the count_plot for different features vs target variable insurance claim. This tells us which features are highly correlated with the target variable insuranceclaim and help us predict it better.<br/>
-- using logistic regression to predict the insuranceclaim. We will select best model by cross-validation using Grid Search.<br/>
-- visualize the performance of a binary classifier. Check the performance of the classifier using roc auc curve <br/>

## Things to learn fro this project . 
-- After completing this project, we will have the better understanding of how to build a logistic regression model. In this project, we will apply the following concepts <br/>
* Train-test split
* Correlation between the features
* Logistic Regression
* Auc score
* Roc auc plot
