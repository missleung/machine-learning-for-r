# machine-learning-for-r
Using caret package, I will be conducting a few data analyses from Kaggle. Currently, there is data set I've analysed on Black Friday sales. Although there are not many predictors, I will use this data set for machine learning to assess the models in predicting the amount of sales per customer. This could potentially lead to customer segmentation.

Here is the link for Kaggle's data set:https://www.kaggle.com/mehdidag/black-friday

Note: .md files are used for report whereas .rmd files are for R coding only

## Exploratory_BlackFriday
This is the exploratory coding on the Black Friday sales. Be warned! There are a lot of plots in there to visualize the data.

## RandomForest_BlackFriday
Using multiple linear regression as the base model, this file contains code on random forest regressions for comparisons. Random forest includes basic tuning such as number of trees and predictors. To further improve on the random forest models, it also uses gradient boosting and XGBoost on the random forest models. Assessments on these models are based on a test data set. 
