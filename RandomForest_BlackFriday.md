Random Forest
================
Lisa Leung
2019-01-27

Load all libraries
------------------

``` r
library(tidyverse)
```

    ## ── Attaching packages ────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✔ ggplot2 3.1.0     ✔ purrr   0.2.4
    ## ✔ tibble  1.4.2     ✔ dplyr   0.7.4
    ## ✔ tidyr   0.8.0     ✔ stringr 1.3.0
    ## ✔ readr   1.1.1     ✔ forcats 0.3.0

    ## ── Conflicts ───────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Warning in as.POSIXlt.POSIXct(Sys.time()): unknown timezone 'zone/tz/2018i.
    ## 1.0/zoneinfo/America/Vancouver'

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(xgboost)
```

    ## 
    ## Attaching package: 'xgboost'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

Purpose
-------

We will conduct random forest to predict the amount of purchase made.

Because amount of purchase made is very well known to be highly correlated with the product category, I will omit the product category and try to instead find more information about who are the consumers who have bought a lot more than the others according to their demographics. Hence, we will not see product category as a predictor in my models.

``` r
# Loading data
dat_User <- read_csv("BlackFriday-User.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   User_ID = col_integer(),
    ##   Gender = col_character(),
    ##   Age = col_character(),
    ##   Occupation = col_integer(),
    ##   City_Category = col_character(),
    ##   Stay_In_Current_City_Years = col_character(),
    ##   Marital_Status = col_integer(),
    ##   sum_Cat_1 = col_integer(),
    ##   sum_Cat_2 = col_integer(),
    ##   sum_Cat_3 = col_integer(),
    ##   sum_Purchase = col_integer()
    ## )

``` r
dat_User <- dat_User[,!colnames(dat_User) %in% c("X1", "User_ID")]
dat_User$Occupation <- as.factor(dat_User$Occupation) #converting to a factor


# Train and Test data
set.seed(10)
num <- round(nrow(dat_User)/2)
vec_Train <- sample(1:nrow(dat_User),size = num)

dat_Train <- dat_User[vec_Train,]
dat_Test <- dat_User[-vec_Train,]
```

Multiple Regression
-------------------

Before starting random forest, I want to use a multiple regression as a base model on the data set.

``` r
lm_multiple <- lm(sum_Purchase~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years + Marital_Status, data=dat_Train)
summary(lm_multiple)
```

    ## 
    ## Call:
    ## lm(formula = sum_Purchase ~ Gender + Age + Occupation + City_Category + 
    ##     Stay_In_Current_City_Years + Marital_Status, data = dat_Train)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -1434983  -509725  -170159   341117  6999167 
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   1072383     146499   7.320 3.19e-13 ***
    ## GenderM                        249011      36553   6.812 1.16e-11 ***
    ## Age18-25                       -69763     133160  -0.524  0.60039    
    ## Age26-35                        72333     134158   0.539  0.58982    
    ## Age36-45                        59456     136647   0.435  0.66352    
    ## Age46-50                        -6973     144062  -0.048  0.96140    
    ## Age51-55                       -81842     144849  -0.565  0.57211    
    ## Age55+                        -213761     149495  -1.430  0.15286    
    ## Occupation1                    -66394      71539  -0.928  0.35344    
    ## Occupation2                   -104740      88510  -1.183  0.23676    
    ## Occupation3                    172870     100072   1.727  0.08419 .  
    ## Occupation4                     48041      68824   0.698  0.48522    
    ## Occupation5                    107415     125510   0.856  0.39217    
    ## Occupation6                     72273      96654   0.748  0.45467    
    ## Occupation7                    -38887      66260  -0.587  0.55732    
    ## Occupation8                    -26798     252553  -0.106  0.91550    
    ## Occupation9                    -39497     140819  -0.280  0.77913    
    ## Occupation10                   -95587     142818  -0.669  0.50336    
    ## Occupation11                  -175312     110762  -1.583  0.11358    
    ## Occupation12                  -217535      79287  -2.744  0.00611 ** 
    ## Occupation13                  -118280     122195  -0.968  0.33315    
    ## Occupation14                  -101667      83896  -1.212  0.22568    
    ## Occupation15                  -111269     108275  -1.028  0.30420    
    ## Occupation16                   218549      96696   2.260  0.02388 *  
    ## Occupation17                  -102678      71320  -1.440  0.15007    
    ## Occupation18                   108994     152509   0.715  0.47487    
    ## Occupation19                   203884     146139   1.395  0.16308    
    ## Occupation20                   200335      88793   2.256  0.02413 *  
    ## City_CategoryB                 -76875      48104  -1.598  0.11013    
    ## City_CategoryC                -716852      44024 -16.283  < 2e-16 ***
    ## Stay_In_Current_City_Years1      5134      52224   0.098  0.92169    
    ## Stay_In_Current_City_Years2    -25938      57362  -0.452  0.65117    
    ## Stay_In_Current_City_Years3     86286      60124   1.435  0.15136    
    ## Stay_In_Current_City_Years4+    44459      60554   0.734  0.46288    
    ## Marital_Status                  19718      34468   0.572  0.56732    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 856900 on 2911 degrees of freedom
    ## Multiple R-squared:  0.1707, Adjusted R-squared:  0.161 
    ## F-statistic: 17.62 on 34 and 2911 DF,  p-value: < 2.2e-16

According to a summary of the multiple linear regression, we see that gender, cities, and occupations play a huge role in total purchases made. I'd like to also see all three cities and check if there are any other differences among the cities.

A couple of measures we will use to compare multiple linear regression to random forests:

Multiple R-squared is squared of correlation between fitted and actual values. Residual standard error is root(mean squared error).

Ultimately, we will also measure the error of predicted rate.

We're going to fit the test data into our multiple linear regression and see how well it predicts.
--------------------------------------------------------------------------------------------------

``` r
# Going to manually calculate the RMSE with the multiple linear regression
vals_predicted <- predict.lm(lm_multiple, newdata = dat_Test)
vals_errors <- dat_Test$sum_Purchase-vals_predicted
RMSE_lm <- sqrt(sum(vals_errors^2)/length(vals_errors))
print(RMSE_lm)
```

    ## [1] 854686.1

``` r
# R squared on predicted values
Rsq_lm <- cor(vals_predicted, dat_Test$sum_Purchase)^2
print(Rsq_lm)
```

    ## [1] 0.1568235

Multiple linear regression seem to do a pretty decent job in terms of predicting values. Later, we will see if we can beat this measure through random forest regression.

### Checking out regressions separated by cities

``` r
dat_A <- dat_Train[dat_Train$City_Category=="A",]
lm_multiple_A <- lm(sum_Purchase~Gender+Age+Occupation+Stay_In_Current_City_Years + Marital_Status, data=dat_A)
summary(lm_multiple_A)
```

    ## 
    ## Call:
    ## lm(formula = sum_Purchase ~ Gender + Age + Occupation + Stay_In_Current_City_Years + 
    ##     Marital_Status, data = dat_A)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -2107804  -890230  -361947   428204  6219546 
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error t value Pr(>|t|)   
    ## (Intercept)                   1374257     665740   2.064  0.03952 * 
    ## GenderM                        365654     140774   2.597  0.00968 **
    ## Age18-25                      -458341     623950  -0.735  0.46295   
    ## Age26-35                      -191673     628891  -0.305  0.76066   
    ## Age36-45                       -95458     640095  -0.149  0.88151   
    ## Age46-50                      -535371     685094  -0.781  0.43491   
    ## Age51-55                      -860872     673265  -1.279  0.20163   
    ## Age55+                       -1111518     706982  -1.572  0.11655   
    ## Occupation1                    -20534     257794  -0.080  0.93655   
    ## Occupation2                   -398309     292008  -1.364  0.17319   
    ## Occupation3                    414583     308820   1.342  0.18007   
    ## Occupation4                    280486     229850   1.220  0.22294   
    ## Occupation5                    599012     479806   1.248  0.21247   
    ## Occupation6                    440427     441600   0.997  0.31909   
    ## Occupation7                    -91370     251564  -0.363  0.71661   
    ## Occupation8                   -372916     984278  -0.379  0.70495   
    ## Occupation9                    241312     633276   0.381  0.70333   
    ## Occupation10                  -307798     634843  -0.485  0.62801   
    ## Occupation11                   -14621     467873  -0.031  0.97508   
    ## Occupation12                  -429695     274279  -1.567  0.11785   
    ## Occupation13                  -369026     719757  -0.513  0.60839   
    ## Occupation14                   -91868     307772  -0.298  0.76545   
    ## Occupation15                    18411     385624   0.048  0.96194   
    ## Occupation16                   111964     448208   0.250  0.80284   
    ## Occupation17                   -15761     294490  -0.054  0.95734   
    ## Occupation18                   866660     701969   1.235  0.21757   
    ## Occupation19                     6112     637565   0.010  0.99235   
    ## Occupation20                   739824     317733   2.328  0.02030 * 
    ## Stay_In_Current_City_Years1   -279050     197984  -1.409  0.15934   
    ## Stay_In_Current_City_Years2   -125472     215003  -0.584  0.55977   
    ## Stay_In_Current_City_Years3     52928     226824   0.233  0.81559   
    ## Stay_In_Current_City_Years4+  -171951     226044  -0.761  0.44721   
    ## Marital_Status                 138698     134209   1.033  0.30191   
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1346000 on 487 degrees of freedom
    ## Multiple R-squared:  0.1039, Adjusted R-squared:  0.04505 
    ## F-statistic: 1.765 on 32 and 487 DF,  p-value: 0.006837

``` r
dat_B <- dat_Train[dat_Train$City_Category=="B",]
lm_multiple_B <- lm(sum_Purchase~Gender+Age+Occupation+Stay_In_Current_City_Years + Marital_Status, data=dat_B)
summary(lm_multiple_B)
```

    ## 
    ## Call:
    ## lm(formula = sum_Purchase ~ Gender + Age + Occupation + Stay_In_Current_City_Years + 
    ##     Marital_Status, data = dat_B)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -1748570  -769346  -266549   596366  3491871 
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   1019963     316338   3.224  0.00131 ** 
    ## GenderM                        398022      81424   4.888 1.23e-06 ***
    ## Age18-25                      -207880     295816  -0.703  0.48242    
    ## Age26-35                       -71512     301131  -0.237  0.81235    
    ## Age36-45                      -171597     309016  -0.555  0.57884    
    ## Age46-50                      -122898     328701  -0.374  0.70858    
    ## Age51-55                      -155208     330137  -0.470  0.63839    
    ## Age55+                        -375100     364983  -1.028  0.30439    
    ## Occupation1                   -143570     162306  -0.885  0.37665    
    ## Occupation2                   -111444     196634  -0.567  0.57103    
    ## Occupation3                    274698     250684   1.096  0.27349    
    ## Occupation4                   -140264     154233  -0.909  0.36339    
    ## Occupation5                     71931     244876   0.294  0.76903    
    ## Occupation6                    202969     200125   1.014  0.31078    
    ## Occupation7                     59986     160070   0.375  0.70795    
    ## Occupation8                   1490329    1049467   1.420  0.15597    
    ## Occupation9                   -124489     348624  -0.357  0.72112    
    ## Occupation10                  -465536     306723  -1.518  0.12946    
    ## Occupation11                  -303262     249856  -1.214  0.22520    
    ## Occupation12                  -355675     179108  -1.986  0.04739 *  
    ## Occupation13                  -690879     302117  -2.287  0.02246 *  
    ## Occupation14                  -196311     195122  -1.006  0.31467    
    ## Occupation15                  -240730     260081  -0.926  0.35493    
    ## Occupation16                   466246     202131   2.307  0.02132 *  
    ## Occupation17                  -316390     161910  -1.954  0.05103 .  
    ## Occupation18                   155743     409838   0.380  0.70404    
    ## Occupation19                   264608     298636   0.886  0.37585    
    ## Occupation20                    77763     186850   0.416  0.67739    
    ## Stay_In_Current_City_Years1    157780     118074   1.336  0.18183    
    ## Stay_In_Current_City_Years2     31672     130772   0.242  0.80869    
    ## Stay_In_Current_City_Years3    239728     134665   1.780  0.07542 .  
    ## Stay_In_Current_City_Years4+   151669     139516   1.087  0.27731    
    ## Marital_Status                 -35284      78918  -0.447  0.65492    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1032000 on 812 degrees of freedom
    ## Multiple R-squared:  0.08194,    Adjusted R-squared:  0.04576 
    ## F-statistic: 2.265 on 32 and 812 DF,  p-value: 9.656e-05

``` r
dat_C <- dat_Train[dat_Train$City_Category=="C",]
lm_multiple_C <- lm(sum_Purchase~Gender+Age+Occupation+Stay_In_Current_City_Years + Marital_Status, data=dat_C)
summary(lm_multiple_C)
```

    ## 
    ## Call:
    ## lm(formula = sum_Purchase ~ Gender + Age + Occupation + Stay_In_Current_City_Years + 
    ##     Marital_Status, data = dat_C)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -675342 -320358 -135171  222434 1839772 
    ## 
    ## Coefficients:
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                    306989      94137   3.261  0.00113 ** 
    ## GenderM                        123955      25475   4.866 1.26e-06 ***
    ## Age18-25                       101138      88654   1.141  0.25412    
    ## Age26-35                       139913      88595   1.579  0.11449    
    ## Age36-45                       133140      89710   1.484  0.13798    
    ## Age46-50                       137079      94245   1.454  0.14601    
    ## Age51-55                       104415      95028   1.099  0.27203    
    ## Age55+                          19824      96142   0.206  0.83667    
    ## Occupation1                    -15668      50123  -0.313  0.75463    
    ## Occupation2                     45248      65396   0.692  0.48910    
    ## Occupation3                     30748      71219   0.432  0.66599    
    ## Occupation4                     19701      50208   0.392  0.69483    
    ## Occupation5                    -78600      95968  -0.819  0.41290    
    ## Occupation6                   -104281      67632  -1.542  0.12330    
    ## Occupation7                    -25864      44684  -0.579  0.56279    
    ## Occupation8                     -2792     148223  -0.019  0.98497    
    ## Occupation9                    -77648      89727  -0.865  0.38696    
    ## Occupation10                    94669      96822   0.978  0.32835    
    ## Occupation11                  -101264      74466  -1.360  0.17407    
    ## Occupation12                   -17154      56443  -0.304  0.76122    
    ## Occupation13                    14177      75842   0.187  0.85175    
    ## Occupation14                   -49006      57585  -0.851  0.39488    
    ## Occupation15                   -66388      73655  -0.901  0.36755    
    ## Occupation16                    80801      66803   1.210  0.22663    
    ## Occupation17                     4846      48321   0.100  0.92013    
    ## Occupation18                    17507      94189   0.186  0.85257    
    ## Occupation19                   165436     102980   1.606  0.10837    
    ## Occupation20                     1393      65963   0.021  0.98316    
    ## Stay_In_Current_City_Years1     29833      36318   0.821  0.41152    
    ## Stay_In_Current_City_Years2     -8691      39828  -0.218  0.82728    
    ## Stay_In_Current_City_Years3     32357      41988   0.771  0.44105    
    ## Stay_In_Current_City_Years4+    60611      41957   1.445  0.14877    
    ## Marital_Status                  10770      23818   0.452  0.65119    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 431900 on 1548 degrees of freedom
    ## Multiple R-squared:  0.0356, Adjusted R-squared:  0.01566 
    ## F-statistic: 1.786 on 32 and 1548 DF,  p-value: 0.004612

Interestingly, the spendings seem to affect most on city A and city B. We see that occupation 20 seem to spend $739,824 more on average.

Let's start the random forest!
==============================

We will use train to tune parameters. The first train will be optimizing the randomly selected predictors
---------------------------------------------------------------------------------------------------------

``` r
# Setting parameters on mtry tuning
control <- trainControl( #trainControl is used to alter the default methods in train function
  method="repeatedcv", # K-fold CV; by default, it uses bootstrap sampling
  number=5, # 3 repeats of 5-fold CV
  repeats=3) 

metric <- "RMSE" #A string that specifies what summary metric will be used to select the optimal model. By default, possible values are "RMSE" and "Rsquared" for regression and "Accuracy" and "Kappa" for classification. 
mtry <- 1:6 # number of variables to use per tree; note that it is usually the best on square root of number of variables. Will try a sequence 
tunegrid <- expand.grid(.mtry=mtry) # Change this parameter to change the candidates for tuning parameters
preProc = c("center", "scale")

# Running the random forest

seed <- 10
set.seed(seed)
rf_simple <- train(
  sum_Purchase~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years + Marital_Status, data=dat_Train, # model
  method = 'rf', #using random forest to train. FYI, I've accidentally used qrf which is supposed to look at high dimensional data. It was usedto estimate conditional quantiles - we are not using that. Big oops!
  metric=metric, #using RMSE (root mean square error) to define my loss function
  tuneGrid=tunegrid, # Tuning parameters uses mytry (randomly selected predictors);
  trControl=control, # method="repeatedcv",number=10, repeats=3
  preProc=preProc) #centering and scaling the predictors
print(rf_simple)
```

    ## Random Forest 
    ## 
    ## 2946 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (34), scaled (34) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 2356, 2355, 2358, 2358, 2357, 2356, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared   MAE     
    ##   1     901916.7  0.1406302  632096.6
    ##   2     874779.7  0.1455999  612180.2
    ##   3     867031.0  0.1437728  605229.1
    ##   4     867846.6  0.1386579  604603.8
    ##   5     872287.0  0.1320574  605985.3
    ##   6     878476.6  0.1248917  608188.8
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 3.

``` r
ggplot(rf_simple)
```

![](RandomForest_BlackFriday_files/figure-markdown_github/unnamed-chunk-8-1.png)

Gradient boosting random forest
-------------------------------

Next I'd like to try is gradient boosting random forest. Boosting algorithms are built so that in each iteration/model that is ran, the observational data points that have larger residuals are weighted more heavily, so that the model can focus more on the data points that were estimated poorly on the previous iteration/model. Adaboost would've been another alternative, however, our random forest is not predicting classes but is predicting numerical values.

``` r
# Running the gradient boosting random forest; keeping everything else the same 

seed <- 10
set.seed(seed)
rf_gbm <- train(
  sum_Purchase~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years + Marital_Status, data=dat_Train, # model
  method = 'gbm', #gradient boosting method; on each 
  metric=metric, #using RMSE (root mean square error) to define my loss function
  trControl=control, # method="repeatedcv",number=10, repeats=3
  preProc=preProc,
  verbose=F) #centering and scaling the predictors
print(rf_gbm)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 2946 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (34), scaled (34) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 2356, 2355, 2358, 2358, 2357, 2356, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared   MAE     
    ##   1                   50      861677.6  0.1520094  604677.2
    ##   1                  100      860053.1  0.1540987  604954.3
    ##   1                  150      860511.5  0.1531818  607115.0
    ##   2                   50      858857.3  0.1570527  602980.5
    ##   2                  100      858904.7  0.1570559  603133.9
    ##   2                  150      859226.3  0.1567434  602846.3
    ##   3                   50      857613.6  0.1592002  601482.7
    ##   3                  100      859915.9  0.1557511  602487.5
    ##   3                  150      861291.8  0.1539836  603148.1
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 50, interaction.depth
    ##  = 3, shrinkage = 0.1 and n.minobsinnode = 10.

Using gradient random forest boosting, it seems like the Rsquared values increased at depth = 2. However, Rsquared for gradient boosting is still relatively lower than the multiple linear regression. Hence, we will try to manually add in more tuning parameters in the gradient boosting method.

``` r
ggplot(rf_gbm)
```

![](RandomForest_BlackFriday_files/figure-markdown_github/unnamed-chunk-10-1.png)

``` r
# Testing the predicted values with test data
vals_predicted_gbm <- predict(rf_gbm, newdata = dat_Test)
vals_errors_gbm <- dat_Test$sum_Purchase-vals_predicted_gbm
RMSE_gbm <- sqrt(sum(vals_errors_gbm^2)/length(vals_errors_gbm))
print(RMSE_gbm)
```

    ## [1] 846990.5

``` r
# R squared on predicted values
Rsq_gbm <- cor(vals_predicted_gbm, dat_Test$sum_Purchase)^2
print(Rsq_gbm)
```

    ## [1] 0.1718203

We see an improvement on Rsquared and RMSE on the gradient random forest model of 0.1696045 and 848294.2 rather than 0.1568235 and 854686.1 from multiple regression. However, only default parameters were tuned. Now I'd like to custom tune a wider range of parameters in the tunegrid on gradient boosting random forest.

``` r
# Running the gradient boosting random forest for more custom tuning parameters; keeping everything else the same 

# Manually adding in a grid to tune three parameters:
tunegrid <- expand.grid(n.trees = (1:10)*50, # number of trees, I originally tried up to 300 in number of trees, but it seemed like it's still going down. Now we will try up to 500 
                        interaction.depth = 1:5, # interaction.depth = # of terminal nodes + 1
                        # I originally tried interaction.depth = 1
                        shrinkage = c(0.1,0.01), # learning rate (how fast can the algorithm adapt to)
                        # Learning rate for 0.01 shows stability of decreasing in RMSE than 0.1. 
                        n.minobsinnode = 20# minimum number of samples in the tree
                        ) 


seed <- 10
set.seed(seed)
rf_gbm2 <- train(
  sum_Purchase~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years + Marital_Status, data=dat_Train, # model
  method = 'gbm', #gradient boosting method; on each 
  metric=metric, #using RMSE (root mean square error) to define my loss function
  tuneGrid=tunegrid, # look above
  trControl=control, # method="repeatedcv",number=10, repeats=3
  preProc=preProc, #centering and scaling the predictors
  verbose=F
)
print(rf_gbm2)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 2946 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (34), scaled (34) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 2356, 2355, 2358, 2358, 2357, 2356, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   shrinkage  interaction.depth  n.trees  RMSE      Rsquared   MAE     
    ##   0.01       1                   50      893076.8  0.1368774  628152.7
    ##   0.01       1                  100      877730.9  0.1371934  618208.2
    ##   0.01       1                  150      871532.8  0.1413735  613750.7
    ##   0.01       1                  200      868147.2  0.1447678  610768.9
    ##   0.01       1                  250      865994.9  0.1473437  608834.1
    ##   0.01       1                  300      864535.0  0.1490838  607552.4
    ##   0.01       1                  350      863468.3  0.1504361  606557.0
    ##   0.01       1                  400      862576.4  0.1515696  605826.7
    ##   0.01       1                  450      861915.6  0.1523194  605378.3
    ##   0.01       1                  500      861482.9  0.1527972  605070.6
    ##   0.01       2                   50      890134.1  0.1491497  625879.1
    ##   0.01       2                  100      872618.7  0.1506273  614594.7
    ##   0.01       2                  150      865634.8  0.1522927  609902.6
    ##   0.01       2                  200      862234.0  0.1546919  607088.8
    ##   0.01       2                  250      860347.1  0.1565219  605197.9
    ##   0.01       2                  300      859283.9  0.1575258  604143.9
    ##   0.01       2                  350      858415.0  0.1585500  603517.5
    ##   0.01       2                  400      857978.8  0.1590257  603050.8
    ##   0.01       2                  450      857715.5  0.1592364  602722.0
    ##   0.01       2                  500      857495.5  0.1595182  602653.6
    ##   0.01       3                   50      888560.1  0.1527743  625092.4
    ##   0.01       3                  100      870903.7  0.1539370  613425.4
    ##   0.01       3                  150      863965.8  0.1548478  608611.4
    ##   0.01       3                  200      860860.2  0.1565119  606067.3
    ##   0.01       3                  250      859158.8  0.1578953  604333.5
    ##   0.01       3                  300      858359.4  0.1584434  603595.5
    ##   0.01       3                  350      857698.5  0.1591664  602941.4
    ##   0.01       3                  400      857509.2  0.1592642  602632.4
    ##   0.01       3                  450      857563.9  0.1590681  602594.5
    ##   0.01       3                  500      857607.2  0.1589487  602741.6
    ##   0.01       4                   50      887944.8  0.1542317  624658.1
    ##   0.01       4                  100      870056.4  0.1549390  612920.7
    ##   0.01       4                  150      862837.0  0.1566556  607922.5
    ##   0.01       4                  200      859703.0  0.1581187  605178.1
    ##   0.01       4                  250      858246.3  0.1589673  603887.1
    ##   0.01       4                  300      857672.0  0.1592771  603291.3
    ##   0.01       4                  350      857402.2  0.1594655  602836.0
    ##   0.01       4                  400      857558.6  0.1590654  602957.0
    ##   0.01       4                  450      857656.9  0.1589425  602924.4
    ##   0.01       4                  500      857821.9  0.1587480  602909.2
    ##   0.01       5                   50      887526.6  0.1541868  624335.2
    ##   0.01       5                  100      869371.4  0.1556497  612469.4
    ##   0.01       5                  150      862644.2  0.1560748  607570.0
    ##   0.01       5                  200      859749.8  0.1572779  605182.2
    ##   0.01       5                  250      858637.3  0.1577004  604057.4
    ##   0.01       5                  300      858256.6  0.1578550  603369.8
    ##   0.01       5                  350      858217.0  0.1577809  603008.6
    ##   0.01       5                  400      858469.9  0.1573524  603080.3
    ##   0.01       5                  450      858563.6  0.1573339  602962.4
    ##   0.01       5                  500      858825.4  0.1570584  603101.2
    ##   0.10       1                   50      861222.9  0.1530319  604936.7
    ##   0.10       1                  100      860277.7  0.1536557  605394.7
    ##   0.10       1                  150      859709.2  0.1547271  606205.8
    ##   0.10       1                  200      859887.3  0.1545281  607495.4
    ##   0.10       1                  250      860135.7  0.1542423  607714.6
    ##   0.10       1                  300      860546.7  0.1533667  608669.7
    ##   0.10       1                  350      860210.6  0.1541089  608110.1
    ##   0.10       1                  400      860434.3  0.1536595  607888.0
    ##   0.10       1                  450      860506.6  0.1532806  608003.2
    ##   0.10       1                  500      860658.3  0.1532987  608830.3
    ##   0.10       2                   50      859054.5  0.1561105  603831.9
    ##   0.10       2                  100      859132.1  0.1561032  604816.3
    ##   0.10       2                  150      860294.6  0.1546694  605563.0
    ##   0.10       2                  200      861390.7  0.1529476  606333.5
    ##   0.10       2                  250      861887.2  0.1522441  606459.5
    ##   0.10       2                  300      861702.4  0.1527317  606086.2
    ##   0.10       2                  350      862372.3  0.1519466  606238.9
    ##   0.10       2                  400      863484.5  0.1503305  606856.7
    ##   0.10       2                  450      864074.8  0.1494677  606632.4
    ##   0.10       2                  500      864615.9  0.1485442  606774.5
    ##   0.10       3                   50      858763.2  0.1568801  602625.0
    ##   0.10       3                  100      860344.4  0.1548413  605186.2
    ##   0.10       3                  150      861514.6  0.1535961  605152.6
    ##   0.10       3                  200      863542.2  0.1504533  605644.1
    ##   0.10       3                  250      863696.3  0.1502263  604778.0
    ##   0.10       3                  300      865468.2  0.1478730  606248.2
    ##   0.10       3                  350      866485.7  0.1467831  606423.4
    ##   0.10       3                  400      867609.8  0.1450701  607014.5
    ##   0.10       3                  450      869412.5  0.1427736  608344.5
    ##   0.10       3                  500      869422.7  0.1432673  608093.6
    ##   0.10       4                   50      858370.1  0.1576154  602944.3
    ##   0.10       4                  100      860465.6  0.1553633  604129.1
    ##   0.10       4                  150      863649.7  0.1507616  605172.0
    ##   0.10       4                  200      865922.8  0.1471001  605320.7
    ##   0.10       4                  250      867771.5  0.1449582  605581.5
    ##   0.10       4                  300      868553.6  0.1442822  605681.0
    ##   0.10       4                  350      870491.8  0.1418829  606174.9
    ##   0.10       4                  400      872031.3  0.1398456  607748.4
    ##   0.10       4                  450      873980.8  0.1373952  608578.0
    ##   0.10       4                  500      874908.8  0.1362994  609465.7
    ##   0.10       5                   50      860502.3  0.1544772  603970.5
    ##   0.10       5                  100      864605.4  0.1483884  606436.1
    ##   0.10       5                  150      868000.9  0.1436866  607230.1
    ##   0.10       5                  200      871705.8  0.1390049  609225.3
    ##   0.10       5                  250      873398.5  0.1370111  609238.7
    ##   0.10       5                  300      874860.1  0.1356871  609516.2
    ##   0.10       5                  350      877319.6  0.1323678  610458.3
    ##   0.10       5                  400      878771.8  0.1310136  611563.5
    ##   0.10       5                  450      880294.4  0.1291154  612178.7
    ##   0.10       5                  500      882987.7  0.1265840  614555.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 20
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 350,
    ##  interaction.depth = 4, shrinkage = 0.01 and n.minobsinnode = 20.

``` r
ggplot(rf_gbm2)
```

![](RandomForest_BlackFriday_files/figure-markdown_github/unnamed-chunk-14-1.png)

``` r
# Testing the predicted values with test data
vals_predicted_gbm2 <- predict(rf_gbm2, newdata = dat_Test)
vals_errors_gbm2 <- dat_Test$sum_Purchase-vals_predicted_gbm2
RMSE_gbm2 <- sqrt(sum(vals_errors_gbm2^2)/length(vals_errors_gbm2))
print(RMSE_gbm2)
```

    ## [1] 848508

``` r
# R squared on predicted values
Rsq_gbm2 <- cor(vals_predicted_gbm2, dat_Test$sum_Purchase)^2
print(Rsq_gbm2)
```

    ## [1] 0.1692089

XGBoost Random Forest
---------------------

Next model I'd like to try is the XGBoost (eXtreme Gradient Boosting) random forest.

``` r
seed <- 10
set.seed(seed)
rf_xgbm <- train(
  sum_Purchase~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years + Marital_Status, data=dat_Train, # model
  method = 'xgbTree', #gradient boosting method; on each 
  metric=metric, #using RMSE (root mean square error) to define my loss function
  trControl=control, # method="repeatedcv",number=10, repeats=3
  preProc=preProc, #centering and scaling the predictors
  verbose=F
)
print(rf_xgbm)
```

    ## eXtreme Gradient Boosting 
    ## 
    ## 2946 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (34), scaled (34) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 2356, 2355, 2358, 2358, 2357, 2356, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   eta  max_depth  colsample_bytree  subsample  nrounds  RMSE    
    ##   0.3  1          0.6               0.50        50      923010.3
    ##   0.3  1          0.6               0.50       100      922443.1
    ##   0.3  1          0.6               0.50       150      923526.4
    ##   0.3  1          0.6               0.75        50      922302.1
    ##   0.3  1          0.6               0.75       100      922364.8
    ##   0.3  1          0.6               0.75       150      922263.9
    ##   0.3  1          0.6               1.00        50      922526.4
    ##   0.3  1          0.6               1.00       100      922269.2
    ##   0.3  1          0.6               1.00       150      922351.2
    ##   0.3  1          0.8               0.50        50      920942.7
    ##   0.3  1          0.8               0.50       100      921319.3
    ##   0.3  1          0.8               0.50       150      921136.7
    ##   0.3  1          0.8               0.75        50      920718.6
    ##   0.3  1          0.8               0.75       100      920279.2
    ##   0.3  1          0.8               0.75       150      921091.5
    ##   0.3  1          0.8               1.00        50      920769.5
    ##   0.3  1          0.8               1.00       100      920548.1
    ##   0.3  1          0.8               1.00       150      920706.4
    ##   0.3  2          0.6               0.50        50      924991.7
    ##   0.3  2          0.6               0.50       100      928702.5
    ##   0.3  2          0.6               0.50       150      930411.7
    ##   0.3  2          0.6               0.75        50      924228.2
    ##   0.3  2          0.6               0.75       100      927539.9
    ##   0.3  2          0.6               0.75       150      929406.4
    ##   0.3  2          0.6               1.00        50      925247.4
    ##   0.3  2          0.6               1.00       100      927898.2
    ##   0.3  2          0.6               1.00       150      929852.3
    ##   0.3  2          0.8               0.50        50      925690.4
    ##   0.3  2          0.8               0.50       100      930024.0
    ##   0.3  2          0.8               0.50       150      933059.2
    ##   0.3  2          0.8               0.75        50      925580.4
    ##   0.3  2          0.8               0.75       100      928186.2
    ##   0.3  2          0.8               0.75       150      932712.4
    ##   0.3  2          0.8               1.00        50      922917.7
    ##   0.3  2          0.8               1.00       100      927011.9
    ##   0.3  2          0.8               1.00       150      929198.6
    ##   0.3  3          0.6               0.50        50      930125.8
    ##   0.3  3          0.6               0.50       100      937972.8
    ##   0.3  3          0.6               0.50       150      939775.3
    ##   0.3  3          0.6               0.75        50      930295.3
    ##   0.3  3          0.6               0.75       100      934369.4
    ##   0.3  3          0.6               0.75       150      936514.1
    ##   0.3  3          0.6               1.00        50      929184.6
    ##   0.3  3          0.6               1.00       100      934306.1
    ##   0.3  3          0.6               1.00       150      936859.8
    ##   0.3  3          0.8               0.50        50      934031.7
    ##   0.3  3          0.8               0.50       100      940293.7
    ##   0.3  3          0.8               0.50       150      943599.5
    ##   0.3  3          0.8               0.75        50      933853.9
    ##   0.3  3          0.8               0.75       100      939740.5
    ##   0.3  3          0.8               0.75       150      944983.0
    ##   0.3  3          0.8               1.00        50      929662.0
    ##   0.3  3          0.8               1.00       100      936588.8
    ##   0.3  3          0.8               1.00       150      942587.3
    ##   0.4  1          0.6               0.50        50      922586.0
    ##   0.4  1          0.6               0.50       100      923718.1
    ##   0.4  1          0.6               0.50       150      922260.1
    ##   0.4  1          0.6               0.75        50      922156.2
    ##   0.4  1          0.6               0.75       100      922135.8
    ##   0.4  1          0.6               0.75       150      922863.1
    ##   0.4  1          0.6               1.00        50      922386.1
    ##   0.4  1          0.6               1.00       100      922340.8
    ##   0.4  1          0.6               1.00       150      922426.4
    ##   0.4  1          0.8               0.50        50      921557.8
    ##   0.4  1          0.8               0.50       100      921207.7
    ##   0.4  1          0.8               0.50       150      921973.8
    ##   0.4  1          0.8               0.75        50      920962.6
    ##   0.4  1          0.8               0.75       100      920814.6
    ##   0.4  1          0.8               0.75       150      920820.3
    ##   0.4  1          0.8               1.00        50      920543.7
    ##   0.4  1          0.8               1.00       100      920613.6
    ##   0.4  1          0.8               1.00       150      920847.7
    ##   0.4  2          0.6               0.50        50      929444.1
    ##   0.4  2          0.6               0.50       100      932765.0
    ##   0.4  2          0.6               0.50       150      933061.4
    ##   0.4  2          0.6               0.75        50      926867.4
    ##   0.4  2          0.6               0.75       100      929183.5
    ##   0.4  2          0.6               0.75       150      930107.7
    ##   0.4  2          0.6               1.00        50      926083.4
    ##   0.4  2          0.6               1.00       100      929492.3
    ##   0.4  2          0.6               1.00       150      931061.2
    ##   0.4  2          0.8               0.50        50      926920.5
    ##   0.4  2          0.8               0.50       100      932704.8
    ##   0.4  2          0.8               0.50       150      937704.8
    ##   0.4  2          0.8               0.75        50      926746.8
    ##   0.4  2          0.8               0.75       100      929858.3
    ##   0.4  2          0.8               0.75       150      933551.5
    ##   0.4  2          0.8               1.00        50      924093.1
    ##   0.4  2          0.8               1.00       100      927565.7
    ##   0.4  2          0.8               1.00       150      931688.7
    ##   0.4  3          0.6               0.50        50      936305.2
    ##   0.4  3          0.6               0.50       100      939524.2
    ##   0.4  3          0.6               0.50       150      943095.4
    ##   0.4  3          0.6               0.75        50      932039.4
    ##   0.4  3          0.6               0.75       100      935220.1
    ##   0.4  3          0.6               0.75       150      937457.4
    ##   0.4  3          0.6               1.00        50      932537.4
    ##   0.4  3          0.6               1.00       100      936660.4
    ##   0.4  3          0.6               1.00       150      939162.5
    ##   0.4  3          0.8               0.50        50      939684.8
    ##   0.4  3          0.8               0.50       100      946327.5
    ##   0.4  3          0.8               0.50       150      949011.5
    ##   0.4  3          0.8               0.75        50      935316.3
    ##   0.4  3          0.8               0.75       100      943399.8
    ##   0.4  3          0.8               0.75       150      946938.3
    ##   0.4  3          0.8               1.00        50      934225.6
    ##   0.4  3          0.8               1.00       100      941966.0
    ##   0.4  3          0.8               1.00       150      945908.8
    ##   Rsquared    MAE     
    ##   0.02681149  647236.7
    ##   0.02868851  645496.4
    ##   0.02697025  647841.9
    ##   0.02792026  646067.0
    ##   0.02855064  645640.1
    ##   0.02892411  646108.0
    ##   0.02717619  646086.8
    ##   0.02825440  646027.2
    ##   0.02848626  646165.9
    ##   0.03158242  645351.9
    ##   0.03196226  645456.1
    ##   0.03314982  645311.0
    ##   0.03241801  645220.0
    ##   0.03393354  644171.1
    ##   0.03268552  645835.7
    ##   0.03166959  644891.9
    ##   0.03263045  644701.9
    ##   0.03286122  644884.9
    ##   0.02618626  646798.1
    ##   0.02333428  649302.5
    ##   0.02222103  649189.1
    ##   0.02711076  646020.4
    ##   0.02405870  648795.2
    ##   0.02288824  649303.5
    ##   0.02488366  648483.4
    ##   0.02383001  649620.4
    ##   0.02246078  650615.0
    ##   0.02763019  646340.4
    ##   0.02469057  647934.8
    ##   0.02379587  649842.2
    ##   0.02714518  646015.1
    ##   0.02759971  646370.1
    ##   0.02489761  648685.6
    ##   0.02957828  645248.4
    ##   0.02684329  647510.8
    ##   0.02567587  648390.6
    ##   0.02235867  650317.1
    ##   0.01777048  656678.0
    ##   0.01685126  657217.3
    ##   0.02197761  650361.3
    ##   0.01928491  652862.1
    ##   0.01788069  655308.7
    ##   0.02243352  649667.4
    ##   0.01916088  652969.9
    ##   0.01780701  654808.7
    ##   0.02235443  649162.0
    ##   0.02117719  654889.8
    ##   0.01939163  656741.8
    ##   0.02231595  649719.9
    ##   0.02101552  653960.1
    ##   0.01807146  657936.4
    ##   0.02444425  648654.7
    ##   0.02112876  652121.8
    ##   0.01923513  655150.8
    ##   0.02840259  647163.7
    ##   0.02667133  648518.3
    ##   0.02863564  645545.4
    ##   0.02877681  645310.3
    ##   0.02932877  647194.6
    ##   0.02778093  647447.4
    ##   0.02771448  646070.4
    ##   0.02847259  646156.0
    ##   0.02851893  646272.7
    ##   0.03198314  645246.3
    ##   0.03315079  645330.8
    ##   0.03163336  645789.7
    ##   0.03261295  644322.1
    ##   0.03335375  644715.2
    ##   0.03395350  644966.3
    ##   0.03226817  644636.6
    ##   0.03297360  644781.5
    ##   0.03307849  645069.1
    ##   0.02187566  650204.2
    ##   0.01970764  653119.9
    ##   0.02013520  652788.3
    ##   0.02462409  648384.4
    ##   0.02341242  649779.0
    ##   0.02359315  649664.0
    ##   0.02502416  648922.7
    ##   0.02291323  650284.3
    ##   0.02207632  651259.7
    ##   0.02785559  647625.3
    ##   0.02415552  649349.3
    ##   0.02162158  652663.1
    ##   0.02703366  646973.9
    ##   0.02680877  648086.5
    ##   0.02447206  649994.0
    ##   0.02906088  645282.3
    ##   0.02682358  647433.3
    ##   0.02438078  649764.8
    ##   0.01910404  654128.7
    ##   0.01676316  657570.5
    ##   0.01487281  657586.3
    ##   0.02012412  652448.0
    ##   0.01840305  654206.7
    ##   0.01759626  656606.4
    ##   0.02022769  651864.6
    ##   0.01791736  654736.8
    ##   0.01650943  656132.2
    ##   0.01899875  655607.0
    ##   0.01870980  656985.1
    ##   0.01731208  660066.0
    ##   0.02238641  650869.0
    ##   0.01878602  656432.7
    ##   0.01726761  658241.5
    ##   0.02146685  650940.0
    ##   0.01898405  655172.5
    ##   0.01786681  657640.9
    ## 
    ## Tuning parameter 'gamma' was held constant at a value of 0
    ## 
    ## Tuning parameter 'min_child_weight' was held constant at a value of 1
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were nrounds = 100, max_depth = 1,
    ##  eta = 0.3, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1
    ##  and subsample = 0.75.

Simply running the XGBoost Random Forest gave very bad rsquared correlations and high RMSEs. I will attempt to tune the paramters in XGBoost and see whether it will run better.

``` r
# Manually adding in a grid to tune three parameters:
tunegrid <- expand.grid(
  nrounds = 1000,
  eta = c(0.0001, 0.00001, 0.000001), # eta (learning rate) of 0.01, 0.001 are not great. Need to use smaller numbers
  max_depth = 1:10,
  gamma = 1,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.75
)

seed <- 10
set.seed(seed)
rf_xgbm2 <- train(
  sum_Purchase~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years + Marital_Status, data=dat_Train, # model
  method = 'xgbTree', #eXtreme gradient boosting method
  metric=metric, #using RMSE (root mean square error) to define my loss function
  trControl=control, # method="repeatedcv",number=10, repeats=3
  preProc=preProc, #centering and scaling the predictors
  tuneGrid=tunegrid,
  verbose=F)
print(rf_xgbm2)
```

    ## eXtreme Gradient Boosting 
    ## 
    ## 2946 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (34), scaled (34) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 2356, 2355, 2358, 2358, 2357, 2356, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   eta    max_depth  RMSE     Rsquared    MAE     
    ##   1e-06   1         1263716  0.01559107  850903.0
    ##   1e-06   2         1263710  0.02205901  850904.7
    ##   1e-06   3         1263707  0.02484273  850906.2
    ##   1e-06   4         1263707  0.02550011  850908.8
    ##   1e-06   5         1263708  0.02460678  850911.5
    ##   1e-06   6         1263709  0.02368539  850914.4
    ##   1e-06   7         1263710  0.02308850  850917.0
    ##   1e-06   8         1263711  0.02253786  850919.4
    ##   1e-06   9         1263712  0.02220469  850921.9
    ##   1e-06  10         1263712  0.02207817  850923.5
    ##   1e-05   1         1258538  0.01589084  843283.9
    ##   1e-05   2         1258475  0.02222606  843300.2
    ##   1e-05   3         1258453  0.02489100  843316.3
    ##   1e-05   4         1258451  0.02540530  843342.3
    ##   1e-05   5         1258458  0.02445103  843370.5
    ##   1e-05   6         1258467  0.02371769  843398.3
    ##   1e-05   7         1258474  0.02303303  843422.4
    ##   1e-05   8         1258485  0.02248101  843448.1
    ##   1e-05   9         1258493  0.02219640  843469.5
    ##   1e-05  10         1258501  0.02202013  843489.2
    ##   1e-04   1         1210527  0.01724229  771058.9
    ##   1e-04   2         1209923  0.02272700  771196.7
    ##   1e-04   3         1209737  0.02502589  771377.8
    ##   1e-04   4         1209726  0.02537880  771640.5
    ##   1e-04   5         1209767  0.02461501  771889.5
    ##   1e-04   6         1209839  0.02391126  772163.1
    ##   1e-04   7         1209931  0.02316589  772422.3
    ##   1e-04   8         1210024  0.02273843  772664.3
    ##   1e-04   9         1210112  0.02227572  772877.5
    ##   1e-04  10         1210175  0.02219704  773066.3
    ## 
    ## Tuning parameter 'nrounds' was held constant at a value of 1000
    ##  0.8
    ## Tuning parameter 'min_child_weight' was held constant at a value of
    ##  1
    ## Tuning parameter 'subsample' was held constant at a value of 0.75
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were nrounds = 1000, max_depth =
    ##  4, eta = 1e-04, gamma = 1, colsample_bytree = 0.8, min_child_weight =
    ##  1 and subsample = 0.75.
