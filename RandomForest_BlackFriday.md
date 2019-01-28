Random Forest
================
Lisa Leung
2019-01-27

Load all libraries
------------------

``` r
library(tidyverse)
```

    ## ── Attaching packages ──────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✔ ggplot2 3.1.0     ✔ purrr   0.2.4
    ## ✔ tibble  1.4.2     ✔ dplyr   0.7.4
    ## ✔ tidyr   0.8.0     ✔ stringr 1.3.0
    ## ✔ readr   1.1.1     ✔ forcats 0.3.0

    ## ── Conflicts ─────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
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
library(quantregForest)
```

    ## Loading required package: RColorBrewer

Purpose
-------

We will conduct random forest to predict the number of purchase made

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

Before starting random forest, I want to use a simple multiple regression to understand simple correlations among the data set

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

Multiple R-squared is squared of correlation between fitted and actual values. Residual standard error is root(mean squared error)

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

Interestingly, the spendings seem to affect most on city A and city B. We see that occupation 20 seem to spend $739,824 more on average

Let's start the random forest!
==============================

We will use train to tune parameters. The first train will be optimizing the randomly selected predictors
---------------------------------------------------------------------------------------------------------

``` r
seed <- 10
set.seed(seed)

# Setting parameters on mtry tuning
control <- trainControl( #trainControl is used to alter the default methods in train function
  method="repeatedcv", # K-fold CV; by default, it uses bootstrap sampling
  number=5, # 3 repeats of 5-fold CV
  repeats=3) 

metric <- "RMSE" #A string that specifies what summary metric will be used to select the optimal model. By default, possible values are "RMSE" and "Rsquared" for regression and "Accuracy" and "Kappa" for classification. 
mtry <- sqrt(ncol(dat_Train))
tunegrid <- expand.grid(.mtry=mtry) # Change this parameter to change the candidates for tuning parameters
preProc = c("center", "scale")
tunelength=10

# Running the random forest
# Note that I am doing a regression random forest so I'l be using quantile random forest. The train() function is used to tune the model
rf_simple <- train(
  sum_Purchase~Gender+Age+Occupation+City_Category+Stay_In_Current_City_Years + Marital_Status, data=dat_Train, # model
  method = 'qrf', #using quantile random forest to train.
  metric=metric, #using RMSE (root mean square error) to define my loss function
  tuneGrid=tunegrid, # Tuning parameters uses mytry (randomly selected predictors);
  trControl=control, # method="repeatedcv",number=10, repeats=3
  tunelength=tunelength,
  preProc=preProc) #centering and scaling the predictors
print(rf_simple)
```

    ## Quantile Random Forest 
    ## 
    ## 2946 samples
    ##    6 predictor
    ## 
    ## Pre-processing: centered (34), scaled (34) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 2356, 2355, 2358, 2358, 2357, 2356, ... 
    ## Resampling results:
    ## 
    ##   RMSE    Rsquared  MAE     
    ##   928303  0.125162  569048.3
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 3.162278

``` r
#ggplot(rf_simple)
```

Rsquared
