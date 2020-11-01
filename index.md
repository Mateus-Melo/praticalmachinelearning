---
title: "Pratical Machine Learning Course Project"
author: "Mateus Melo"
date: "30/10/2020"
output: 
  html_document: 
    keep_md: yes
---



## Loading and Setting The Data

We start our analysis by loading both the test and training datasets and looking at the number of variables and observations. 


```r
trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainingUrl,"training.csv")
download.file(testingUrl,"testing.csv")

training <- read.csv("training.csv")
testing <- read.csv("testing.csv")

dimTraining <- dim(training)
dimTesting <- dim(testing)

dimBoth <- rbind(dimTraining, dimTesting)
colnames(dimBoth) <- c("Observations", "Variables")
print(dimBoth)
```

```
##             Observations Variables
## dimTraining        19622       160
## dimTesting            20       160
```

Since we have a large number of observations in the  training dataset, we are going to break it into a new training and a validation dataset. This way, we are going to be able to perform a cross-validation in the following way: training our model with the training dataset, validating it with the validation dataset and testing it with the testing dataset to get unbiased results and an out-sample error. To make our analysis reproducible, we are going to set a seed.


```r
library(caret)

set.seed(1234)

inTrain <- createDataPartition(y=training$class, p=0.7, list=F)
validation <- training[-inTrain,]
training <- training[inTrain,]
```

## Variables Selection

Since we have 160 variables, build a model that uses all of them would be time consuming and would have a large variance. To avoid such a thing, we must cut down some of these variables. Let us take a look on some general information about them.


```r
summary(training)
```

```
##        X          user_name         raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    3   Length:13737       Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4903   Class :character   1st Qu.:1.323e+09    1st Qu.:252293      
##  Median : 9803   Mode  :character   Median :1.323e+09    Median :496316      
##  Mean   : 9814                      Mean   :1.323e+09    Mean   :498942      
##  3rd Qu.:14730                      3rd Qu.:1.323e+09    3rd Qu.:744387      
##  Max.   :19622                      Max.   :1.323e+09    Max.   :998750      
##                                                                              
##  cvtd_timestamp      new_window          num_window      roll_belt     
##  Length:13737       Length:13737       Min.   :  1.0   Min.   :-28.60  
##  Class :character   Class :character   1st Qu.:224.0   1st Qu.:  1.10  
##  Mode  :character   Mode  :character   Median :424.0   Median :113.00  
##                                        Mean   :431.3   Mean   : 64.13  
##                                        3rd Qu.:645.0   3rd Qu.:123.00  
##                                        Max.   :864.0   Max.   :162.00  
##                                                                        
##    pitch_belt          yaw_belt       total_accel_belt kurtosis_roll_belt
##  Min.   :-53.9000   Min.   :-179.00   Min.   : 0.00    Length:13737      
##  1st Qu.:  1.7900   1st Qu.: -88.30   1st Qu.: 3.00    Class :character  
##  Median :  5.2800   Median : -13.40   Median :17.00    Mode  :character  
##  Mean   :  0.4123   Mean   : -11.69   Mean   :11.28                      
##  3rd Qu.: 15.0000   3rd Qu.:  10.80   3rd Qu.:18.00                      
##  Max.   : 60.3000   Max.   : 179.00   Max.   :29.00                      
##                                                                          
##  kurtosis_picth_belt kurtosis_yaw_belt  skewness_roll_belt skewness_roll_belt.1
##  Length:13737        Length:13737       Length:13737       Length:13737        
##  Class :character    Class :character   Class :character   Class :character    
##  Mode  :character    Mode  :character   Mode  :character   Mode  :character    
##                                                                                
##                                                                                
##                                                                                
##                                                                                
##  skewness_yaw_belt  max_roll_belt     max_picth_belt  max_yaw_belt      
##  Length:13737       Min.   :-94.300   Min.   : 3.00   Length:13737      
##  Class :character   1st Qu.:-88.000   1st Qu.: 5.00   Class :character  
##  Mode  :character   Median : -5.700   Median :18.00   Mode  :character  
##                     Mean   : -3.847   Mean   :12.96                     
##                     3rd Qu.: 54.300   3rd Qu.:19.00                     
##                     Max.   :180.000   Max.   :30.00                     
##                     NA's   :13451     NA's   :13451                     
##  min_roll_belt      min_pitch_belt  min_yaw_belt       amplitude_roll_belt
##  Min.   :-180.000   Min.   : 0.00   Length:13737       Min.   :  0.000    
##  1st Qu.: -88.400   1st Qu.: 3.00   Class :character   1st Qu.:  0.300    
##  Median : -10.250   Median :16.00   Mode  :character   Median :  1.000    
##  Mean   :  -8.521   Mean   :10.57                      Mean   :  4.673    
##  3rd Qu.:  40.325   3rd Qu.:17.00                      3rd Qu.:  2.382    
##  Max.   : 173.000   Max.   :23.00                      Max.   :360.000    
##  NA's   :13451      NA's   :13451                      NA's   :13451      
##  amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt avg_roll_belt   
##  Min.   : 0.000       Length:13737       Min.   : 0.000       Min.   :-27.40  
##  1st Qu.: 1.000       Class :character   1st Qu.: 0.100       1st Qu.:  1.10  
##  Median : 1.000       Mode  :character   Median : 0.200       Median :115.45  
##  Mean   : 2.392                          Mean   : 1.119       Mean   : 67.09  
##  3rd Qu.: 3.000                          3rd Qu.: 0.400       3rd Qu.:124.20  
##  Max.   :12.000                          Max.   :16.500       Max.   :157.40  
##  NA's   :13451                           NA's   :13451        NA's   :13451   
##  stddev_roll_belt var_roll_belt     avg_pitch_belt    stddev_pitch_belt
##  Min.   : 0.000   Min.   :  0.000   Min.   :-51.400   Min.   :0.000    
##  1st Qu.: 0.200   1st Qu.:  0.000   1st Qu.: -1.425   1st Qu.:0.200    
##  Median : 0.400   Median :  0.100   Median :  4.900   Median :0.400    
##  Mean   : 1.575   Mean   :  9.843   Mean   : -1.014   Mean   :0.641    
##  3rd Qu.: 0.975   3rd Qu.:  0.875   3rd Qu.: 13.875   3rd Qu.:0.800    
##  Max.   :14.200   Max.   :200.700   Max.   : 59.700   Max.   :3.600    
##  NA's   :13451    NA's   :13451     NA's   :13451     NA's   :13451    
##  var_pitch_belt    avg_yaw_belt      stddev_yaw_belt    var_yaw_belt      
##  Min.   : 0.000   Min.   :-138.300   Min.   :  0.000   Min.   :    0.000  
##  1st Qu.: 0.000   1st Qu.: -88.200   1st Qu.:  0.100   1st Qu.:    0.010  
##  Median : 0.100   Median :  -7.000   Median :  0.300   Median :    0.095  
##  Mean   : 0.846   Mean   :  -6.582   Mean   :  1.693   Mean   :  152.299  
##  3rd Qu.: 0.600   3rd Qu.:  43.950   3rd Qu.:  0.775   3rd Qu.:    0.573  
##  Max.   :13.100   Max.   : 173.500   Max.   :176.600   Max.   :31183.240  
##  NA's   :13451    NA's   :13451      NA's   :13451     NA's   :13451      
##   gyros_belt_x        gyros_belt_y       gyros_belt_z      accel_belt_x    
##  Min.   :-1.040000   Min.   :-0.64000   Min.   :-1.4600   Min.   :-81.000  
##  1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.2000   1st Qu.:-21.000  
##  Median : 0.030000   Median : 0.02000   Median :-0.1000   Median :-15.000  
##  Mean   :-0.007405   Mean   : 0.03956   Mean   :-0.1285   Mean   : -5.718  
##  3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.: 0.0000   3rd Qu.: -5.000  
##  Max.   : 2.220000   Max.   : 0.64000   Max.   : 1.6100   Max.   : 85.000  
##                                                                            
##   accel_belt_y     accel_belt_z     magnet_belt_x    magnet_belt_y  
##  Min.   :-65.00   Min.   :-275.00   Min.   :-52.00   Min.   :354.0  
##  1st Qu.:  3.00   1st Qu.:-162.00   1st Qu.:  9.00   1st Qu.:581.0  
##  Median : 33.00   Median :-151.00   Median : 35.00   Median :601.0  
##  Mean   : 30.11   Mean   : -72.23   Mean   : 55.26   Mean   :593.5  
##  3rd Qu.: 61.00   3rd Qu.:  27.00   3rd Qu.: 59.00   3rd Qu.:610.0  
##  Max.   :150.00   Max.   : 103.00   Max.   :476.00   Max.   :673.0  
##                                                                     
##  magnet_belt_z       roll_arm         pitch_arm          yaw_arm         
##  Min.   :-623.0   Min.   :-180.00   Min.   :-88.800   Min.   :-180.0000  
##  1st Qu.:-376.0   1st Qu.: -30.50   1st Qu.:-25.600   1st Qu.: -42.8000  
##  Median :-320.0   Median :   0.00   Median :  0.000   Median :   0.0000  
##  Mean   :-346.2   Mean   :  18.43   Mean   : -4.611   Mean   :  -0.3954  
##  3rd Qu.:-306.0   3rd Qu.:  77.60   3rd Qu.: 11.000   3rd Qu.:  45.6000  
##  Max.   : 289.0   Max.   : 180.00   Max.   : 88.500   Max.   : 180.0000  
##                                                                          
##  total_accel_arm var_accel_arm      avg_roll_arm     stddev_roll_arm 
##  Min.   : 1.00   Min.   :  0.000   Min.   :-166.59   Min.   : 0.000  
##  1st Qu.:17.00   1st Qu.:  9.893   1st Qu.: -38.45   1st Qu.: 1.344  
##  Median :27.00   Median : 42.187   Median :   0.00   Median : 5.683  
##  Mean   :25.47   Mean   : 54.825   Mean   :  12.72   Mean   :10.888  
##  3rd Qu.:33.00   3rd Qu.: 78.138   3rd Qu.:  74.62   3rd Qu.:14.921  
##  Max.   :65.00   Max.   :331.699   Max.   : 163.33   Max.   :93.559  
##                  NA's   :13451     NA's   :13451     NA's   :13451   
##   var_roll_arm      avg_pitch_arm     stddev_pitch_arm var_pitch_arm     
##  Min.   :   0.000   Min.   :-81.773   Min.   : 0.000   Min.   :   0.000  
##  1st Qu.:   1.813   1st Qu.:-20.584   1st Qu.: 1.522   1st Qu.:   2.325  
##  Median :  32.293   Median :  0.000   Median : 8.084   Median :  65.351  
##  Mean   : 326.331   Mean   : -4.258   Mean   :10.515   Mean   : 202.790  
##  3rd Qu.: 222.647   3rd Qu.:  7.250   3rd Qu.:16.513   3rd Qu.: 272.687  
##  Max.   :8753.283   Max.   : 75.659   Max.   :43.412   Max.   :1884.565  
##  NA's   :13451      NA's   :13451     NA's   :13451    NA's   :13451     
##   avg_yaw_arm       stddev_yaw_arm     var_yaw_arm         gyros_arm_x     
##  Min.   :-164.639   Min.   :  0.000   Min.   :    0.000   Min.   :-6.3700  
##  1st Qu.: -25.004   1st Qu.:  1.955   1st Qu.:    3.843   1st Qu.:-1.3600  
##  Median :   0.000   Median : 18.556   Median :  344.342   Median : 0.0600  
##  Mean   :   5.422   Mean   : 22.857   Mean   : 1077.245   Mean   : 0.0268  
##  3rd Qu.:  39.978   3rd Qu.: 36.441   3rd Qu.: 1327.974   3rd Qu.: 1.5600  
##  Max.   : 150.458   Max.   :163.258   Max.   :26653.192   Max.   : 4.8700  
##  NA's   :13451      NA's   :13451     NA's   :13451                        
##   gyros_arm_y       gyros_arm_z       accel_arm_x       accel_arm_y     
##  Min.   :-3.4400   Min.   :-2.2800   Min.   :-383.00   Min.   :-318.00  
##  1st Qu.:-0.7900   1st Qu.:-0.0800   1st Qu.:-242.00   1st Qu.: -53.00  
##  Median :-0.2200   Median : 0.2100   Median : -45.00   Median :  16.00  
##  Mean   :-0.2494   Mean   : 0.2619   Mean   : -60.73   Mean   :  33.64  
##  3rd Qu.: 0.1600   3rd Qu.: 0.7200   3rd Qu.:  82.00   3rd Qu.: 140.00  
##  Max.   : 2.8400   Max.   : 3.0200   Max.   : 437.00   Max.   : 303.00  
##                                                                         
##   accel_arm_z       magnet_arm_x     magnet_arm_y     magnet_arm_z   
##  Min.   :-630.00   Min.   :-584.0   Min.   :-386.0   Min.   :-597.0  
##  1st Qu.:-142.00   1st Qu.:-303.0   1st Qu.:  -7.0   1st Qu.: 137.0  
##  Median : -47.00   Median : 280.0   Median : 204.0   Median : 444.0  
##  Mean   : -70.79   Mean   : 189.8   Mean   : 157.8   Mean   : 307.9  
##  3rd Qu.:  24.00   3rd Qu.: 638.0   3rd Qu.: 325.0   3rd Qu.: 545.0  
##  Max.   : 292.00   Max.   : 782.0   Max.   : 583.0   Max.   : 694.0  
##                                                                      
##  kurtosis_roll_arm  kurtosis_picth_arm kurtosis_yaw_arm   skewness_roll_arm 
##  Length:13737       Length:13737       Length:13737       Length:13737      
##  Class :character   Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character   Mode  :character  
##                                                                             
##                                                                             
##                                                                             
##                                                                             
##  skewness_pitch_arm skewness_yaw_arm    max_roll_arm    max_picth_arm    
##  Length:13737       Length:13737       Min.   :-73.10   Min.   :-164.00  
##  Class :character   Class :character   1st Qu.:  0.00   1st Qu.:   0.00  
##  Mode  :character   Mode  :character   Median :  5.10   Median :  31.90  
##                                        Mean   : 11.79   Mean   :  39.94  
##                                        3rd Qu.: 25.70   3rd Qu.: 100.00  
##                                        Max.   : 85.50   Max.   : 180.00  
##                                        NA's   :13451    NA's   :13451    
##   max_yaw_arm     min_roll_arm    min_pitch_arm      min_yaw_arm   
##  Min.   : 4.00   Min.   :-88.80   Min.   :-180.00   Min.   : 1.00  
##  1st Qu.:29.00   1st Qu.:-41.62   1st Qu.: -69.17   1st Qu.: 8.00  
##  Median :34.00   Median :-22.45   Median : -32.55   Median :12.00  
##  Mean   :35.15   Mean   :-21.25   Mean   : -32.36   Mean   :14.19  
##  3rd Qu.:41.00   3rd Qu.:  0.00   3rd Qu.:   0.00   3rd Qu.:18.00  
##  Max.   :62.00   Max.   : 66.40   Max.   : 140.00   Max.   :38.00  
##  NA's   :13451   NA's   :13451    NA's   :13451     NA's   :13451  
##  amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm roll_dumbbell    
##  Min.   :  0.00     Min.   :  0.00      Min.   : 0.00     Min.   :-153.71  
##  1st Qu.:  5.70     1st Qu.:  9.22      1st Qu.:13.00     1st Qu.: -18.31  
##  Median : 28.57     Median : 58.95      Median :22.00     Median :  47.93  
##  Mean   : 33.04     Mean   : 72.30      Mean   :20.96     Mean   :  23.70  
##  3rd Qu.: 51.85     3rd Qu.:120.17      3rd Qu.:29.00     3rd Qu.:  67.40  
##  Max.   :119.50     Max.   :359.00      Max.   :51.00     Max.   : 153.38  
##  NA's   :13451      NA's   :13451       NA's   :13451                      
##  pitch_dumbbell     yaw_dumbbell      kurtosis_roll_dumbbell
##  Min.   :-149.59   Min.   :-150.871   Length:13737          
##  1st Qu.: -40.96   1st Qu.: -77.662   Class :character      
##  Median : -20.89   Median :  -3.323   Mode  :character      
##  Mean   : -10.70   Mean   :   1.895                         
##  3rd Qu.:  17.80   3rd Qu.:  80.182                         
##  Max.   : 149.40   Max.   : 154.952                         
##                                                             
##  kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
##  Length:13737            Length:13737          Length:13737          
##  Class :character        Class :character      Class :character      
##  Mode  :character        Mode  :character      Mode  :character      
##                                                                      
##                                                                      
##                                                                      
##                                                                      
##  skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
##  Length:13737            Length:13737          Min.   :-70.10   
##  Class :character        Class :character      1st Qu.:-27.88   
##  Mode  :character        Mode  :character      Median : 10.60   
##                                                Mean   : 12.72   
##                                                3rd Qu.: 48.38   
##                                                Max.   :137.00   
##                                                NA's   :13451    
##  max_picth_dumbbell max_yaw_dumbbell   min_roll_dumbbell min_pitch_dumbbell
##  Min.   :-112.90    Length:13737       Min.   :-134.90   Min.   :-147.00   
##  1st Qu.: -67.45    Class :character   1st Qu.: -61.17   1st Qu.: -96.20   
##  Median :  24.40    Mode  :character   Median : -45.35   Median : -74.40   
##  Mean   :  25.13                       Mean   : -43.63   Mean   : -41.19   
##  3rd Qu.: 127.35                       3rd Qu.: -28.05   3rd Qu.:  10.03   
##  Max.   : 155.00                       Max.   :  73.20   Max.   : 120.90   
##  NA's   :13451                         NA's   :13451     NA's   :13451     
##  min_yaw_dumbbell   amplitude_roll_dumbbell amplitude_pitch_dumbbell
##  Length:13737       Min.   :  0.00          Min.   :  0.00          
##  Class :character   1st Qu.: 14.12          1st Qu.: 16.88          
##  Mode  :character   Median : 34.55          Median : 41.66          
##                     Mean   : 56.35          Mean   : 66.32          
##                     3rd Qu.: 86.08          3rd Qu.:101.04          
##                     Max.   :256.48          Max.   :273.59          
##                     NA's   :13451           NA's   :13451           
##  amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
##  Length:13737           Min.   : 0.00        Min.   :  0.000   
##  Class :character       1st Qu.: 4.00        1st Qu.:  0.400   
##  Mode  :character       Median :10.00        Median :  1.143   
##                         Mean   :13.73        Mean   :  4.931   
##                         3rd Qu.:20.00        3rd Qu.:  3.507   
##                         Max.   :58.00        Max.   :230.428   
##                                              NA's   :13451     
##  avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell  avg_pitch_dumbbell
##  Min.   :-128.96   Min.   :  0.00       Min.   :    0.00   Min.   :-70.53    
##  1st Qu.: -11.08   1st Qu.:  4.59       1st Qu.:   21.07   1st Qu.:-42.34    
##  Median :  51.81   Median : 11.87       Median :  140.80   Median :-21.69    
##  Mean   :  25.10   Mean   : 20.65       Mean   : 1020.37   Mean   :-13.06    
##  3rd Qu.:  65.47   3rd Qu.: 26.36       3rd Qu.:  694.65   3rd Qu.: 11.84    
##  Max.   : 125.99   Max.   :123.78       Max.   :15321.01   Max.   : 94.28    
##  NA's   :13451     NA's   :13451        NA's   :13451      NA's   :13451     
##  stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell  
##  Min.   : 0.000        Min.   :   0.00    Min.   :-117.950  
##  1st Qu.: 3.431        1st Qu.:  11.77    1st Qu.: -77.071  
##  Median : 8.013        Median :  64.21    Median : -16.441  
##  Mean   :13.286        Mean   : 342.40    Mean   :  -7.971  
##  3rd Qu.:19.976        3rd Qu.: 399.03    3rd Qu.:  55.795  
##  Max.   :62.881        Max.   :3953.97    Max.   : 134.905  
##  NA's   :13451         NA's   :13451      NA's   :13451     
##  stddev_yaw_dumbbell var_yaw_dumbbell   gyros_dumbbell_x    gyros_dumbbell_y 
##  Min.   :  0.000     Min.   :    0.00   Min.   :-204.0000   Min.   :-2.0700  
##  1st Qu.:  3.881     1st Qu.:   15.06   1st Qu.:  -0.0300   1st Qu.:-0.1400  
##  Median : 10.238     Median :  104.83   Median :   0.1400   Median : 0.0300  
##  Mean   : 16.615     Mean   :  586.37   Mean   :   0.1582   Mean   : 0.0453  
##  3rd Qu.: 25.439     3rd Qu.:  647.15   3rd Qu.:   0.3500   3rd Qu.: 0.2100  
##  Max.   :107.088     Max.   :11467.91   Max.   :   2.2000   Max.   :52.0000  
##  NA's   :13451       NA's   :13451                                           
##  gyros_dumbbell_z   accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z 
##  Min.   : -2.3000   Min.   :-419.00   Min.   :-189.00   Min.   :-334.00  
##  1st Qu.: -0.3100   1st Qu.: -51.00   1st Qu.:  -8.00   1st Qu.:-142.00  
##  Median : -0.1300   Median :  -8.00   Median :  41.00   Median :  -1.00  
##  Mean   : -0.1248   Mean   : -28.56   Mean   :  52.43   Mean   : -38.08  
##  3rd Qu.:  0.0300   3rd Qu.:  11.00   3rd Qu.: 111.00   3rd Qu.:  39.00  
##  Max.   :317.0000   Max.   : 235.00   Max.   : 315.00   Max.   : 318.00  
##                                                                          
##  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z  roll_forearm    
##  Min.   :-643.0    Min.   :-3600.0   Min.   :-262.00   Min.   :-180.00  
##  1st Qu.:-535.0    1st Qu.:  230.0   1st Qu.: -45.00   1st Qu.:   0.00  
##  Median :-479.0    Median :  310.0   Median :  13.00   Median :  23.50  
##  Mean   :-326.7    Mean   :  218.4   Mean   :  46.74   Mean   :  35.24  
##  3rd Qu.:-302.0    3rd Qu.:  389.0   3rd Qu.:  96.00   3rd Qu.: 141.00  
##  Max.   : 592.0    Max.   :  633.0   Max.   : 452.00   Max.   : 180.00  
##                                                                         
##  pitch_forearm     yaw_forearm      kurtosis_roll_forearm
##  Min.   :-72.50   Min.   :-180.00   Length:13737         
##  1st Qu.:  0.00   1st Qu.: -68.90   Class :character     
##  Median :  9.28   Median :   0.00   Mode  :character     
##  Mean   : 10.59   Mean   :  19.26                        
##  3rd Qu.: 28.20   3rd Qu.: 110.00                        
##  Max.   : 88.70   Max.   : 180.00                        
##                                                          
##  kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
##  Length:13737           Length:13737         Length:13737         
##  Class :character       Class :character     Class :character     
##  Mode  :character       Mode  :character     Mode  :character     
##                                                                   
##                                                                   
##                                                                   
##                                                                   
##  skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm max_picth_forearm
##  Length:13737           Length:13737         Min.   :-64.00   Min.   :-151.0   
##  Class :character       Class :character     1st Qu.:  0.00   1st Qu.:   0.0   
##  Mode  :character       Mode  :character     Median : 24.55   Median : 110.0   
##                                              Mean   : 23.74   Mean   :  80.9   
##                                              3rd Qu.: 45.15   3rd Qu.: 173.0   
##                                              Max.   : 89.80   Max.   : 180.0   
##                                              NA's   :13451    NA's   :13451    
##  max_yaw_forearm    min_roll_forearm  min_pitch_forearm min_yaw_forearm   
##  Length:13737       Min.   :-66.600   Min.   :-180.00   Length:13737      
##  Class :character   1st Qu.: -6.475   1st Qu.:-174.00   Class :character  
##  Mode  :character   Median :  0.000   Median : -44.55   Mode  :character  
##                     Mean   : -0.175   Mean   : -53.94                     
##                     3rd Qu.: 10.775   3rd Qu.:   0.00                     
##                     Max.   : 62.100   Max.   : 167.00                     
##                     NA's   :13451     NA's   :13451                       
##  amplitude_roll_forearm amplitude_pitch_forearm amplitude_yaw_forearm
##  Min.   :  0.00         Min.   :  0.0           Length:13737         
##  1st Qu.:  0.90         1st Qu.:  1.0           Class :character     
##  Median : 15.59         Median : 84.6           Mode  :character     
##  Mean   : 23.92         Mean   :134.8                                
##  3rd Qu.: 39.62         3rd Qu.:349.0                                
##  Max.   :126.00         Max.   :360.0                                
##  NA's   :13451          NA's   :13451                                
##  total_accel_forearm var_accel_forearm avg_roll_forearm  stddev_roll_forearm
##  Min.   :  0.00      Min.   :  0.000   Min.   :-177.23   Min.   :  0.000    
##  1st Qu.: 29.00      1st Qu.:  6.572   1st Qu.:   0.00   1st Qu.:  0.297    
##  Median : 36.00      Median : 23.358   Median :  14.09   Median :  6.833    
##  Mean   : 34.69      Mean   : 33.617   Mean   :  33.77   Mean   : 43.162    
##  3rd Qu.: 41.00      3rd Qu.: 51.145   3rd Qu.: 109.13   3rd Qu.: 89.064    
##  Max.   :108.00      Max.   :172.606   Max.   : 177.12   Max.   :179.171    
##                      NA's   :13451     NA's   :13451     NA's   :13451      
##  var_roll_forearm   avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm 
##  Min.   :    0.00   Min.   :-65.25    Min.   : 0.000       Min.   :   0.000  
##  1st Qu.:    0.09   1st Qu.:  0.00    1st Qu.: 0.260       1st Qu.:   0.068  
##  Median :   46.83   Median : 11.60    Median : 4.998       Median :  24.984  
##  Mean   : 5509.77   Mean   : 11.44    Mean   : 7.763       Mean   : 137.736  
##  3rd Qu.: 7935.87   3rd Qu.: 26.46    3rd Qu.:12.967       3rd Qu.: 168.150  
##  Max.   :32102.24   Max.   : 72.09    Max.   :47.745       Max.   :2279.617  
##  NA's   :13451      NA's   :13451     NA's   :13451        NA's   :13451     
##  avg_yaw_forearm   stddev_yaw_forearm var_yaw_forearm    gyros_forearm_x   
##  Min.   :-155.06   Min.   :  0.00     Min.   :    0.00   Min.   :-22.0000  
##  1st Qu.: -21.09   1st Qu.:  0.47     1st Qu.:    0.22   1st Qu.: -0.2200  
##  Median :   0.00   Median : 24.96     Median :  622.99   Median :  0.0500  
##  Mean   :  20.00   Mean   : 43.87     Mean   : 4537.98   Mean   :  0.1573  
##  3rd Qu.:  85.93   3rd Qu.: 79.27     3rd Qu.: 6283.69   3rd Qu.:  0.5600  
##  Max.   : 169.24   Max.   :197.51     Max.   :39009.33   Max.   :  3.9700  
##  NA's   :13451     NA's   :13451      NA's   :13451                        
##  gyros_forearm_y    gyros_forearm_z    accel_forearm_x   accel_forearm_y 
##  Min.   : -7.0200   Min.   : -6.9900   Min.   :-498.00   Min.   :-632.0  
##  1st Qu.: -1.4800   1st Qu.: -0.1800   1st Qu.:-177.00   1st Qu.:  57.0  
##  Median :  0.0300   Median :  0.0800   Median : -56.00   Median : 201.0  
##  Mean   :  0.0734   Mean   :  0.1566   Mean   : -60.74   Mean   : 163.5  
##  3rd Qu.:  1.6100   3rd Qu.:  0.4900   3rd Qu.:  77.00   3rd Qu.: 312.0  
##  Max.   :311.0000   Max.   :231.0000   Max.   : 389.00   Max.   : 923.0  
##                                                                          
##  accel_forearm_z   magnet_forearm_x  magnet_forearm_y magnet_forearm_z
##  Min.   :-391.00   Min.   :-1280.0   Min.   :-896.0   Min.   :-973.0  
##  1st Qu.:-182.00   1st Qu.: -615.0   1st Qu.:   0.0   1st Qu.: 185.0  
##  Median : -40.00   Median : -377.0   Median : 592.0   Median : 511.0  
##  Mean   : -55.14   Mean   : -312.2   Mean   : 379.8   Mean   : 392.1  
##  3rd Qu.:  27.00   3rd Qu.:  -73.0   3rd Qu.: 737.0   3rd Qu.: 653.0  
##  Max.   : 287.00   Max.   :  672.0   Max.   :1480.0   Max.   :1090.0  
##                                                                       
##     classe         
##  Length:13737      
##  Class :character  
##  Mode  :character  
##                    
##                    
##                    
## 
```

The first 7 variables represent some general information about that activities that may be not important or even misleading to the model building. Also, we have a large number of NAs in some variables and some have the words std, avg and var on their names, indicating that they must be highly correlated with some other variables. Let us remove these variables from the datasets. Then, we will see how many variables we have left.


```r
takeOut <- 1:7
stdNames <- grep("std",colnames(training))
avgNames <- grep("avg", colnames(training))
varNames <- grep("var", colnames(training))

for(i in 8:(ncol(training)-1)){
        if(sum(is.na(training[,i]))>0|sum(training[,i]==""))
                takeOut <- c(takeOut, i)
}
takeOut <- c(takeOut, stdNames, avgNames, varNames)
training <- training[,-takeOut]
testing <- testing[,-takeOut]
validation <- validation[,-takeOut]
rbind(dim(training),dim(validation),dim(testing))
```

```
##       [,1] [,2]
## [1,] 13737   53
## [2,]  5885   53
## [3,]    20   53
```

We end up with 53 variables, which still is a large amount.

## Model Building

Since we have a classification problem and the model interpretability is not crucial, a random forest model would be an interesting choice providing a good accuracy. We are going to use it along with a PCA pre-processing so we can lower the number of predictors. Since the a random forest model can take a long time to be built, we are going to make use of the CPU multiple cores. We are going to perform a 5 fold cross validation to get an estimate of the out-of-sample error. We will start with a PCA that explains 80% of the variance. Then, we will check whether or not this give us an accuracy larger than 90% in the validation dataset. If not, we will keep rising it until the accuracy is large enough.


```r
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method="cv", number = 5, preProcOptions=list(thresh=0.8), allowParallel = T)
y = training[,53]
x = training[,-53]
fit1 <- train(x,y, method="rf",preProc ="pca", trControl = fitControl)
fit1
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: principal component signal extraction (52), centered
##  (52), scaled (52) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10989, 10991, 10989, 10991, 10988 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9500625  0.9368300
##   27    0.9396511  0.9236562
##   52    0.9387775  0.9225562
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

We have got an out-of-sample accuracy estimate of 95%, which is good enough. Now let us test the model in both validation and training dataset.


```r
validationPred <- predict(fit1, validation)
confusionMatrix(factor(validationPred),factor(validation$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1633   17    9   13    1
##          B   12 1083   25    4   13
##          C   11   26  975   48    7
##          D   17   10   14  897    4
##          E    1    3    3    2 1057
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9592          
##                  95% CI : (0.9538, 0.9641)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9484          
##                                           
##  Mcnemar's Test P-Value : 0.0005138       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9755   0.9508   0.9503   0.9305   0.9769
## Specificity            0.9905   0.9886   0.9811   0.9909   0.9981
## Pos Pred Value         0.9761   0.9525   0.9138   0.9522   0.9916
## Neg Pred Value         0.9903   0.9882   0.9894   0.9864   0.9948
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2775   0.1840   0.1657   0.1524   0.1796
## Detection Prevalence   0.2843   0.1932   0.1813   0.1601   0.1811
## Balanced Accuracy      0.9830   0.9697   0.9657   0.9607   0.9875
```

```r
trainingPred <- predict(fit1, training)
confusionMatrix(factor(trainingPred),factor(training$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```


We have got an accuracy of 100% in the training dataset. That happens due to overfitting the model. In the validation dataset we've had an accuracy of almost 96% which is close enough to our previous estimate.

## Testing The Model


```r
testPred <- predict(fit1, testing)
testPred
```

```
##  [1] B A A A A E D B A A A C B A E E A B B B
## Levels: A B C D E
```

Submitting the answers to the quiz, we have got an accuracy of 90%.
