
- [01. introduction](#01)
- [02. linear regression](#02)
- [03. logistic regression](#03)
- [04. regularization](#04)
- [05. neural network](#05)
- [06. some advice](#06)
- [07. SVM](#07)
- [08. clustering](#08)
- [09. dimensionality reduction](#09)
- [10. anomaly detection](#10)
- [11. recommender systems](#11)
- [12. large scale machine learning](#12)
- [13. application photo OCR](#13)

This is a note to record some key points in course "machine learning" of Andrew. I think it is a good introductory course in machine learning, which not only talking about the basic theory of pupular machine learning model also introduce some hot application like recommender system and photo OCR.  

It is strongly recommend to follow the whole couse and finish the assignment and course project. 

<h1 id="01"> 01. introduction </h1>


## Welcome  

**Machine Learning**  

* Grew out of work in AI
* New capability of computers 

**Examples**

* Database Mining
    - Large datasets from growth of automation/web
    - E.g., Web clicimak data, medical records, biology, engineering
* Application can't program by hand
    - E.g., Autonomous helicoper, handwriting recognition, most of natural language processing (NLP), Computer Vision
* Self-customizing programs
    - E.g., Amazon, Netflix product recommendations 
* Understanding human learning (brain, real AI )

## What is machine learning  

**Machine Learning Definition**
> Arthur Samuel(1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.  
> Tom Mitchell(1998). Well-posed Learning Problem: A computer program is said to learn from **experience E** with respect to some **task T** and some **performance measure P**, if its **performance on T, as measured by P, improves with experience E**.  

**This course content**

* Machine Learning:
    - supervised learning: "right answers"(ground truth) given
        * regression: label is continuous value 
        * classification: label is categories 
    - unsupervised learning
        * Clustering  
        * Dimensionality Reduction  
        * Anomaly detection
        * application: Organize computing clusters, Social network analysis, Market segmentation, Astronomical data analysis
* Others:
    - Reinforcement learning, recommender systems
* Also talk about: 
    - Practical advice for applying learning algorithms

![Supervised learning](image/01_supervised.png "Supervised Learning")
![Unsupervised learning](image/01_unsupervised.png "Supervised Learning")


<h1 id="02"> 02. linear regression </h1>
 

## model representation

In general, learning is a process to fit a hypothesis from data, however there are more than one hypothesis, the winner will be choosed by some given inductive bias.   

Linear regression can be view as a process to use straight lines to fit the trajectory of data, and the line with minimum error will be prefer, also to minimize the distence from each data point to line.  

![](image/02_model_representation.png)

## cost function

Cost function is convex function.

![](image/02_define_cost_function.png)
![](image/02_plot_cost_function.png)

## gradient descent (GD)

Learning: adjust the parameter based on some rule to find global minimum in solution space, which space in linear regression just like the curve above.   

Gradient Descent is a method to update parameter, along the gradient is the increasing direction so along the reverse direction can reduce the objective function. 

![](image/02_gd_definition.png)
![](image/02_gd_algorighm.png)

- note: different initial point may lead different extremum. 
- random choose the initial point and repeat more than ones. 

![](image/02_gd_plot_1.png)
![](image/02_gd_plot_2.png)

## one result

![](image/02_best_fit.png)


# Linear regression with multiple variable

### model definition

![](image/03_multiple_variable.png)   
![](image/03_multiple_variable_polynomial.png)
![](image/03_multiple_variable_polynomial_2.png)

### GD Learning

![](image/03_multiple_variable_gd.png)
![](image/03_multiple_variable_gd_scaling_1.png)
![](image/03_multiple_variable_gd_scaling_2.png)
![](image/03_multiple_variable_gd_mean_normalization.png)

### Debugging and Learning Rate

![](image/03_multiple_variable_gd_learning_rate_2.png)
![](image/03_multiple_variable_gd_learning_rate_3.png)
![](image/03_multiple_variable_gd_learning_rate_4.png)

![](image/02_gd_adative_learning_rate.png)

- The GD above is stochastic GD (SGD), each GD just use one training instance.
- Batch GD: each step of GD uses all training examples. 
- Mini Batch, each step of GD uses part of training examples. (split the whole training set into multiple block, each block can lead GD ones.)

### normal equation

![](image/03_multiple_variable_normal_equaltion.png)
![](image/03_multiple_variable_normal_equaltion_2.png)
![](image/03_multiple_variable_normal_equaltion_3.png)
![](image/03_multiple_variable_normal_equaltion_non_invertible.png)

### GD vs Normal equation
![](image/03_multiple_variable_gd_vs_normal_equaltion.png)


<h1 id="03"> 03.logistic regression </h1>

## Abstract

- some application
    - Email: Spam / Not Spam
    - Online Transactions : Fraudulent (Yes/No)
    - Tumor : Malignant / Benign

- classification:
    - hypothesis : h, threshold : alpha (decision boundary)
        - if h(x) >= alpha, predict 1
        - if h(x) < alpha, predict 0
        - Note: h(x) is a real value
    - logistic regression, map h(x) to range [0,1], then h(x) can be the probability of predicting 1.

- logistic regression
    - sigmoid function: h(x) = 1 / (1+exp(-w*x)), h(x) = [0,1]
    - h(x) = estimated probability that y=1 on input x
    - binary distribution(Bernoulli distributed): p = h(x)^y*(1-h(x))^(1-y)  
    - maximize maximum likelihood: maximize p <-> maximize ln(p)
    - maximize ln(p) = maximize yln(h(x)) + (1-(h(x)))ln(1-h(x))
    - maximize ln(p) = minimize -ln(p) = minimize -(yln(h(x)) + (1-(h(x)))ln(1-h(x)))
    - cost function, sum up on all m instance

- other topic:
    - optimization method: GD or advanced method: BFGS, L-BFGS, Conjugate gradient
    - multi-class classification: one-vs-all
    - logistic regression is a special case of generalized linear model 


## Hypothesis Representation

![](image/04_lr_sigmoid.png)
![](image/04_lr_output.png)

## Decision boundary
![](image/04_lr_decision_boundary_1.png)  
![](image/04_lr_decision_boundary_2.png)
![](image/04_lr_decision_boundary_3.png)

## Cost Function
![](image/04_lr_cost_function_1.png)  
![](image/04_lr_cost_function_2.png)  
![](image/04_lr_cost_function_3.png)
![](image/04_lr_cost_function_4.png)
![](image/04_lr_cost_function_5.png)

## Optimization 

![](image/04_lr_gd_1.png)
![](image/04_lr_gd_2.png)
![](image/04_lr_advanced_optimization.png)  


## Multi-class 

![](image/04_lr_multiple_class_2.png)  
![](image/04_lr_multiple_class.png)



<h1 id="04"> 04. regularization </h1>


- regularization is a efficent way to address overfitting problem.
- overfitting: the learned hypothesis may fit the training set very well, but fail to generalize to new examples. 
- cost funtion can be split into two part
    - error part: measure the accuracy of model
    - regularization part : punish the complexity of model
        - L2 : ridge regression, prefer more feature with small weight. works well, when we have a lots of features, each of which contributes a bit to predicting y.
        - L1 : Lasso, sparcity, feature selection. 
- regularization parameter:
    - too large: underfitting
    - too small: cann't address the overfitting problem

## overfitting
![](image/05_overfitting.png)
![](image/05_address_overfitting.png)

## regularization part

![](image/05_regularization_gd_linear.png)
![](image/05_regularization_gd_logistic.png)
![](image/05_regularization_gd_normal.png)
![](image/05_regularization_gd_normal2.png)

## regularization parameter

![](image/05_regularization_parameter_2.png)
![](image/05_regularization_parameter.png)
![](image/05_regularization.png)


<h1 id="05"> 05. neural network </h1>

- brief introduction 
    - non-linear hypothesis
    - Origins: Algorithms that try to mimic the brain
    - Was very widely used in 80s and early 90s; popularity diminished in late 90s
    - Recent resurgence: State-of-the-art techique for many applications

- component:
    - neurons 
    - activation
    - bias / threshold, which define a neuron is activated or not. 
    - layer
        - input layer
        - hidden layer: active function always is the sigmoid functon 
        - out put layer : fully connecting, one-hot for multi-class

- basic operation:
    - inference: forward propogate 
    - learning: backpropogate (BP)(error)

- basic example:
    - AND, OR, XOR

- multi-classification:
    - one-vs-all, one-hot-vector

- training a neural network:
    - Randomly initialize weights
    - Implement forward propagation, get predict value for each input
    - Implement backpropagation to minimize error
    - gradient checking 
    - use gradient descent or advanced optimization method with backprobagation to try to minimize cost funciton 


## Basic 

### non-linear hypothesis**  

![](image/06_non_linear.png)
![](image/06_classify_car.png)

### mimic brain 

![](image/06_brain_1.png)
![](image/06_brain_2.png)

### model representation

![](image/06_neural_in_brain.png)

![](image/06_neural_network.png)
![](image/06_neural_network_2.png)

![](image/06_forward_propagation.png)

### example

![](image/06_example_1.png)
![](image/06_example_2.png)

### multi-class classification

![](image/06_multiple_classification_1.png)
![](image/06_multiple_classification_2.png)


## Cost function

![](image/07_definition.png)  
![](image/07_cost_function.png)

![](image/07_bp_algo.png)  
![](image/07_gradient_checking.png)


## putting together


![](image/07_putting_together_1.png)
  
![](image/07_putting_together_2.png)

![](image/07_putting_together_3.png)

![](image/07_implement_note_1.png)



<h1 id="06"> 06. some advise for applying machine learning</h1>


- underfitting vs overffiting
- bias vs variance
- split data to evaluating model
    - training data : train model
    - validation data : model selection
    - testing data : test generative ability
- some helpful figure
   - x : different model (diff structure or parameter) vs y : different kinds of error
   - learning curve: x : data set size vs y : different error
- machine learning diagnostic
   - diagnostic: A test that you can run to gain insight what is/isn't working with a learning algorithm, and gain guidance as to how best to improve its performace.
   - diagnostic can take time to implement, but doing so can be a very good use of your time.  

- some solution 
    - get more training examples (high variance will help)
    - try smaller sets of features (high variance)
    - try getting additional features (high bias)
    - try adding polynomial feature (high bias)
    - try decreasing regularization parameter (high bias)
    - try increasing regularization parameter (high variance)

## overfitting 

![](image/08_overfitting_example.png)
![](image/08_evaluating_hypothesis.png)

## bias vs variance

![](image/08_bias_variance_1.png)
![](image/08_bias_variance_2.png)

## Learning curve

![](image/08_learning_curve_1.png)
  
![](image/08_learning_curve_2.png)

![](image/08_learning_curve_3.png)

## Regularization parameter selection

![](image/08_regularization_parameter_1.png)
![](image/08_regularization_parameter_2.png)

## neural network

![](image/08_neural_network.png)

# Machine learning system design

### example-spam classifier:

- collect lots of data
- feature preparation 
    - develop sophisticated features based on email routing information (from email header)
    - develop sophisticated features for message body 
    - develop sophisticated algorithm to detect misspellings. 
- choose classifer:
    - logistic regression, cnn ...

### recommended approach:
	
- start with a simple algorithm that you can implement quickly
- plot learning curves to decide if more data, more features, etc. are likely to help
- error analysis 
    - manually examine the examples (in cross validation set) that your algorithm made errors.
    - see if you spot any systematic trend in what type of examples it is making errors on.
    - some nemerical evaluation methods are important
        - accuracy 
        - error 
    - control variate method to inspect which can improve your result
    - Only solution is to try it and see if it words 
- unbalanced data evaluation
    - precision / recall (PR curve, how to balance precision and recall)
    - F-scoer, f1-score

![](image/09_pr_balance.png)

### It's not who has the best algorithm that wins, It's who has the most data.  

![](image/09_more_data.png)  
![](image/09_more_data_2.png)
![](image/09_more_data_3.png)


<h1 id="07"> 07. SVM </h1>

# Support Vector Machine

- optimization objective:
    - large margin  
- linear separable:
    - Yes : linear svm
    - No : kernel function (Gaussian kernel, String kernel, chi-square kernel, histogram intersection kernel)
- parameter
    - C : is the reciprocal of the regularization parameter, (1/lamda)
    - kernel parameter : 
- overfitting:
    - noise: slack variable
    - cross-validation to choose model parameter
- multi-class classfication
    - on-vs-all
- logistic regression vs SVM

![](image/10_use_svm_5.png)

## Optimization objective and Large Margin Intuition

![](image/10_large_margin.png)

![](image/10_decision_boundary.png)

![](image/10_decision_boundary2.png)
![](image/10_decistion_boundary3.png)

## The mathematics behind large margin classification 

![](image/10_math_1.png)

![](image/10_math_2.png)
	
![](image/10_math_3.png)

## Kernel-I

![](image/10_kernel_1.png)

![](image/10_kernel_2.png)
![](image/10_kernel_3.png)

![](image/10_kernel_4.png)
![](image/10_kernel_5.png)

## Kernel-II

![](image/10_kernel_2_1.png)
![](image/10_kernel_2_2.png)
![](image/10_kernel_2_3.png)
![](image/10_kernel_2_4.png)

## Use SVM
![](image/10_use_svm.png)
![](image/10_use_svm_2.png)
![](image/10_use_svm_3.png)
![](image/10_use_svm_4.png)




<h1 id="08"> 08. clustering </h1>


- unsupervised learning
- cost funtion is non-convex, the result sensitive to the initializing point and K
- choosing parameter
    - random initialization and choose the model with minimum objective value
    - K:
        - K vs objective value  
        - based on the performance of downstream purpose
- EM ideology
    - E: estimate new centroid (hidden variable, random initialization in the beginning)
    - M: maximization, <=> minimize Objective function 

![](image/11_objective_function.png)

![](image/11_random_initialization.png)

![](image/11_choosing_k.png)

![](image/11_choosing_k_2.png)



<h1 id="09"> 09. dimensionality reduction </h1>


- PCA:Definition
    - reduce from n-dimension to k-dimension: find k vectors onto which to project the data, so as to minimize the project error.  
- Motivation
    -  compression 
        - reduce memory / disk needed to store data
        - speed up learning algorithm  
    -  visualization
    -  Bad use of PCA:  
        -  address overffitting:
        - may work, but!!! Note, it is recommend to address overfitting with regularization. 
- Algorithm:
     - PCA: 
         - [U,S,V] = svd(Sigma), U_[n*n]
         - U\_reduce = U(1,2..,k), 1<=k<=n, n feature, z = U\_reduce' * x , z\_[k*1]
     - Reconstruction: x = U_reduce * z
- advice
     - choose K, appropriate variance should be retained
     - for supervised learning: PCA not only use on training data but also on validatoin and test data
     - how about performance with PCA or not. PCA is sometimes used where it shouldn't be 
     - scale features to have comparable range of values
     - Note: PCA != linear regression

## definition

![](image/12_pca.png)

![](image/12_pca_2.png)

![](image/12_pca_5.png)

![](image/12_pca_6.png)
     
## choosing k

![](image/12_choosing_k_1.png)

![](image/12_choosing_k_2.png)

![](image/12_choosing_k_3.png)

## advice 

![](image/12_advice_1.png)

![](image/12_advice_2.png)

![](image/12_advice_4.png)

![](image/12_advice_5.png)



<h1 id="10"> 10. anomaly detection </h1>


- density estimation
- guassian distribution
    - one variable
    - multi-variable
- parameter estimation
    - mean
    - standard
- prediction
    - calculate density for example x, p(x)
    - p(x) large for normal examples
    - p(x) small for anomalous examples  
- evaluation
    - charateristic: 
        - very small number of postive examples
        - larege number of negative
        - many different 'types' of anomalies
        - future anomalies may look nothing like any anomalous examples we've seen so far
    - true positive, false positive, true negative, false negative
    - precision / recall
    - f-score 
- choosing feature
    - non-guassian feature, (log transformation)
    - error analysis 
- application:
    - fraud detection
    - manufacturing monitor
    
## problem motivation

![](image/13_problem_motivation_1.png)
![](image/13_problem_motivation_2.png)
![](image/13_problem_motivation_3.png)


## algorithm

![](image/13_algorithm_1.png)
![](image/13_algorithm_2.png)

## multivariate guassian distribution for anormaly detection

![](image/13_multivariate_guassian_distribution_adnormaly_detection_1.png)
![](image/13_multivariate_guassian_distribution_adnormaly_detection_2.png)
![](image/13_multivariate_guassian_distribution_adnormaly_detection_3.png)
![](image/13_multivariate_guassian_distribution_adnormaly_detection_4.png)

## evaluation

![](image/13_evaluation_1.png)
![](image/13_evaluation_2.png)
![](image/13_evaluation_3.png)


## vs supervised learning

![](image/13_vs_supervised_learning_1.png)
![](image/13_vs_supervised_learning_2.png)

## choosing feature 

![](image/13_choosing_feature_1.png)
![](image/13_choosing_feature_2.png)
![](image/13_choosing_feature_3.png)

## singular gaussian distribution

![](image/13_gaussian distribution_1.png)
![](image/13_gaussian distribution_2.png)
![](image/13_gaussian distribution_3.png)

## multivariate guassian distribution

![](image/13_multivariate_guassian_distribution_1.png)
![](image/13_multivariate_guassian_distribution_2.png)
![](image/13_multivariate_guassian_distribution_3.png)
![](image/13_multivariate_guassian_distribution_4.png)
![](image/13_multivariate_guassian_distribution_5.png)
![](image/13_multivariate_guassian_distribution_6.png)
![](image/13_multivariate_guassian_distribution_7.png)
![](image/13_multivariate_guassian_distribution_8.png)



<h1 id="11"> 11. recommender systems </h1>

- data structure (two dimention array)
    - row : user
    - column : items (eg. movies)
    - r[i,j] : user_j rating for items_i

- contented based recomendation
    - features are some characters on item 
    - row[i], item score vector (knew)
    - colunm[j], user parameter on each feature (learn from rating data)
    - rating = row[i] * colunm[j], user j rating item i (predict unrating element)

- collaborative filtering
    -  row[i], item score vector (learn)(x)
    -  colunm[j], user parameter on each feature (learn)(theta)
    -  x->theta->x->theta...

- similarity based on CF
    - user similarity, calculate distence between colum(i) and colum(j)
    - item similarity, calculate distence between x(i) and x(j) 

- implementation
    - mean normalization (there are big differences on absolute score among different user)

## problem formulation

![](image/14_problem_formulation.png)


## content based recomendation

![](image/14_content_based_1.png)
![](image/14_content_based_2.png)
![](image/14_content_based_3.png)
![](image/14_content_based_4.png)

## collaborative filtering

![](image/14_collaborative_filtering_1.png)
![](image/14_collaborative_filtering_2.png)
![](image/14_collaborative_filtering_3.png)
![](image/14_collaborative_filtering_4.png)
![](image/14_collaborative_filtering_5.png)
![](image/14_collaborative_filtering_6.png)

## vectorization

![](image/14_cf_vectorization_1.png)
![](image/14_cf_vectorization_2.png)
![](image/14_cf_vectorization_3.png)


## implementation detial

![](image/14_implementation_detial_1.png)
![](image/14_implementation_detial_2.png)



<h1 id="12"> 12. large scale machine learning </h1>


- "It's not who has the best algorithm that win. It's who has the most data"
- gradient descent
    - batch GD : use all examples in each iteration
        - time expensive
    - stochastic GD (SGD) : use 1 example in each iteration
        - time efficent 
        - convergence, learning rate can be slowly decrease over time
    - mini-batch GD : use b example (a small subset) in each iteration
        - balance choice 
    - checking for convergence
        - error vs number of iteration
        - Note: for SGD, error should be a mean error during a time section
- online learning
    - shipping service website
    - product search
        - click throgh rate 
    - choosing special offers to show user
    - customized selection of news articles
    - product recommendation
- large scale machine learning
    - mini-bach
    - map: divide training data and learn each part on parallel computer
    - reduce: combine all error for model update

## the role of data

![](image/15_large_data_win_1.png)
![](image/15_large_data_win_2.png)

## stochastic gradient descent (SGD)

![](image/15_sgd_convergence_3.png)
![](image/15_sgd_convergence_4.png)

![](image/15_sgd_convergence_1.png)
![](image/15_sgd_convergence_2.png)

## mini-batch gradient descent

![](image/15_mini-batch_gd_1.png)
![](image/15_mini-batch_gd_2.png)

## online learning

![](image/15_online_learning_1.png)
![](image/15_online_learning_2.png)

## map-reduce

![](image/15_map_reduce_1.png)
![](image/15_map_reduce_2.png)
![](image/15_map_reduce_3.png)
![](image/15_map_reduce_4.png)





<h1 id="13"> 13. application photo OCR </h1>


- set up a pipline for your project
    - for instance OCR: text detection -> character segmentation -> charater classification 
- segmentation : sliding window
    - win size
    - classfication problem for each segmentation
- get more data
    - artifical
    - sythesis
    - Note:
        - make sure you have a low bias classification
        - "how much work would it be to get 10x as much data as we currently have?"
           - artificial data synthesis
           - collect / label it yourself
           - "crowd source"
- ceiling analysis   

## pipline

![](image/16_pipline_1.png)

![](image/16_pipline_2.png)

## sliding window

![](image/16_sliding_window_1.png)

![](image/16_sliding_window_2.png)

![](image/16_sliding_window_3.png)
![](image/16_sliding_window_4.png)

![](image/16_sliding_window_5.png)

![](image/16_sliding_window_6.png)
![](image/16_sliding_window_7.png)
![](image/16_sliding_window_8.png)
![](image/16_sliding_window_9.png)

## get more data

![](image/16_get_more_data_1.png)
![](image/16_get_more_data_2.png)
![](image/16_get_more_data_3.png)
![](image/16_get_more_data_4.png)
![](image/16_get_more_data_5.png)
![](image/16_get_more_data_6.png)
![](image/16_get_more_data_7.png)

## ceiling analysis

![](image/16_ceiling_analysis_1.png)

![](image/16_ceiling_analysis_2.png)
![](image/16_ceiling_analysis_3.png)


