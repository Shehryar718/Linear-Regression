## Linear Regression Model

LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
to minimize the residual sum of squares between the observed targets in
the dataset, and the targets predicted by the linear approximation.

---

### Parameters

**epochs :** 

    int, default=100
    Number of iterations to perform.

**learning_rate :**

    float, default=0.001
    Rate of learning/updating thetas.

**threshold :**

    float, default=0.0001
    Epochs breaker after a certain value has been reached of recent thetas.

**alpha :**
    
    int, default=0
    Value that assigns weightage to thetas in case of Lasso or Ridge.

**mode :**
    
    string, default=None
    Selection of Ridge or Lasso model.

    Values:
    
        'Lasso'/'L1' : The Lasso is a linear model that estimates
            sparse coefficients with l1 regularization.

        'Ridge'/'L2' : Ridge regression addresses some of the
            problems of Ordinary Least Squares by imposing a penalty on the
            size of the coefficients with l2 regularization.

**show_epochs :**

    bool, default=False
    Displays progress with each epoch by showing loss at each iteration.

**scale_features :**
    
    bool, default=False
    Scales features in range of [0, 1] using min max.

**validate :**
    
    bool, default=False
    Splits 0.3 of the training data into validation data.
        
---

### Attributes

**coef_ :**
    
    array of shape (n_features, ) or (n_targets, n_features)
    Estimated coefficients for the linear regression problem.
    If multiple targets are passed during the fit (y 2D), this
    is a 2D array of shape (n_targets, n_features), while if only
    one target is passed, this is a 1D array of length n_features.

---

### Authors: 
    Shehryar Sohail and Abdulwadood.
