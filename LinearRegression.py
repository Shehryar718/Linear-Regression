import numpy as np

class LinearRegression():
    '''
    Linear Regression model : for regression problems.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    epochs : int, default=100
        Number of iterations to perform.
        
    learning_rate : float, default=0.001
        Rate of learning/updating thetas.
        
    threshold : float, default=0.0001
        Epochs breaker after a certain value has been reached of recent thetas.
        
    alpha : int, default=0
        Value that assigns weightage to thetas in case of Lasso or Ridge.
        
    mode : string, default=None
        Selection of Ridge or Lasso model.
        
        Values:
        
            'Lasso'/'L1' : The Lasso is a linear model that estimates
                sparse coefficients with l1 regularization.
                
            'Ridge'/'L2' : Ridge regression addresses some of the
                problems of Ordinary Least Squares by imposing a penalty on the
                size of the coefficients with l2 regularization.

    show_epochs : bool, default=False
        Displays progress with each epoch by showing loss at each iteration.
        
    scale_features : bool, default=False
        Scales features in range of [0, 1] using min max.
        
    validate : bool, default=False
        Splits 0.3 of the training data into validation data.
        
    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    -------
    Authors: Shehryar Sohail and Abdulwadood.
    '''
    def __init__(self,
                 epochs=100,
                 learning_rate=0.001,
                 threshold = 0.0001,
                 alpha = 0,
                 mode = None,
                 show_epochs = False,
                 scale_features = False,
                 validate=False):
        
        if mode not in ['Lasso', 'Ridge', 'L1', 'L2', None]:
            raise Exception(f'\'{mode}\' not found! Must be \'Lasso\'/\'L1\', \'Ridge\'/\'L2\', or \'None\'.')
            
        self._thetas = None
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._threshold = threshold
        self._alpha = alpha
        self._mode = mode
        self._show_epochs = show_epochs
        self._scale = scale_features
        self._validate = validate
        self._losses = []

    def _hypothesis(self, X, thetas):
        '''
        Dot product of X and thetas
        '''
        return np.dot(X, thetas)

    def _loss_function(self, y_act, y_pred):
        '''
        Mean squared error based loss function.
        '''
        loss = np.mean((y_act-y_pred)**2)
        
        if self._mode in ['Lasso', 'L1']:
            return loss + self._alpha*np.sum(abs(self._thetas))
        
        if self._mode in ['Ridge', 'L2']:
            return loss + self._alpha*np.sum(self._thetas**2)
            
        return loss

    def _gradient_descent(self, learning_rate, thetas, g_thetas):
        '''
        Update thetas.
        '''
        return (thetas-learning_rate*(g_thetas))

    def _der_loss_fun(self, X, y, thetas):
        '''
        Calculate partial derivative with respect to thetas.
        '''
        hypothesis = self._hypothesis(X,thetas)
        PD_wrt_thetas = np.dot((y-hypothesis).T, X)
        PD_wrt_thetas = -2*PD_wrt_thetas/float(X.shape[0])
        
        if self._mode in ['Lasso', 'L1']:
            return PD_wrt_thetas + self._alpha
        
        if self._mode in ['Ridge', 'L2']:
            return PD_wrt_thetas + self._alpha*self._thetas
        
        return PD_wrt_thetas
    
    def _normalise(self, X):
        '''
        Min Max normalisation.
        '''
        return (X - np.min(X))/np.ptp(X)
    
    def _traintestsplit(self, X, y, ratio=0.3):
        '''
        Splits 0.3 of the training data into validation data.
        '''
        indexes = np.arange(X.shape[0])
        np.random.shuffle(indexes)
        X = X[indexes]
        y = y[indexes]
        return X[:int(0.3*len(X))], X[int(0.3*len(X)):], y[:int(0.3*len(X))], y[int(0.3*len(X)):]
            
    def fit(self, X, y, thetas=None):
        '''
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        thetas : array-like of shape (n_features + 1,), default=None
            Pre-defined thetas.

        Returns
        -------
        self : object
        '''
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        if thetas is None:
            self._thetas = np.ones(X.shape[1])
        else:
            if thetas.shape[0] != X.shape[1]:
                raise Exception(f'Thetas shape expected: {X.shape[1]}, but found: {thetas.shape[0]}')
                
            self._thetas = thetas
        
        if self._scale:
            X = self._normalise(X)
        
        if self._validate:
            X, X_val, y, y_val = self._traintestsplit(X, y)
            
        prev = []
        
        for i in range(self._epochs):
            partialDerivate = self._der_loss_fun(X, y, self._thetas)
            self._thetas = self._gradient_descent(self._learning_rate, self._thetas, partialDerivate)
            loss = self._loss_function(y, self._hypothesis(X, self._thetas))
            
            if self._show_epochs:
                print('Epoch:', i+1, '='*(10 - len(str(i+1))), '\b> Loss:', loss, end='\t\t')
                if self._validate:
                    v_loss = self._loss_function(y_val, self._hypothesis(X_val, self._thetas))
                    print('\b Validation Loss:', v_loss)
                else:
                    print()
                    
            prev.append(loss)
            
            if i > 10 and np.var(prev[-10:]) < self._threshold:
                prev = prev[-10:]
                if self._show_epochs:
                    print('\nThreshold reached!\n')
                break

        print(f'Loss: {loss}', end='\t\t')
        if self._validate:
            v_loss = self._loss_function(y_val, self._hypothesis(X_val, self._thetas))
            print('\b Validation Loss:', v_loss)
                
    def predict(self, X):
        '''
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        '''
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        if self._scale:
            X = self._normalise(X)
            
        return self._hypothesis(X, self._thetas)
    
    def coef_(self):
        '''
        Returns the array of thetas.

        Returns
        -------
        thetas : array, shape (n_features + 1,)
            Returns thetas.
        '''
        return self._thetas
