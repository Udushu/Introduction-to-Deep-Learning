# 1. Introduction to Deep Learning
## 1.1 Optimization
### 1.1.1 Preliminaries
#### 1.1.1.1 Supervised Learning

Adopt the following notation:

* $x_i$ - training example, may be an image or an object to be analyzed

* $y_i$ - target value or label, the ground truth that should come out of the analysis

* $(x_{i1},\dots, x_{ip})$ - features vector of the training example x_i that characterize the example

* $X = \left(x_1,\dots, x_n\right)$ - training set built from all available training examples and their respective labels.

* $h_w(x)$ - model or hypothesis function parameterized by $w$ that maps the training examples $x_i$ onto the target/response values/labels $y_i$.

The goal of the Supervised Learning is to use the known ground truth values\labels $y_i$ of data points $x_i$ to create a model that is capable of making predictions:

$x_i \rightarrow h_w(x_i) \rightarrow y_i$

Two main types of the Supervised Learning are _Regression_ and _Classification_:

* In _Regression_ tasks $y_i \in \mathbb{R}$, that is the target is a real value and the typical tasks would be predicting salary or movie rating.

* In _Classification_ tasks $y_i \in S=\left\{S_1,S_2, \dots, S_k\right\}$ where $k$ is the number of possible values in the target value $y_i$ may take; a typical example would be pattern/object recognition or topic classification.

#### 1.1.1.2 Linear Models
Linear Models are a fundamental building block upon which more sophisticated models such as Neural Networks are founded. For the Linear Models, given the random sample $(y_i, x_{i1}, \dots, x_{ip}), i=1,\dots,n$ the relationship between the observation $y_i$ and independent variables $x_{ij}$ is formulated as follows:

$y_i=\beta_0 + \beta_1 \times \phi_1(x_{i1})+\dots+\beta_p \times \phi_p(x_{ip}) + \epsilon_i \quad i=1,\dots, n$

where $\phi_1,\dots, \phi_p$ may be non-linear functions, the quantities $\epsilon_i$ are random variables representing errors in the relationship. The "linear" part of the designation relates to the appearance of the regression coefficients $\beta_j$ in a linear way in the above relationship.

##### Linear Model for Regression
Consider a simple dataset where each object/example is described by a single feature and the target is the real valued:

![linear model](images\1_1_1_2-1.png =300x)

Figure above shows an example of such dataset. A clear linear trend is visible: if the value of the feature increases two times, the value of the response will increase two times as well. This suggests that a linear model is an appropriate approximation of the data and can be used to describe the relationship between the feature variable $x$ and the response variable $y$. Such model would require only two parameters: intercept term $w_0$ and slope $w_1$. Figure illustrates the best fit that such model can produce given the data.

In most Machine Learning tasks there are many features, so a generic _Linear Regression_ model has the following form:

$h_w(x) = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p$

where:
  - $w_0$ is the intercept term also known as the bias parameter of the model.

  - $w_1,\dots, w_p$ are the coefficients\weights\parameters of the model.

a Linear Regression model has $p+1$ parameters for $p$ features present in the input training example vector $x$.

When creating a Linear Regression model, it is often assumed that the training examples have artificial or augmented feature that is always equal to one, that is $x=(1,x_1, \dots, x_p)$. This extra feature accounts for the bias term so that it does not have to be added to the model explicitly and analyzed separately. Thus the weight vector $w=(w_0, w_1, \dots, w_p) \in \mathbb{R}^{p+1}$ where $w_0$ is the bias term. The feature vector with the artificial "1" added to it is referred to as the _augmented feature vector_.

With the augmented feature vector $x$, it is very convenient to write down the linear model in the matrix/vector multiplication form:

$h_w(x_i) = w^Tx_i$

So the linear model is basically a dot product of the weight vector and the feature vector. For the entire training set the model may be written as:

$h_w(x_i) = Xw$

where $X=\begin{bmatrix}
   x_{10} & \dots & x_{1d}  \\
   \vdots & \ddots & \vdots  \\
   x_{n0} & \dots & x_{nd}
\end{bmatrix}$

##### Regression Loss Function
In statistics a _Loss Function_ or _Cost Function_ is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function. An objective function is either a loss function or its negative in which case it is to be maximized. In statistics, typically a loss function is used for parameter estimation, and the event in question is some function of the difference between estimated and true values for an instance of data.

One of the most popular Loss Functions used for Regression problems is the _Mean Squared Error (MSE)_ loss:

$L(w)= \frac{1}{n} {\displaystyle\sum_{i=1}^{n}{(w^T x_i - y_i)^2}}= \frac{1}{n} {\lVert Xw-y \rVert}^2$

A prediction model $\hat{y}_i= w^T x_i$ is calculated for observation $x_i$ and the true target value $y_i$ is subtracted from the prediction $\hat{y}_i$. The deviation of the prediction from the target is calculated. The square of the deviation is then averaged across all examples that are used in the calculation. The less is the MSE value, the better the model fits the data.

##### Training a Model
Once the Loss Function of a model that measures how well the model fits the data is defined, the model can be trained by minimizing the value of the Loss Function with respect to the values of the parameter vector $w$; that is to find the parameter set $w$ that gives the lowest value of the loss.

$\underset{w}{\text{argmin}} \left[ \frac{1}{n} {\lVert Xw-y \rVert}^2 \right]$

For the Linear Regression model with the MSE loss, the stated optimization problem has an exact solution that is available in closed form:

$w = (X^T X)^{-1} X^T y$

The exact closed form solution, however, requires to invert matrix $(X^T X) \in \mathbb{R}^{p \times p}$, an operation that for large values of $p$, i.e. high-dimensional data, can be prohibitively expensive. Thus for many practical linear regression problem the search for optimal parameter vector $w$ is posed as an optimization problem rather than the matrix inversion/multiplication problem.

##### Linear Model for Classification
Linear regression method may be modified to solve the classification problems, in which the target variable $y$ may take only a finite set of discrete values.

###### Binary Classification
Binary or binomial classification is the task of classifying the elements of a given data set into two groups (predicting from which of the groups the observations may have originated from) on the bases of the rule / hypothesis $h_w(x)$.

Typically in binary classification tasks $y \in {-1,1}$, representing the negative class and the positive class of the outcomes. Such problem requires to essentially calculate the dot product of the weight vector $w$ and feature vector $x$. The real-valued dot product must then be transformed to either $-1$ or $1$, easily achieved by taking the sign of the dot product:

$h_w(x) = \text{sign}(w^T x)$

Such model for $x_i \in \mathbb{R}^d$ has $d+1$ parameters with $w \in \mathbb{R}^{d+1}$ with the bias/intercept term included into the parameter vector.

Geometrically, in two dimensions with feature vector $x_i = (x_i^1, x_i^2)$, the model has the following look:

![classification model](images\1_1_1_2-2.png =350x)

If the model has only two features $x_i^1$, and $x_i^2$, such that $x_i \in \mathbb{R}^2$, and the red points denote the negative class (i.e. $y_i = -1$), blue points denote the positive class (i.e. $y_i = -1$), then the linear classification model would try to find some line that separates the blue points from the red points. The sign of the dot product indicates on which side of the line a point lies: a positive dot product indicates that the point lies on the positive side of the line, and vise versa.

###### Multi-Class Classification
The binary classification problem can be extended to to situations when the number of classes, or discrete values that the target value $y_i$ can take is more than two: $y_i \in S=\left\{1,2, \dots, K\right\}$.

One of the most popular approaches to the Multi-Class Classification problem is to build a separate classifier for each class, so each class has a dedicated model that separates the points of that class from all other points; so that the points of each class are on the positive side of the associated decision hyperplane, and points of all other classes are on the negative side of that hyperplane. The dot product of such model is essentially a score: the further away the point is from the associated decision hyperplane on the positive side, the higher is the value of the dot product and therefore the model is more confident in its decision that the point belongs to the class.

Each of the $K$ linear models calculates a score, the new example is assigned to the class with the highest positive score, e.g. if the scores are $z=(7,-7.5,10)$, then the outcome is assigned to the third class and $h_w(x)=3$:

$h_w(x) = \underset{k \in \left\{1, \dots, K \right\}}{\text{max}} \left( w_k^T x \right)$

The number of parameters for such model with $x_i \in \mathbb{R}^p$ is $K \times (p+1)$.

###### Classification Loss Function
Classification performance is often expressed in terms of classification accuracy, i.e. what is the fraction of the examples that the model got right:

$\text{acc} = \frac{1}{n} \displaystyle\sum_{i=1}^{n}{ \left[ h_w(x_i) = y_i \right] }$

Note that $[\bullet]$ is the inversion bracket, they contain a logic expression such that if the expression is True, then the value of the expression is $1$,  and is $0$ otherwise:

$\left[P\right] = \begin{cases}
   1 &\text{if P is True} \\
   0 &\text{if P is False}
\end{cases}$

Accuracy measure is easily interpretable, but it has two large disadvantages:

* The accuracy is not differentiable (gradient is needed to optimize the loss function efficiently, and accuracy does not have gradients with respect to the models parameters).

* The accuracy does not asses model's confidence, the accuracy saturates at the value of 1 and further increase in the model's performance will not be reflected in the accuracy.

Another potential candidate for the classification loss function is the MSE. For a training example $x_i$ from a positive class such that $y_i=1$ the loss becomes $\left( w^T x_i - 1 \right)^2$, so if $\hat{y}_i=1$, then the guess is correct and the loss is zero; conversely if the prediction is less than $1$, then the model is unconfident in it's decision and as such is penalized by growing loss function. If the mode outputs incorrect value $\hat{y}_i<0$, than the model is penalized even further.

However, MSE loss would penalize model for high confidence predictions as well as the low confidence predictions (with $\hat{y}_i > 1$). Figure below shows the value of the loss function as a function of the model outputs:

![mse loss](images\1_1_1_2-3.png =300x)

This is a highly undesirable behavior for the mode, since, intuitively, higher classification confidences are typically required for better generalization of the model, and small or zero loss values should be assigned to the high confidence decisions.

A better classification function is the one that only uses the less-than-one part of the squared loss:

![rectified mse loss](images\1_1_1_2-4.png =300x)

Such loss function would result in a penalty for incorrect results and no penalty for correct classifications. There are many loss function like this one and they all lead to their own classification methods.

Arguably the most appropriate loss function for the linear classification model is the _Logistic Loss Function_. The class scores (_logits_), outputs of the linear classifier, can be converted to probabilities. Let $z$ be a vector of scores:

$z=(w_1^Tx, w_1^Tx, \dots, w_k^Tx)$

Vector dot products can have any sign and magnitude, so they cannot be interpreted as probabilities. The, however, can be doe by taking the exponents of the scores and then renormalizing the scores so that the sum of scores equals to $1$.

$(w_1^Tx, w_1^Tx, \dots, w_k^Tx) \rightarrow (e^{w_1^Tx}, e^{w_1^Tx}, \dots, e^{w_k^Tx})$

The new transformed class score values has only positive values and the magnitude of these values is now proportional to the confidence of the model. The new components now need to be normalized to be interpretable as a probability distribution:

$\sigma(z) = \left( \frac{e^{z_1}}{\sum_{k=1}^{K}{e^{z_k}}}, \frac{e^{z_2}}{\sum_{k=1}^{K}{e^{z_k}}}, \dots, \frac{e^{z_K}}{\sum_{k=1}^{K}{e^{z_k}}} \right)$

The resulting vector is normalized and has only non-negative components, so it can be interpreted as a probability distribution. This is transform is known as _Softmax Transform_.

For example:

$z=(7,-7.5,10)\rightarrow (e^7, e^{-7.5}, e^{10})=(1096.6,0.0005,22026.5)\rightarrow \sigma(z)=(0.047, 0.000, 0.953)$

Now that the class scores are converted into probabilities, the loss function can be defined. The target values for the class probabilities can be written as:

$p = ([y=1], \dots, [y=K])$

Similarity between the class scores $z$ and the target probabilities $p$ can be measured using the _Cross-Entropy Loss_:

$L(w) = -\displaystyle\sum_{k=1}^{K}{ \left\{ \left[ y=k \right] \times \ln{\frac{e^{z_k}}{\sum_{j=1}^{K}{e^{z_j}}}} \right\}} = -\ln{\frac{e^{z_y}}{\sum_{j=1}^{K}{e^{z_j}}}}$

For classification tasks, the Cross-Entropy can be simply summed over all examples from the training set:

$L(w) = -\displaystyle\sum_{i=1}^{n} { \displaystyle\sum_{k=1}^{K}{ \left\{ \left[ y=k \right] \times \ln{\frac{e^{w_k^T x_i}}{\sum_{j=1}^{K}{e^{w_j^T x_i}}}} \right\}} } = -\displaystyle\sum_{i=1}^{n} { \ln{\frac{e^{w_{y_i}^T x_i}}{\sum_{j=1}^{K}{e^{w_{j}^T x_i}}}} }$

The resulting loss function can now be minimized with respect to the parameter vector $w$ to fit the optimal linear classification model.

#### 1.1.2 Gradient Descent
_Gradient Descent (GD)_ is a generic method that can minimize any differentiable loss function. GD solves the following optimization problem:

$\underset{w}{\text{argmin}} \ L(w)$

Suppose that an initial approximation of $w$ denoted $w^0$ is an arbitrary point on the surface defined by the loss function $L(w)$ is available. It now needs to be refined, such that the value of $L(w)$ is minimized. The surface of the loss function can be very complicated: multiple optima and saddle points may be present, as shown on the figure below:

![loss surface](images\1_1_2-1.png =300x)

A direction moving in which from point $w^0$ leads to a decrease of the function $L(w)$ value is required. From the basic calculus, the gradient vector of the loss function with respect to the parameter vector $w$ points to the direction of the steepest ascent of the loss function. Therefore, the negative of the gradient vector will point to the direction of the steepest descent:

$\nabla L(w) = \left( \frac{\partial L(w)}{\partial w_0}, \dots, \frac{\partial L(w)}{\partial w_p} \right)$

So in order to reduce the value of the loss function $L(w)$ one should calculate the gradient vector of the loss function and move in the direction of opposite to the one in which the gradient vector points. Given an initial guess values of the parameter vector $w$ set at $w^0$, the next value $w^1% can be calculated as:

$w^1 \leftarrow w^0 - \eta_1 \nabla L(w^0)$

The process is then repeated until the convergence criterion is achieved

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad w^t \leftarrow w^{t-1} - \eta_t \nabla L(w^{t-1})\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

where $\epsilon$ is some convergence criterion selected a priory.

The path of the optimization algorithm along the surface of a simple loss function is shown on the figure:

![loss path](images\1_1_2-2.png =250x)

There are many heuristics that are used any time the GD is applied:

* How to initialize the parameter vector $w$ and select the value $w^0$; this can be done either randomly or using some heuristic.

* How to select the step size $\eta_t$; if the step size is too small then the convergence will be unnecessarily slow, and a large value of the step size may result in a divergence.

* How and when to stop the optimization process. Once approach is to monitor the change in the distance between the parameter vectors from two consecutive iterations and stop when the step change becomes smaller than some criterion $\epsilon$, however, there are many other options. For example, the difference between the values of the loss function can be monitored as well in a similar fashion.

* How to approximate gradient $\nabla L(w^{t-1})$; calculating the gradient of the loss function and summing the losses for each example from the training set can be computationally challenging for large datasets; thus the gradients are typically approximated.

Gradient for the MSE Loss Function can be expressed as:

$L(w) = \frac{1}{n} {\lVert Xw-y \rVert}^2 \rightarrow \nabla L_w(w) = \frac{2}{n} X^T (Xw-y)$

GD gives a general solution that can be applied to any differentiable loss function and typically requires reasonable amount of memory and computations for its deterministic variants.

#### 1.1.3 Regularization
In mathematics, statistics, and computer science, particularly in machine learning and inverse problems, regularization is the process of adding information in order to solve an ill-posed problem or to prevent overfitting. Regularization applies to objective functions in ill-posed optimization problems.

##### 1.1.3.1 Overfitting Problem and Model Validation
A model needs to be validated to check how well they perform not only on the training data, but also how well do they generalize to the new data.

Suppose that a classifier was trained and achieved 80% accuracy on the training data. There is no guarantee that the model will work well on new data. There is a possibility that the model overfits the data: the model has simply remembered the training examples and is not capable to generalize at all.

Consider an example with $X \in \mathbb{R}$ with the following linear regression model: $h_w(x)=w_0 + w_1x$:

![underfitting](images\1_1_3_1-1.png =300x)

On the figure, the green line is the true target function, and the blue line is the best prediction by the proposed linear regression model. This is an example of the model _underfitting_ the data, meaning that the model is too simple for the data and thus can not capture the complexity of the dataset, since the dependency between $x$ and $y$ is not linear.

This problem of underfitting can be fixed by proposing and using a polynomial model: $h_w(x)=w_0 + w_1x + w_2x^2 + w_3x^3 + w_4x^4$:

![just right](images\1_1_3_1-2.png =300x)

This model adds artificial features to the training examples such as powers of $x$ up to the 4th degree. This model fits the data very well and provides good predictive capabilities, since it captures the target function.

The number of artificial feature and therefore the parameters of the model may be increased to an arbitrary degree: $h_w(x)= w_0 + w_1x + w_2x^2 + \dots + w_{15}x^{15}$:

![overfitting](images\1_1_3_1-3.png =300x)


This model fits the training data nearly perfectly by placing the model function almost exactly through each training data point. However, the resulting function is too complex for the training data, so it _overfits_ the data, and will have poor performance on the new examples, that is this model will not generalize to the new data.

Another example of overfitting would be memorization of the dataset by the model. Suppose that the dataset consists of 8 points: $x=\left\{ 0.2, 0.4, \dots, 1.6 \right\} \in \mathbb{R}$ and the target function is $y=sin(x)+\epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and $\sigma^2$ is some small number. Let the model be $h_w(x)= w_0 + w_1x + w_2x^2 + \dots + w_{8}x^{8}$. After the model was fitted into the data, the parameter vector $w=(130.0, −525.8, \dots, 102.6)$  was found:

![overfitting](images\1_1_3_1-4.png =300x)

he model simply memorized the training examples exactly and will give perfect predictions for the training set. This model however will have no ability to generalize to the new data, that is for any point not from the data set, the quality of predictions will be very poor. Note that all values of the data points are within $[−1, 1]$  range, but the parameter values are relatively large (in hundreds).

###### Holdout Validation Test
Ability of models to underfit or overfit the training data leads to requirement to validate the models and check whether they are overfitted or not. This can be done by taking all training examples and splitting them into two pars:

![holdout](images\1_1_3_1-5.png =400x)


The model is trained using the _Training set_ data and then tested with the _Holdout set_. If the loss is high on both the holdout set and the training set, then the model has high bias and is underfitting, and if the loss is high on the holdout set, but is low on the training set, then the model has high variance and is overfitting the data.

The size of the Holdout set is chosen based on the following considerations:

* Small holdout set:
  * The training set is representative of the entire data set.
  * The holdout set quality has high variance and may be not representative.

* Large holdout set:
  * Holdout set quality has low variance.
  * Holdout set quality has high bias
  * The training set may be no longer representative of the entire data set.

Typical training/holdout breakdown is 70% / 30%, but there is a better approach.

###### $k$-Fold Cross-Validation Test
In _$k$-Fold Cross-Validation_, the original sample is randomly partitioned into $k$ equal size subsamples. Of the $k$ subsamples, a single subsample is retained as the validation data for testing the model, and the remaining $k-1$ subsamples are used as training data. The cross-validation process is then repeated $k$ times (the folds), with each of the $k$ subsamples used exactly once as the validation data. The $k$ results from the folds can then be averaged (or otherwise combined) to produce a single estimation.

![k-fold cross-validation](images\1_1_3_1-6.png =400x)

The advantage of this method is that all observations are used for both training and validation, and each observation is used for validation exactly once.

For classification problems, one typically uses _stratified $k$-fold cross-validation_, in which the folds are selected so that each fold contains roughly the same proportions of class labels.

##### 1.1.3.2 Model Regularization
One of the major aspects in training machine learning models is to avoid overfitting. Going back to the example with just 8 data points $x=\left\{ 0.2, 0.4, \dots, 1.6 \right\} \in \mathbb{R}$ and the target function is $y=sin(x)+\epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and $\sigma^2$ is some small number, a simpler model such as $h_w(x)=w_0 + w_1x + w_2x^2 + w_3x^3$ can be used to achieve much better performance and a better parameter vector $w=(0.634, 0.918, -0.626)$:

![better model](images\1_1_3_2-1.png =300x)

This model has only three features, and its complexity matches the complexity of the data. Observe that the model's weights are not very large, this an indicator of a well fitting model. A model that overfits the data will typically have weights that are very large. This observation can be used to solve the problem of complex models overfitting the data:

Modify the loss function as follows:

$L_{reg}(w) = L(w) + \lambda R(w)$

where:

* $L(w)$ is the original loss function of the model.
* $R(w)$ is the regularizer (or penalty) function (a function that penalizes the loss function for having large weights).
* $\lambda$ is the regularization strength, a hyperparameter that controls the model's performance on the training set and the model's ultimate complexity.

Now the new, modified, loss function $L_{reg}(w)$ is minimized with respect to the parameter vector $w$.

###### L2 Penalty
One of the most commonly used regularizers is the _L2 Penalty_. This regularizer drives all weight closer to zero without explicitly reaching it.

$R(w) = {\lVert w \rVert}^2 = \displaystyle\sum_{j=1}^{p}{w_j^2}$

**Note**: L2 regularization does not include the bias term!

It can be shown that L2 Regularization effectively converts the unconstrained optimization problem of GD into a constrained optimization problem:

$\begin{cases}
   \underset{w}{\text{argmin}} \ L(w) \\
   \text{subject to:} \ {\lVert w \rVert}^2 \le C
\end{cases}$

![l2 constrained optimization](images\1_1_3_2-2.png =300x)

There is a one-to-one correspondence with the constraint $C$ and the regularization parameter $\lambda$. This forces the optimizer to select the point that has the minimum loss within the ball of radius $C$ and centered at zero.

Going back the example problem with 8 data points, if parameterized by 8th order polynomial hypothesis function $h_w(x)= w_0 + w_1x + w_2x^2 + \dots + w_{8}x^{8}$ and optimized under the L2 Penalty loss function, the model yields the parameter vector $w=(0.166, 0.168, 0.130, 0.075, 0.014, −0.040, −0.050, 0.018)$. Compare to the not regularized parameter vector for the same model: $w_{\text{original}}=(130.0, −525.8, \dots, 102.6)$ that has values in hundreds. The model provides the following fit:

![l2 constrained model](images\1_1_3_2-3.png =300x)

Note that the weights are small and the fit is similar to the one produced by a less complex model.

###### L1 Penalty
Another commonly used regularizer is the _L1 Penalty_. This regularizer drives some of the weights exactly to zero, and tends to learn sparse models as well as select only the most important features:

$R(w) = {\lVert w \rVert}^1 = \displaystyle\sum_{j=1}^{p}{|w_j|}$

**Note**: L1 regularization does not include the bias term!

L1 penalty is not differentiable (due to the absolute value operation in the regularizer function) and therefore cannot be optimized with simple gradient methods, and requires more advanced optimization techniques.

It can be shown that L1 Regularization effectively converts the unconstrained optimization problem of GD into a constrained optimization problem:

$\begin{cases}
   \underset{w}{\text{argmin}} \ L(w) \\
   \text{subject to:} \ |w| \le C
\end{cases}$

![l1 constrained optimization](images\1_1_3_2-4.png =300x)

Going back the example problem with 8 data points, if parameterized by 8th order polynomial hypothesis function $h_w(x)= w_0 + w_1x + w_2x^2 + \dots + w_{8}x^{8}$ and optimized under the L2 Penalty loss function, the model yields the parameter vector  $w=(0.78, 0.03, \mathbf{0.00}, \mathbf{0.00}, \mathbf{0.00}, −0.16, −0.01, \mathbf{0.00})$  with the following data fit:

![l1 constrained model](images\1_1_3_2-5.png =300x)

Note that some of the weights are exactly zero and others are small, and the fit is similar to the one produced by a less complex model.

###### Other Regularization Techniques
Other commonly used methods for regularization are:

* _Dimensionality Reduction_ by removing some redundant features or by deriving some new features from existing features (e.g. with the Principal Component Analysis) that replace the existing features.

* _Data Augmentation_ through generation of new artificial training examples by transforming existing examples so it is harder for the model to memorize the training set.

* _Dropout_ is the technique of rand disabling some of the neurons in a neural network. This method will be studies in details in future sections.

* _Early Stopping_ is the method that compares the performance of the model on the test set vs. performance of the model on the training set during the GD and stops training when the performance on the test set stops improving.

* _Collecting more data_ is also a way to avoid overfitting, since the more data is available, the harder it is for the model to overfit.

#### 1.1.4 Stochastic Gradient Descent
Recall that the GD attempts to minimize a loss function which is usually a sum of losses on separate examples from the entire training set:

$L(w) = \frac{1}{n} \displaystyle\sum_{i=1}^{n}{L(w|x_i, y_i)}$

Next, with the initial guess of the optimal parameter vector $w^0$:

The process is then repeated until the convergence criterion is achieved

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad w^t \leftarrow w^{t-1} - \eta_t \nabla L(w^{t-1})\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

The gradient term $\nabla L(w)$ for the simple MSE loss function is $\nabla L(w) = \frac{1}{n} \sum_{i=1}^{n}{(w^T x_i - y_i)^2}$, where $n$ gradients should be computed on each step. If the data doesn't fit into memory, it should be read from storage on every GD step, which can be costly. This makes GD infeasible for large scale problems.

To overcome this problem, _Stochastic Gradient Descent (SGD)_ may be used. It is similar to the regular GD, with one key difference: it starts at some initial location $w^0$ and the on every step $t$, it selects the random example $i$ from the training set; the gradient is then calculated only for this selected random example. The step is then made in the direction of this gradient. Thus SGD approximates the gradient of the loss function by the gradient of the loss only on one example:

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad i \leftarrow \mathcal{U}\left\{ 1,n \right\}\\
\quad w^t \leftarrow w^{t-1} - \eta_t \nabla L(w^{t-1}|x_i, y_i)\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

In the algorithm above $\mathcal{U}\left\{ 1,n \right\}$ denotes a uniform discrete distribution that starts at value $1$ and ends on value $n$ inclusively. Here and in the future algorithms expression $i \leftarrow \mathcal{U}\left\{ 1,n \right\}$ indicates drawing one random sample from the specified distribution.

This approach leads to noisy approximations, analysis of the SGD performance on some sample shows that the loss function can increase or decrease during the optimization process:

![sgd performance](images\1_1_4-1.png =300x)

However, if enough iterations of the SGD are performed, then the process will converge to some minimum.

SGD has an advantage that it can be used in an online learning setting. If the data comes from a stream, then with each new particular example, the weights of the model may be updated by making a single step along the gradient.

With SGD, learning rate $\eta_t$  must be chosen very carefully, because with the large learning rates, the SGD will not converge, and the small values of the learning rate lead to unnecessary long convergence times.

#### 1.1.5 Mini-Batch Gradient Descent
To overcome some of the limitations of the SGD, the _Mini-Batch Gradient Descent_ was introduced. In Mini-Batch GD on every iteration $m$ random examples are chosen from the Training set. The loss function gradient is approximated by the average gradient across the selected $m$ examples, instead of a single example like in SGD. The Mini-Batch GD steps toward the approximation of the gradient on each iteration:

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad i_1, \dots, i_m \leftarrow \mathcal{U}\left\{ 1,n \right\}\\
\quad w^t \leftarrow w^{t-1} - \eta_t \frac{1}{m} \displaystyle\sum_{j=1}^{m}{\nabla L(w^{t-1}|x_j, y_j)}\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

Note, that regular SGD becomes a special case of the Mini-Batch GD with $m=1$.

Depending on the batch size $m$, this approach can still be used in online learning setting: an updated is made when $m$ examples from the stream are accumulated. The updates of the Mini-Batch GD have much less noise and the variation of the gradient approximations is reduced. The learning rate $\eta_t$  should still be chosen carefully.

There is another problem with both Deterministic and Stochastic variations of the GD method; it is known from calculus that the gradient is always orthogonal to the level lines. If one starts at some point $w^0$, and then makes a step that lands on the other side of the function, and then another step that lands on the opposite side again, and so on... In such setting the GD will oscillate and it will take the GD many iterations to converge. See figure below for illustration:

![gd oscillation](images\1_1_5-1x.png =300x)

#### 1.1.6 Gradient Descent Extensions
Some advanced optimization techniques can be used to improve the GD methods. It was shown above that in some conditions GD can oscillate on difficult functions before it converges.

##### 1.1.6.1 Momentum
_Momentum_ method is an extension of the Mini-Batch GD that maintains an additional vector $h$ at every iteration of the GD and uses it in its updates of the parameter vector $w$. The algorithm is:

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad i_1, \dots, i_m \leftarrow \mathcal{U}\left\{ 1,n \right\}\\
\quad g_t \leftarrow \frac{1}{m} \displaystyle\sum_{j=1}^{m}{\nabla L(w^{t-1}|x_j, y_j)}\\
\quad h_t \leftarrow \alpha h_{t-1} + \eta_t g_t\\
\quad w^t \leftarrow w^{t-1} - h_t\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

Vector $h_t$ is essentially a weighted sum of gradients from all previous iterations, as well as the current iteration. This modification allows the Mini-Batch GD to work in conditions when in some locations of the parameter space the gradients tend to have the same sign, so they lead to the minimum, while in others they tend to revert the sign and result in oscillations. Momentum vector $h_t$ would be large for coordinates where gradients have the same sign on every iteration and will allow for large steps at these coordinates. For coordinates where the gradients revert the sign, their contributions to $h_t$ will mostly cancel each other out and $h_t$ will be close to zero. So $h_t$ cancels some of the parameter space coordinates that lead to oscillations and helps to achieve better convergence.

This additional feature introduces a new hyperparameter $\alpha$ with the typical value $\alpha=0.9$.

Figure below shows the modification of the oscillation example from the previous section; with Momentum added and the same initial step size, the GD does not oscillate, and converges towards the optimum instead:

![momentum gd](images\1_1_6_1-1.png =300x)

In practice, the momentum leads to faster convergence. On the figure above, SGD with momentum does not oscillate nearly as much as the one on the previous figure.

##### 1.1.6.2 Nesterov Momentum
_Nesterov Momentum_ is an extension of the Momentum method. It has a stronger theoretical convergence guarantees for convex functions and in practice also works slightly better than standard momentum.

In the simple Momentum method, on every iteration, a gradient $g_t$ at current point $w^{t-1}$ is calculated, and then a gradient step is taken in the direction that is a linear combination of the momentum vector $h_{t-1}$ and the gradient vector $g_t$.

Since the optimizer will move in the direction of momentum, it may be beneficial to do the first step in the direction $h_t$ to get some new approximation of the parameter vector and then to calculate approximation of the gradient of the loss function at some new point $w^{t-1} + h_t$:

![nesterov momentum](images\1_1_6_2-1.png =600x)

Mathematically, the following update is being carried out:

$h_t \leftarrow \alpha h_{t-1} + \eta_t \nabla L(w^{t-1} - \alpha h_{t-1})$

The algorithm for Mini-Batch GD with Nesterov Momentum is:

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad i_1, \dots, i_m \leftarrow \mathcal{U}\left\{ 1,n \right\}\\
\quad g_t \leftarrow \frac{1}{m} \displaystyle\sum_{j=1}^{m}{\nabla L(w^{t-1} - \alpha h_{t-1}|x_j, y_j)}\\
\quad h_t \leftarrow \alpha h_{t-1} + \eta_t g_t\\
\quad w^t \leftarrow w^{t-1} - h_t\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

In practice this method leads to better convergence than momentum method.

##### 1.1.6.3 Ada Grad
Both Momentum method and Nesterov Momentum method work well for complex functions, but they still require to choose the learning rate and are very sensitive to the choice.

_Ada Grad_ (for adaptive gradient algorithm) is an algorithm for gradient based optimization that adapts the learning rate to the parameters, performing smaller updates (i.e. low learning rates) for parameters associated with frequently occurring features and larger updates (i.e. high learning rates) for parameters associated with infrequent features. It is well suited for dealing with sparse data. Informally, this increases the learning rate for more sparse parameters and decreases the learning rate for less sparse ones. This strategy often improves convergence performance over standard stochastic gradient descent in settings where data is sparse and sparse parameters are more informative. It still has a base learning rate $\eta$.

Recall how a gradient step is made just for one coordinate $g$ of the parameter vector $w$. Now let $w_j^{t-1}$ be the $j$-th element of the $t-1$ iteration of the parameter vector $w$. The corresponding component from the gradient vector $g_{tj}$ (iteration $t$, component $j$) is then subtracted to obtain the new value:

$w_j^t \leftarrow w_j^{t-1} - g_{tj}$

where $g_{tj}$ is the gradient at time/iteration $t$ with respect to the $j$-th parameter.

In Ada Grad, an additional axillary vector $G$ for each element of the parameter vector $w$ is maintained. Such elements $G_j$ match with the elements of the parameter vector $w_j$. Values $G_j$ are calculated as follows:

$G_j^t \leftarrow G_j^{t-1} + g_{tj}^2$

Essentially, elements of vector $G$ are sums of squares of gradients from all previous iterations. The gradient step is then modified as follows:

$w_j^t \leftarrow w_j^{t-1} - \eta_t \frac{g_{tj}}{\sqrt{G_j^t + \epsilon}}$

The learning rate $\eta_t$ divided by the $\sqrt{G_j^t + \epsilon}$ where $\epsilon$ is some small number added to ensure that no division by zero  is possible. This makes the Ada Grad method to choose the learning rate adaptively and allows to relax the requirements to selection of $\eta_t$. Learning rate $\eta_t$ can the a fixed number at $\eta_t=\eta=0.01$, and since $G_j^t$ always increases, the will lead to the reduction of the learning rate and early stops.

Ada Grad has several disadvantages. The axillary vector $G_j^t$ accumulates the squares of gradients and at some step $t$ it will become to large and numerically unstable. If the learning rate $\eta$ is divided by the large number the GD will stop from progressing.

Algorithm below implements the Ada Grad algorithm:

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad i_1, \dots, i_m \leftarrow \mathcal{U}\left\{ 1,n \right\}\\
\quad g_t \leftarrow \frac{1}{m} \displaystyle\sum_{j=1}^{m}{\nabla L(w^{t-1}|x_j, y_j)}\\
\quad \ \textbf{for} \ j \ \textbf{in} \ 1 \dots p \textbf{:}\\
\quad \quad G_j^t \leftarrow G_j^{t-1} + g_{tj}^2\\
\quad \quad w_j^t \leftarrow w_j^{t-1} - \eta_t \frac{g_{tj}}{\sqrt{G_j^t + \epsilon}}\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

While designed for convex problems, AdaGrad has been successfully applied to non-convex optimization.

##### 1.1.6.4 RMSProp
_RMSProp_ (for Root Mean Square Propagation) is an improvement of Ada Grad and is also a method in which the learning rate is adapted for each of the parameters. This method is very similar to Ada Grad, but here an exponentially weighted average of squares of gradients on each step. So the update of the axillary parameter vector $G^t$ becomes:

$G_j^t \leftarrow \alpha G_j^{t-1} + (1-\alpha) g_{tj}^2$

The additional _forgetting parameter_ $\alpha$ is typically set at $0.9$. The elements of the parameter vector $w_j^t$ are then updated in the same way as in the Ada Grad algorithm. This modification overcomes the problem of large sums of square gradients, and the learning rate depends mostly on the last example from the gradient descent method.

##### 1.1.6.4 RMSProp
_Adam_ (short for Adaptive Moment Estimation) is an update to the RMSProp optimizer. In this optimization algorithm, running averages of both the gradients and the second moments of the gradients are used. The axillary vector $G^t$ of the RMSProp is further augmented and conventionally denoted as $\nu_j^t$:

$\nu_j^t \leftarrow \frac{ \beta_2 \nu_j^{t-1} + (1-\beta_2) g_{tj}^2 }{ 1 - \beta_2^t }$

Note that $\nu_j^t$ has some bias towards zero, especially in the first iterations since it is initialized with zero. So to overcome the bias, it is divided by $1 - \beta_2^t$, where $\beta$ is raised to the power of $t$. This normalization allows to get rid of the zero bias. At first steps the normalization denominator is large, but as $t$ increases, the value of $\beta^t$ decreases, and the normalization denominator approaches $1.0$.

The $\nu_j^t$ variable is then used to update the gradient similar to the Ada Grad and RMSProp:

$w_j^t \leftarrow w_j^t - \eta_t \frac{ g_{tj} }{ \sqrt{\nu_j^t + \epsilon} }$

From the SGD and Momentum methods, it was shown that stochastic optimization may be prone to oscillations. The gradients may be smoothed by adding another auxiliary variable $m_j^t$, which is essentially a sum of gradients to address that issue:

$m_j^t \leftarrow \frac{ \beta_1 m_j^{t-1} + (1-\beta_1) g_{tj} }{ 1 - \beta_1^t }$

The weight update step then becomes:

$w_j^t \leftarrow w_j^t - \eta_t \frac{ m_{tj} }{ \sqrt{\nu_j^t + \epsilon} }$

Putting all of the above together yields the following algorithm:

>$t \leftarrow 0\\
\textbf{while True:}\\
\quad i_1, \dots, i_m \leftarrow \mathcal{U}\left\{ 1,n \right\}\\
\quad g_t \leftarrow \frac{1}{m} \displaystyle\sum_{j=1}^{m}{\nabla L(w^{t-1}|x_j, y_j)}\\
\quad \ \textbf{for} \ j \ \textbf{in} \ 1 \dots p \textbf{:}\\
\quad \quad \nu_j^t \leftarrow \frac{ \beta_2 \nu_j^{t-1} + (1-\beta_2) g_{tj}^2 }{ 1 - \beta_2^t }\\
\quad \quad m_j^t \leftarrow \frac{ \beta_1 m_j^{t-1} + (1-\beta_1) g_{tj} }{ 1 - \beta_1^t }\\
\quad \quad w_j^t \leftarrow w_j^t - \eta_t \frac{ m_{tj} }{ \sqrt{\nu_j^t + \epsilon} }\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$

In Adam $\beta_1$ and $\beta_2$ are the forgetting factors for gradients and second moments of gradients, respectively. Squaring and square-rooting is done elementwise. Typical values (as per the original paper [_Adam: A Method for Stochastic Optimization_ by Diederik P. Kingma and Jimmy Baare](https://arxiv.org/abs/1412.6980)) are $\eta=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, and $\epsilon=10^{-8}$.
