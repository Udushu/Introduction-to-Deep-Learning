---
# 1. Introduction to Deep Learning
---
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






$t \leftarrow 0\\
\textbf{while True:}\\
\quad w^t \leftarrow w^{t-1} - \eta \nabla L(w^{t-1})\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$
