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

![linear model](1_1_1_2-1.png =300x)

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










$t \leftarrow 0\\
\textbf{while True:}\\
\quad w^t \leftarrow w^{t-1} - \eta \nabla L(w^{t-1})\\
\quad \textbf{if} \ {\lVert w^t - w^{t-1} \rVert}^2 < \epsilon \ \textbf{then break}\\
\quad t \leftarrow t+1$
