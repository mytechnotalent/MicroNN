![image](https://github.com/mytechnotalent/MicroNN/blob/main/MicroNN.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# MicroNN: A Neural Network Tutorial from Scratch

A step-by-step guide to understanding neural networks with actual math and real numbers. By the end of this tutorial, you'll understand exactly how a neural network learns to recognize handwritten digits.

---

## Dataset Setup

Before running the notebooks, you'll need to download the MNIST dataset:

1. Go to [MNIST in CSV on Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
2. Download `mnist_train.csv`
3. Place the file in the root folder of this project

---

## Table of Contents

1. [What is a Neural Network?](#what-is-a-neural-network)
2. [Our Network Architecture](#our-network-architecture)
3. [Step 1: Initialize Weights and Biases](#step-1-initialize-weights-and-biases)
4. [Step 2: Forward Pass](#step-2-forward-pass)
5. [Step 3: Activation Functions](#step-3-activation-functions)
6. [Step 4: One-Hot Encoding](#step-4-one-hot-encoding)
7. [Step 5: Loss Function](#step-5-loss-function)
8. [Step 6: Backpropagation](#step-6-backpropagation)
9. [Step 7: Gradient Descent](#step-7-gradient-descent)
10. [Step 8: Prediction](#step-8-prediction)
11. [Full Worked Example](#full-worked-example)

---

## What is a Neural Network?

A neural network is a math function that learns patterns from data. Think of it like this:

- **Input**: A picture of a handwritten digit (784 pixel values for a 28×28 image)
- **Output**: A guess of which digit (0-9) the picture shows
- **Learning**: The network adjusts its internal numbers (weights) to make better guesses

The "learning" happens by:
1. Making a guess (forward pass)
2. Checking how wrong the guess was (loss)
3. Figuring out how to adjust weights to be less wrong (backpropagation)
4. Actually adjusting the weights (gradient descent)
5. Repeating thousands of times

---

## Our Network Architecture

Our network has 3 layers:

```
Input Layer (784 neurons) → Hidden Layer (10 neurons) → Output Layer (10 neurons)
```

Each connection between neurons has a **weight** (a number that gets multiplied). Each neuron also has a **bias** (a number that gets added).

**Shape notation**: We write shapes as (rows, columns). A single image with 784 pixels is shaped (784, 1).

---

## Step 1: Initialize Weights and Biases

Before training, we randomly initialize all weights and biases to small numbers between -0.5 and 0.5.

### Weight Matrices

**W1** connects input to hidden layer:
- Shape: (hidden_size, input_size) = (10, 784)
- That's 7,840 individual weights!

**W2** connects hidden to output layer:
- Shape: (output_size, hidden_size) = (10, 10)
- That's 100 weights

### Bias Vectors

**b1** for hidden layer: Shape (10, 1)
**b2** for output layer: Shape (10, 1)

### Simplified Example

Let's use a tiny network to make the math easy to follow:
- 3 input features (instead of 784)
- 2 hidden neurons (instead of 10)
- 2 output classes (instead of 10)

Random initialization:

$$
W_1 = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4 & 0.1 & -0.2 \end{bmatrix}
\quad
b_1 = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
$$

$$
W_2 = \begin{bmatrix} 0.3 & -0.2 \\ -0.1 & 0.4 \end{bmatrix}
\quad
b_2 = \begin{bmatrix} 0.05 \\ -0.05 \end{bmatrix}
$$

---

## Step 2: Forward Pass

The forward pass pushes input data through the network to get a prediction.

### Layer 1: Input → Hidden

First, we compute the **pre-activation** (before applying the activation function):

$$
z_1 = W_1 \cdot x + b_1
$$

Where:
- $x$ is the input (shape: 3×1)
- $W_1$ is the weight matrix (shape: 2×3)
- $b_1$ is the bias vector (shape: 2×1)
- $z_1$ is the result (shape: 2×1)

### Worked Example

Let's say our input is:

$$
x = \begin{bmatrix} 0.5 \\ 0.8 \\ 0.2 \end{bmatrix}
$$

Calculate $z_1 = W_1 \cdot x + b_1$:

$$
z_1 = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4 & 0.1 & -0.2 \end{bmatrix} \cdot \begin{bmatrix} 0.5 \\ 0.8 \\ 0.2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
$$

Let's compute each element:

**First row of z1:**
$$
z_1[0] = (0.2 \times 0.5) + (-0.3 \times 0.8) + (0.1 \times 0.2) + 0.1
$$
$$
z_1[0] = 0.1 - 0.24 + 0.02 + 0.1 = -0.02
$$

**Second row of z1:**
$$
z_1[1] = (0.4 \times 0.5) + (0.1 \times 0.8) + (-0.2 \times 0.2) + (-0.1)
$$
$$
z_1[1] = 0.2 + 0.08 - 0.04 - 0.1 = 0.14
$$

So:
$$
z_1 = \begin{bmatrix} -0.02 \\ 0.14 \end{bmatrix}
$$

### Apply ReLU Activation

Now we apply ReLU to get $a_1$ (the actual activation/output of hidden layer):

$$
a_1 = \text{ReLU}(z_1) = \begin{bmatrix} \max(0, -0.02) \\ \max(0, 0.14) \end{bmatrix} = \begin{bmatrix} 0 \\ 0.14 \end{bmatrix}
$$

### Layer 2: Hidden → Output

Now we do the same thing for the output layer:

$$
z_2 = W_2 \cdot a_1 + b_2
$$

$$
z_2 = \begin{bmatrix} 0.3 & -0.2 \\ -0.1 & 0.4 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 0.14 \end{bmatrix} + \begin{bmatrix} 0.05 \\ -0.05 \end{bmatrix}
$$

**First row of z2:**
$$
z_2[0] = (0.3 \times 0) + (-0.2 \times 0.14) + 0.05 = 0 - 0.028 + 0.05 = 0.022
$$

**Second row of z2:**
$$
z_2[1] = (-0.1 \times 0) + (0.4 \times 0.14) + (-0.05) = 0 + 0.056 - 0.05 = 0.006
$$

So:
$$
z_2 = \begin{bmatrix} 0.022 \\ 0.006 \end{bmatrix}
$$

### Apply Softmax Activation

Finally, we apply softmax to get probabilities:

$$
a_2 = \text{softmax}(z_2)
$$

---

## Step 3: Activation Functions

### ReLU (Rectified Linear Unit)

ReLU is simple: if the input is negative, output 0. Otherwise, output the input unchanged.

$$
\text{ReLU}(z) = \max(0, z)
$$

**Examples:**
- $\text{ReLU}(3.5) = 3.5$
- $\text{ReLU}(-2.1) = 0$
- $\text{ReLU}(0) = 0$

**Why use ReLU?** It introduces non-linearity (the network can learn curved patterns, not just straight lines) and it's fast to compute.

### ReLU Derivative

For backpropagation, we need the derivative:

$$
\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
$$

**Examples:**
- $\text{ReLU}'(3.5) = 1$
- $\text{ReLU}'(-2.1) = 0$

### Softmax

Softmax converts raw scores (logits) into probabilities that sum to 1.

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

**Worked Example:**

Given $z_2 = \begin{bmatrix} 0.022 \\ 0.006 \end{bmatrix}$:

First, compute $e^z$ for each element:
$$
e^{0.022} \approx 1.0222
$$
$$
e^{0.006} \approx 1.0060
$$

Sum: $1.0222 + 1.0060 = 2.0282$

Now divide each by the sum:
$$
a_2[0] = \frac{1.0222}{2.0282} \approx 0.504
$$
$$
a_2[1] = \frac{1.0060}{2.0282} \approx 0.496
$$

So:
$$
a_2 = \begin{bmatrix} 0.504 \\ 0.496 \end{bmatrix}
$$

This means the network thinks there's a 50.4% chance it's class 0 and 49.6% chance it's class 1. (Not very confident—but we haven't trained it yet!)

---

## Step 4: One-Hot Encoding

One-hot encoding converts a class label (like "3") into a vector with a 1 in that position and 0s everywhere else.

### Formula

For a label $y$ with $C$ classes:

$$
\text{one\_hot}(y) = \text{vector of length } C \text{ with } 1 \text{ at position } y
$$

### Examples

With 10 classes (digits 0-9):

| Label | One-Hot Vector |
|-------|----------------|
| 0 | $[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]$ |
| 1 | $[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]$ |
| 3 | $[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$ |
| 7 | $[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]$ |

### Worked Example (2 classes)

If the true label is $y = 1$ (class 1):

$$
\text{one\_hot}(1) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

If the true label is $y = 0$ (class 0):

$$
\text{one\_hot}(0) = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

---

## Step 5: Loss Function

The **loss** tells us how wrong our prediction is. We use **cross-entropy loss** for classification.

### Cross-Entropy Loss Formula

$$
L = -\sum_{i} y_i \log(a_i)
$$

Where:
- $y_i$ is the one-hot encoded true label (0 or 1)
- $a_i$ is the predicted probability for class $i$

Since $y$ is one-hot, only one term survives (where $y_i = 1$):

$$
L = -\log(a_{\text{true class}})
$$

### Worked Example

Our prediction: $a_2 = \begin{bmatrix} 0.504 \\ 0.496 \end{bmatrix}$

**Case 1: True label is 0**

One-hot: $y = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

$$
L = -(1 \times \log(0.504) + 0 \times \log(0.496))
$$
$$
L = -\log(0.504) \approx -(-0.686) = 0.686
$$

**Case 2: True label is 1**

One-hot: $y = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

$$
L = -(0 \times \log(0.504) + 1 \times \log(0.496))
$$
$$
L = -\log(0.496) \approx -(-0.701) = 0.701
$$

**Interpretation:**
- Loss of 0 means perfect prediction (predicted probability = 1.0 for correct class)
- Higher loss = worse prediction
- Loss approaches infinity as predicted probability approaches 0 for the correct class

---

## Step 6: Backpropagation

Backpropagation calculates how much each weight contributed to the error, so we know how to adjust them.

The key idea: use the **chain rule** from calculus to work backwards from the loss to each weight.

### The Magic Shortcut: Softmax + Cross-Entropy

When you combine softmax with cross-entropy loss, the gradient at the output layer simplifies beautifully:

$$
\frac{\partial L}{\partial z_2} = a_2 - y
$$

That's it! Just subtract the one-hot label from the prediction.

### Worked Example: Output Layer Gradient

Prediction: $a_2 = \begin{bmatrix} 0.504 \\ 0.496 \end{bmatrix}$

True label: $y = 1$, so one-hot $y = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

$$
dz_2 = a_2 - y = \begin{bmatrix} 0.504 \\ 0.496 \end{bmatrix} - \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.504 \\ -0.504 \end{bmatrix}
$$

**Interpretation:**
- $dz_2[0] = 0.504$ means "you predicted 0.504 for class 0, but should have predicted 0"
- $dz_2[1] = -0.504$ means "you predicted 0.496 for class 1, but should have predicted 1"

### Weight Gradients for Layer 2

$$
dW_2 = \frac{1}{m} \cdot dz_2 \cdot a_1^T
$$

Where $m$ is the batch size (number of samples). For one sample, $m = 1$.

$$
dW_2 = dz_2 \cdot a_1^T = \begin{bmatrix} 0.504 \\ -0.504 \end{bmatrix} \cdot \begin{bmatrix} 0 & 0.14 \end{bmatrix}
$$

$$
dW_2 = \begin{bmatrix} 0.504 \times 0 & 0.504 \times 0.14 \\ -0.504 \times 0 & -0.504 \times 0.14 \end{bmatrix} = \begin{bmatrix} 0 & 0.0706 \\ 0 & -0.0706 \end{bmatrix}
$$

### Bias Gradients for Layer 2

$$
db_2 = \frac{1}{m} \sum dz_2 = dz_2 = \begin{bmatrix} 0.504 \\ -0.504 \end{bmatrix}
$$

### Backpropagate to Hidden Layer

Now we push the gradient backwards through the network:

$$
dz_1 = (W_2^T \cdot dz_2) \odot \text{ReLU}'(z_1)
$$

Where $\odot$ means element-wise multiplication.

**Step 1: Compute $W_2^T \cdot dz_2$**

$$
W_2^T = \begin{bmatrix} 0.3 & -0.1 \\ -0.2 & 0.4 \end{bmatrix}
$$

$$
W_2^T \cdot dz_2 = \begin{bmatrix} 0.3 & -0.1 \\ -0.2 & 0.4 \end{bmatrix} \cdot \begin{bmatrix} 0.504 \\ -0.504 \end{bmatrix}
$$

First element: $(0.3 \times 0.504) + (-0.1 \times -0.504) = 0.1512 + 0.0504 = 0.2016$

Second element: $(-0.2 \times 0.504) + (0.4 \times -0.504) = -0.1008 - 0.2016 = -0.3024$

$$
W_2^T \cdot dz_2 = \begin{bmatrix} 0.2016 \\ -0.3024 \end{bmatrix}
$$

**Step 2: Compute ReLU derivative**

Recall $z_1 = \begin{bmatrix} -0.02 \\ 0.14 \end{bmatrix}$

$$
\text{ReLU}'(z_1) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

(0 because -0.02 ≤ 0, and 1 because 0.14 > 0)

**Step 3: Element-wise multiply**

$$
dz_1 = \begin{bmatrix} 0.2016 \\ -0.3024 \end{bmatrix} \odot \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ -0.3024 \end{bmatrix}
$$

### Weight Gradients for Layer 1

$$
dW_1 = dz_1 \cdot x^T = \begin{bmatrix} 0 \\ -0.3024 \end{bmatrix} \cdot \begin{bmatrix} 0.5 & 0.8 & 0.2 \end{bmatrix}
$$

$$
dW_1 = \begin{bmatrix} 0 & 0 & 0 \\ -0.1512 & -0.2419 & -0.0605 \end{bmatrix}
$$

### Bias Gradients for Layer 1

$$
db_1 = dz_1 = \begin{bmatrix} 0 \\ -0.3024 \end{bmatrix}
$$

---

## Step 7: Gradient Descent

Now we update each weight and bias by subtracting a small fraction of its gradient.

### The Update Rule

$$
W = W - \alpha \cdot dW
$$
$$
b = b - \alpha \cdot db
$$

Where $\alpha$ is the **learning rate** (typically 0.01 to 0.5).

### Worked Example with Learning Rate = 0.1

**Update W2:**

$$
W_2^{\text{new}} = W_2 - 0.1 \times dW_2
$$

$$
W_2^{\text{new}} = \begin{bmatrix} 0.3 & -0.2 \\ -0.1 & 0.4 \end{bmatrix} - 0.1 \times \begin{bmatrix} 0 & 0.0706 \\ 0 & -0.0706 \end{bmatrix}
$$

$$
W_2^{\text{new}} = \begin{bmatrix} 0.3 & -0.2 - 0.00706 \\ -0.1 & 0.4 + 0.00706 \end{bmatrix} = \begin{bmatrix} 0.3 & -0.2071 \\ -0.1 & 0.4071 \end{bmatrix}
$$

**Update b2:**

$$
b_2^{\text{new}} = \begin{bmatrix} 0.05 \\ -0.05 \end{bmatrix} - 0.1 \times \begin{bmatrix} 0.504 \\ -0.504 \end{bmatrix} = \begin{bmatrix} -0.0004 \\ 0.0004 \end{bmatrix}
$$

**Update W1:**

$$
W_1^{\text{new}} = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4 & 0.1 & -0.2 \end{bmatrix} - 0.1 \times \begin{bmatrix} 0 & 0 & 0 \\ -0.1512 & -0.2419 & -0.0605 \end{bmatrix}
$$

$$
W_1^{\text{new}} = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4151 & 0.1242 & -0.1940 \end{bmatrix}
$$

**Update b1:**

$$
b_1^{\text{new}} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} - 0.1 \times \begin{bmatrix} 0 \\ -0.3024 \end{bmatrix} = \begin{bmatrix} 0.1 \\ -0.0698 \end{bmatrix}
$$

---

## Step 8: Prediction

After training, prediction is just the forward pass followed by picking the class with highest probability.

### Formula

$$
\text{prediction} = \arg\max(a_2)
$$

### Example

If after training, the network outputs:

$$
a_2 = \begin{bmatrix} 0.02 \\ 0.03 \\ 0.01 \\ 0.85 \\ 0.02 \\ 0.01 \\ 0.02 \\ 0.01 \\ 0.02 \\ 0.01 \end{bmatrix}
$$

The prediction is **3** because $a_2[3] = 0.85$ is the largest value.

---

## Full Worked Example

Let's trace through one complete training step with our tiny network.

### Setup

**Network:**
- 3 inputs, 2 hidden neurons, 2 output classes
- Learning rate: $\alpha = 0.1$

**Initial Weights:**

$$
W_1 = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4 & 0.1 & -0.2 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
$$

$$
W_2 = \begin{bmatrix} 0.3 & -0.2 \\ -0.1 & 0.4 \end{bmatrix}, \quad b_2 = \begin{bmatrix} 0.05 \\ -0.05 \end{bmatrix}
$$

**Input and Label:**

$$
x = \begin{bmatrix} 0.5 \\ 0.8 \\ 0.2 \end{bmatrix}, \quad y = 1 \text{ (true class)}
$$

---

### Step 1: Forward Pass - Layer 1

$$
z_1 = W_1 \cdot x + b_1 = \begin{bmatrix} -0.02 \\ 0.14 \end{bmatrix}
$$

$$
a_1 = \text{ReLU}(z_1) = \begin{bmatrix} 0 \\ 0.14 \end{bmatrix}
$$

---

### Step 2: Forward Pass - Layer 2

$$
z_2 = W_2 \cdot a_1 + b_2 = \begin{bmatrix} 0.022 \\ 0.006 \end{bmatrix}
$$

$$
a_2 = \text{softmax}(z_2) = \begin{bmatrix} 0.504 \\ 0.496 \end{bmatrix}
$$

---

### Step 3: Calculate Loss

True label one-hot: $y = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$

$$
L = -\log(0.496) \approx 0.701
$$

---

### Step 4: Backpropagation

**Output layer:**
$$
dz_2 = a_2 - y = \begin{bmatrix} 0.504 \\ -0.504 \end{bmatrix}
$$

$$
dW_2 = \begin{bmatrix} 0 & 0.0706 \\ 0 & -0.0706 \end{bmatrix}, \quad db_2 = \begin{bmatrix} 0.504 \\ -0.504 \end{bmatrix}
$$

**Hidden layer:**
$$
dz_1 = \begin{bmatrix} 0 \\ -0.3024 \end{bmatrix}
$$

$$
dW_1 = \begin{bmatrix} 0 & 0 & 0 \\ -0.1512 & -0.2419 & -0.0605 \end{bmatrix}, \quad db_1 = \begin{bmatrix} 0 \\ -0.3024 \end{bmatrix}
$$

---

### Step 5: Gradient Descent Update

$$
W_2^{\text{new}} = \begin{bmatrix} 0.3 & -0.2071 \\ -0.1 & 0.4071 \end{bmatrix}
$$

$$
b_2^{\text{new}} = \begin{bmatrix} -0.0004 \\ 0.0004 \end{bmatrix}
$$

$$
W_1^{\text{new}} = \begin{bmatrix} 0.2 & -0.3 & 0.1 \\ 0.4151 & 0.1242 & -0.1940 \end{bmatrix}
$$

$$
b_1^{\text{new}} = \begin{bmatrix} 0.1 \\ -0.0698 \end{bmatrix}
$$

---

### Step 6: Verify Improvement

Let's run forward pass with new weights:

$$
z_1^{\text{new}} = W_1^{\text{new}} \cdot x + b_1^{\text{new}}
$$

First neuron: $(0.2)(0.5) + (-0.3)(0.8) + (0.1)(0.2) + 0.1 = -0.02$ → ReLU → $0$

Second neuron: $(0.4151)(0.5) + (0.1242)(0.8) + (-0.194)(0.2) + (-0.0698) = 0.1979$ → ReLU → $0.1979$

$$
a_1^{\text{new}} = \begin{bmatrix} 0 \\ 0.1979 \end{bmatrix}
$$

$$
z_2^{\text{new}} = W_2^{\text{new}} \cdot a_1^{\text{new}} + b_2^{\text{new}}
$$

First output: $(0.3)(0) + (-0.2071)(0.1979) + (-0.0004) = -0.0414$

Second output: $(-0.1)(0) + (0.4071)(0.1979) + (0.0004) = 0.0810$

$$
z_2^{\text{new}} = \begin{bmatrix} -0.0414 \\ 0.0810 \end{bmatrix}
$$

Softmax:
- $e^{-0.0414} \approx 0.9594$
- $e^{0.0810} \approx 1.0844$
- Sum: $2.0438$

$$
a_2^{\text{new}} = \begin{bmatrix} 0.469 \\ 0.531 \end{bmatrix}
$$

**Before training:** Class 1 probability = 0.496
**After one step:** Class 1 probability = 0.531

The network moved in the right direction! Repeat this thousands of times and the network learns.

---

## Summary

| Step | What Happens | Formula |
|------|--------------|---------|
| Initialize | Random weights | $W \sim \text{Uniform}(-0.5, 0.5)$ |
| Forward L1 | Linear transform | $z_1 = W_1 x + b_1$ |
| Activate L1 | Apply ReLU | $a_1 = \max(0, z_1)$ |
| Forward L2 | Linear transform | $z_2 = W_2 a_1 + b_2$ |
| Activate L2 | Apply softmax | $a_2 = \frac{e^{z_2}}{\sum e^{z_2}}$ |
| Loss | Measure error | $L = -\log(a_{2,\text{true}})$ |
| Backprop L2 | Output gradient | $dz_2 = a_2 - y$ |
| Backprop L1 | Hidden gradient | $dz_1 = W_2^T dz_2 \odot \text{ReLU}'(z_1)$ |
| Weight grads | Compute gradients | $dW = dz \cdot a_{\text{prev}}^T$ |
| Update | Gradient descent | $W = W - \alpha \cdot dW$ |
| Predict | Pick best class | $\hat{y} = \arg\max(a_2)$ |

---

## Key Takeaways

1. **Forward pass** = matrix multiplication + activation functions
2. **Softmax** converts scores to probabilities
3. **Cross-entropy loss** measures how wrong predictions are
4. **Backpropagation** = chain rule to find gradients
5. **Gradient descent** = subtract gradients to improve weights
6. **Training** = repeat forward → loss → backward → update, thousands of times

The network "learns" by slowly adjusting weights to make better predictions!

