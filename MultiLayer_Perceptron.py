import matplotlib.pyplot as plt
import numpy as np
from random import random
from math import pi, sin, tanh

# 1. Data preparation
x = np.arange(0, 1, 1/22)
print(x)
print(type(x))
print()

d = []
for xn in x:
    d.append(((1 + 0.6 * sin (2 * pi * xn / 0.7)) + 0.3 * sin (2 * pi * xn)) / 2)
print(d)

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.plot(x, d, label="Desired output")

# 2. Structure selection
# one hidden layer with four neurons in it; one output neuron
# activation function in hidden layer: tanh(v)
# activation function in output layer: linear
# the cost function: E =

# 3. Initate parameters
# First (hidden) layer
w11_1 = random(); w12_1 = random(); w13_1 = random(); w14_1 = random()
b1_1  = random(); b2_1  = random(); b3_1  = random(); b4_1  = random()

# Last (output) layer
w11_2 = random(); w12_2 = random(); w13_2 = random(); w14_2 = random()
b1_2  = random()
eta = 0.01

# Training
for k in range(20000):
    for i in range(len(x)):
        # 4. Estimate output
        # First (hidden) layer
        v1_1 = x[i]*w11_1 + b1_1
        v2_1 = x[i]*w12_1 + b2_1
        v3_1 = x[i]*w13_1 + b3_1
        v4_1 = x[i]*w14_1 + b4_1
        # Activation
        y1_1 = tanh(v1_1)
        y2_1 = tanh(v2_1)
        y3_1 = tanh(v3_1)
        y4_1 = tanh(v4_1)
        # Last (output) layer
        v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2
        #Activation
        y1_2 = v1_2

        # 5. Calculate error
        e = d[i] - y1_2

        # 6. Update parameters
        # for output neurons: w = w + n*D_out*Input
        # D_out = derivative_of_activation*derivative_of_cost
        delta1_2 = e
        delta1_1 = (1-(tanh(y1_1)**2))*delta1_2*w11_2
        delta2_1 = (1-(tanh(y2_1)**2))*delta1_2*w12_2
        delta3_1 = (1-(tanh(y3_1)**2))*delta1_2*w13_2
        delta4_1 = (1-(tanh(y4_1)**2))*delta1_2*w14_2

        # Last layer
        w11_2 = w11_2+eta*delta1_2*y1_1
        w12_2 = w12_2+eta*delta1_2*y2_1
        w13_2 = w13_2+eta*delta1_2*y3_1
        w14_2 = w14_2+eta*delta1_2*y4_1
        b1_2  = b1_2+eta*delta1_2

        # First layer
        w11_1 = w11_1+eta*delta1_1*x[i]
        w12_1 = w12_1+eta*delta2_1*x[i]
        w13_1 = w13_1+eta*delta3_1*x[i]
        w14_1 = w14_1+eta*delta4_1*x[i]
        b1_1  = b1_1+eta*delta1_1
        b2_1  = b2_1+eta*delta2_1
        b3_1  = b3_1+eta*delta3_1
        b4_1  = b4_1+eta*delta4_1

# Testing
X = np.arange(0, 1, 1/22)
Y = np.zeros(len(X))
for i in range(len(X)):
    # First (hidden) layer
    v1_1 = X[i]*w11_1 + b1_1; v2_1 = X[i]*w12_1 + b2_1; v3_1 = X[i]*w13_1 + b3_1; v4_1 = X[i]*w14_1 + b4_1
    # Activation
    y1_1 = tanh(v1_1); y2_1 = tanh(v2_1); y3_1 = tanh(v3_1); y4_1 = tanh(v4_1)
    # Last (output) layer
    v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2
    # Activation
    Y[i] = v1_2
    
    # 5. Calculate error
    #e = d[i] - y1_2

    

print()
print(Y)
ax.plot(X, Y, label="Percepton output")
ax.set_title("MultiLayer Perceptron")
ax.legend(loc = "lower left")
plt.show()











        
