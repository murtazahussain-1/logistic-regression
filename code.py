import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt


def HypothesisFunction(z):
    gz = 1/(1+np.exp(-z))
    return gz


def EvaluateLogisticRegression(Theta0, Theta1, Theta2):
    exam1 = 45
    exam2 = 85
    print('Passing Probability of this Student = ',
          HypothesisFunction(Theta0+Theta1*exam1+Theta2*exam2))


def CostFunction(Theta0, Theta1, Theta2):
    data = np.loadtxt("ex1data.txt", delimiter=",", skiprows=0)
    SubjectOneMarks = data[:, 0]
    SubjectTwoMarks = data[:, 1]
    PassOrFail = data[:, 2]

    Total = SubjectOneMarks.size  # Total Samples

    H_Theeta = HypothesisFunction(
        (Theta0+(Theta1*SubjectOneMarks)+(Theta2*SubjectTwoMarks)))
    J_Theta = ((1/Total)*np.sum((-PassOrFail*np.log(H_Theeta)) -
               (1-PassOrFail)*(np.log(1-H_Theeta))))
    return(J_Theta)


def LogisticRegressionTraining():
    print('Training Begins')
    data = np.loadtxt("ex1data.txt", delimiter=",", skiprows=0)
    SubjectOneMarks = data[:, 0]
    SubjectTwoMarks = data[:, 1]
    PassOrFail = data[:, 2]

    # Total Samples=
    Total = SubjectOneMarks.size

    SubjectOneMarks_M = SubjectOneMarks.reshape(Total, 1)
    SubjectTwoMarks_M = SubjectTwoMarks.reshape(Total, 1)

    PassOrFail = PassOrFail.reshape(Total, 1)

    alpha = 0.004
    Total_Iterations = 400000
    Theta0 = 0
    Theta1 = 0
    Theta2 = 0

    for i in range(Total_Iterations):
        Hypo = HypothesisFunction(
            (Theta0+(Theta1*SubjectOneMarks_M)+(Theta2*SubjectTwoMarks_M)))
        Theta0 = Theta0 - ((alpha/Total)*np.sum(Hypo - PassOrFail))
        Theta1 = Theta1 - \
            ((alpha/Total)*np.sum((Hypo-PassOrFail)*SubjectOneMarks_M))
        Theta2 = Theta2 - \
            ((alpha/Total)*np.sum((Hypo-PassOrFail)*SubjectTwoMarks_M))

    print('Final Values of Theta')
    print('Theta 0 = ', Theta0, 'Theta 1 = ', Theta1, 'Theta 2 = ', Theta2)
    print('J_Theta = ', CostFunction(Theta0, Theta1, Theta2))

    temp1 = 0
    temp2 = 0

    # Plotting the scatter graph
    for i in range(Total):
        if PassOrFail[i] == 0:
            if temp1 == 0:
                plt.scatter(SubjectOneMarks[i], SubjectTwoMarks[i],
                            label='Failed', color='y', marker='o', s=50)
                temp1 += 1
            else:
                plt.scatter(
                    SubjectOneMarks[i], SubjectTwoMarks[i], color='y', marker='o', s=50)
        else:
            if temp2 == 0:
                plt.scatter(SubjectOneMarks[i], SubjectTwoMarks[i],
                            label='Passed', color='black', marker='+', s=50)
                temp2 += 1
            else:
                plt.scatter(
                    SubjectOneMarks[i], SubjectTwoMarks[i], color='black', marker='+', s=50)

    # plotting the decision boundary
    X_ValuesARR = np.linspace(0, Total-1, Total)
    X_ValuesARR = X_ValuesARR.reshape(Total, 1)

    plt.plot(X_ValuesARR, (-Theta0-(Theta1*X_ValuesARR))/Theta2)

    plt.xlabel('Subject 1')
    plt.ylabel('Subject 2')
    plt.title('Pass or Fail')
    plt.legend()
    plt.show()

    # Predict the passing or failing of a student. Using the trained algorithm
    EvaluateLogisticRegression(Theta0, Theta1, Theta2)


def DataPlot():
    print('Opening File...')
    data = np.loadtxt("ex1data.txt", delimiter=",", skiprows=0)
    SubjectOneMarks = data[:, 0]
    SubjectTwoMarks = data[:, 1]
    PassOrFail = data[:, 2]

    Total = SubjectOneMarks.size

    SubjectOneMarks = SubjectOneMarks.reshape(Total, 1)
    SubjectTwoMarks = SubjectTwoMarks.reshape(Total, 1)
    PassOrFail = PassOrFail.reshape(Total, 1)

    temp1 = 0
    temp2 = 0

    for i in range(Total):
        if PassOrFail[i] == 0:
            if temp1 == 0:
                plt.scatter(SubjectOneMarks[i], SubjectTwoMarks[i],
                            label='Failed', color='y', marker='o', s=50)
                temp1 += 1
            else:
                plt.scatter(
                    SubjectOneMarks[i], SubjectTwoMarks[i], color='y', marker='o', s=50)
        else:
            if temp2 == 0:
                plt.scatter(SubjectOneMarks[i], SubjectTwoMarks[i],
                            label='Passed', color='black', marker='+', s=50)
                temp2 += 1
            else:
                plt.scatter(
                    SubjectOneMarks[i], SubjectTwoMarks[i], color='black', marker='+', s=50)

    plt.xlabel('Subject 1')
    plt.ylabel('Subject 2')
    plt.title('Pass or Fail')
    plt.legend()
    plt.show()


# =================================================

def mapFeature(x1, x2, degree):
    out = np.ones(len(x1)).reshape(len(x1), 1)
    for i in range(1, degree+1):
        for j in range(i+1):
            terms = (x1**(i-j) * x2**j).reshape(len(x1), 1)
            out = np.hstack((out, terms))  # Combies two arrays horizontally
    return out


def costFunctionReg(theta, X, y, Lambda):
    """
    Take in numpy array of theta, X, and y to return the regularize cost function and gradient
    of a logistic regression
    """

    m = len(y)
    y = y[:, np.newaxis]
    predictions = HypothesisFunction(X @ theta)
    error = (-y * np.log(predictions)) - ((1 - y) * np.log(1 - predictions))
    cost = 1 / m * sum(error)
    regCost = cost + Lambda / (2 * m) * sum(theta ** 2)

    # compute gradient
    j_0 = 1 / m * (X.transpose() @ (predictions - y))[0]
    j_1 = 1 / m * (X.transpose() @ (predictions - y)
                   )[1:] + (Lambda / m) * theta[1:]
    grad = np.vstack((j_0[:, np.newaxis], j_1))
    return cost[0], grad


def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        cost, grad = costFunctionReg(theta, X, y, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history


def mapFeaturePlot(x1, x2, degree):
    # take in numpy array of x1 and x2, return all polynomial terms up to the given degree
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            terms = (x1**(i-j) * x2**j)
            out = np.hstack((out, terms))  # Combines two arrays horizontally
    return out


def RegularizedLogisticRegression():
    print('Regularized Logistic Regression Training Begins')
    df = pd.read_csv("ex2data.txt", header=None)
    df.head()
    df.describe()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    pos, neg = (y == 1).reshape(118, 1), (y == 0).reshape(118, 1)
    plt.scatter(X[pos[:, 0], 0], X[pos[:, 0], 1], c="r", marker="+")
    plt.scatter(X[neg[:, 0], 0], X[neg[:, 0], 1], marker="o", s=10)
    plt.xlabel("Test 1")
    plt.ylabel("Test 2")
    plt.legend(["Accepted", "Rejected"], loc=0)
    plt.show()

    #MapFeature(x1, x2, Max_Degree)
    X = mapFeature(X[:, 0], X[:, 1], 6)

    # Initialize fitting parameters set to zero
    initial_theta = np.zeros((X.shape[1], 1))

    # Set regularization parameter lambda to 1
    Lambda = 0

    # Compute and display initial cost and gradient for regularized logistic regression
    cost, grad = costFunctionReg(initial_theta, X, y, Lambda)

    print("Cost at initial theta (zeros):", cost)

    theta, J_history = gradientDescent(X, y, initial_theta, 1, 800, 0.2)

    print("The regularized theta using ridge regression:\n", theta)

    plt.scatter(X[pos[:, 0], 1], X[pos[:, 0], 2],
                c="r", marker="+", label="Admitted")
    plt.scatter(X[neg[:, 0], 1], X[neg[:, 0], 2],
                c="b", marker="x", label="Not admitted")

    # Plotting decision boundary
    u_vals = np.linspace(-1, 1.5, 50)
    v_vals = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u_vals), len(v_vals)))
    for i in range(len(u_vals)):
        for j in range(len(v_vals)):
            z[i, j] = mapFeaturePlot(u_vals[i], v_vals[j], 6) @ theta

    plt.contour(u_vals, v_vals, z.T, 0)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(loc=0)
    plt.show()


def main():
    DataPlot()
    LogisticRegressionTraining()
    RegularizedLogisticRegression()

main()
