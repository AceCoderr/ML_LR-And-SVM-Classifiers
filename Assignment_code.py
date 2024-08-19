import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier


# load dataset
col_names = ['x', 'y','out']
dataset = pd.read_csv("dataset.csv", header=None, names=col_names)
print(dataset.head())
X1=dataset.iloc[:,0]
X2=dataset.iloc[:,1]
X=np.column_stack((X1,X2))
Y=dataset.iloc[:,2]

#split the data
negative_mask = dataset['out'] == -1
positive_mask = dataset['out'] == 1
negative_data = dataset[negative_mask]
positive_data = dataset[positive_mask]
print(negative_data.head())
print(positive_data.head())

negative_X=negative_data.iloc[:,0]
negative_Y=negative_data.iloc[:,1]
positive_X=positive_data.iloc[:,0]
positive_Y=positive_data.iloc[:,1]

#plot the data
plt.scatter(positive_X, positive_Y, marker='+', label='+1', c='blue')  # + marker for target +1
plt.scatter(negative_X, negative_Y, marker='o', label='-1', c='red')  # o marker for target -1
plt.xlabel('Feature X1')
plt.ylabel('Feature X2')
plt.show()

#split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=16)

#train the model
logreg = LogisticRegression(random_state=16)

#fit the model with data 
logreg.fit(X,Y)

#find Feature importance
intercept = logreg.intercept_[0]
coefficients = logreg.coef_[0]
abs_coefficients = np.abs(coefficients)
print(intercept)
print(coefficients)         #coefficients with highest value will be more influenctial

y_pred = logreg.predict(X)

print(logreg.score(X,Y))


plt.scatter(X[:, 0], X[:, 1], c='green', cmap=plt.cm.Paired, marker='^',  label='Training Data')
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.Paired, marker='x',label='Predictions')
plt.xlabel('Feature X1')
plt.ylabel('Feature X2')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')

plt.legend()
plt.show()

#SVM

Penalty_Parameter_C = [0.001,1,100]
models = []
svcModels = {}
fig, axes = plt.subplots(1, len(Penalty_Parameter_C), figsize=(15, 5))
#implement LinearSVC

for i, C in enumerate(Penalty_Parameter_C):
    SvmClassifiers = LinearSVC(C=C)
    SvmClassifiers.fit(X_train,y_train)
    y_pred_svm=SvmClassifiers.predict(X_train)
    
    svcModels[f'SVM_C_{C}'] = SvmClassifiers
    models.append(SvmClassifiers)
    
    #data points
    ax = axes[i]
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,marker='*', cmap=plt.cm.Paired)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_svm,marker='x', cmap=plt.cm.Paired)
    ax.set_title(f'C = {C}')
    
    # Plot decision boundary
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = SvmClassifiers.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
plt.legend()
plt.show()
#Report model parameters
for model_name, model in svcModels.items():
    print(f"Model: {model_name}")
    print(f"C={model.C}")
    print(f"coefficients:{model.coef_[0]} ")
    print(f"intercept:{model.intercept_[0]}")
    print(f"Accuracy: {model.score(X_train, y_train)}")
    print("\n")

#Adding Squared x and y features to dataset
dataset['square_x'] = dataset['x']**2
dataset['square_y'] = dataset['y']**2

X1=dataset.iloc[:,0]
X2=dataset.iloc[:,1]
X3=dataset.iloc[:,3]
X4=dataset.iloc[:,4]
X=np.column_stack((X1,X2,X3,X4))
Y=dataset.iloc[:,2]

#train the model
logreg_squared = LogisticRegression(random_state=16)

#fit the model with data 
logreg_squared.fit(X,Y)

#find Feature importance
intercept = logreg_squared.intercept_[0]
coefficients = logreg_squared.coef_[0]
abs_coefficients = np.abs(coefficients)
print(intercept)
print(coefficients)         #coefficients with highest value will be more influenctial

y_pred_squared = logreg_squared.predict(X)

print(logreg_squared.score(X,Y))


plt.scatter(X[:, 0], X[:, 1], c='green', cmap=plt.cm.Paired, marker='^',  label='Training Data')
plt.scatter(X[:, 0], X[:, 1], c=y_pred_squared, cmap=plt.cm.Paired, marker='x',label='Predictions')
plt.xlabel('Feature X1')
plt.ylabel('Feature X2')


x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = logreg_squared.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel()**2,yy.ravel()**2])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')

plt.legend()
plt.show()

BaselineClassifier= DummyClassifier(strategy="most_frequent")
BaselineClassifier.fit(X,Y)
BaselinePredictor=BaselineClassifier.predict(X)

print(f"Accuracy of Logistic Regression - {logreg.score(X,Y)*100}")
print(f"Accuracy of Dummy Classifier - {BaselineClassifier.score(X,Y)*100}")
