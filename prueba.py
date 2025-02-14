import numpy as np
import seaborn as sns
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
# print(np.mean(X),np.mean(y),np.sum(X))

w=(np.sum(X * y)-np.mean(y))/(np.sum(X * X)-(np.mean(X)*np.sum(X)))
# print(w) 

w2= np.cov(X, y, bias=True)[0, 1] / np.var(X)#NumPy calcula la matriz de covarianza, pero por defecto divide por n-1 en lugar de n

w1=np.cov(X,y)[0][1]/np.var(X)
# print(w1)
# print(w2)
# # print(w1)
# print(np.cov(X,y))
# print(np.var(X))

b=np.mean(y)-w2*np.mean(X)
# print(b)


np.random.seed(42)
X = np.random.rand(100, 3)
true_coefficients = np.array([2.5, -1.5, 3.0])
true_intercept = 1.0
y = true_intercept + X.dot(true_coefficients) + np.random.normal(0, 0.1, 100)
 # Añadir una columna de unos para el intercepto (b_0)
X_b = np.c_[np.ones((X.shape[0], 1)), X]
#np.ones((X.shape[0], 1)) crea una matriz con X.shape[0] filas, que es el numero de filas de la matriz X y 1 columna , llena de unos
# print(X)
# print()
# print(X_b)
# print()
w= np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(w)
# print(w[-1])
intercept=w[0]
coef=w[1:]
# print(intercept, coef)

# print(f"Esto es X {X} y esto son los coeficientesb {coef}")
# print(X*coef)
# print(intercept + X*coef)

anscombe = sns.load_dataset("anscombe")
print(anscombe,type(anscombe))
print(list(anscombe.index))
datasets=anscombe["dataset"].unique()
for dataset in datasets:
    data = anscombe[anscombe["dataset"] == dataset]
    # print(f"Los datos son {data}")
    X=np.array(data["x"])
    y=np.array(data["y"])
    # print(X,y)