import pandas as pd



df = pd.read_csv("data/raw/winequality-white.csv", sep=';')
print(df.head())
print(df.shape)
print(df.columns)

X = df.drop("quality", axis=1)
y = df["quality"]
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42  )

X_train_final,X_val,y_train_final,y_val = train_test_split(X_train,y_train,test_size=0.3, random_state=42 )  

lambdas = [0.001, 0.01, 0.1, 1, 10, 100]

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()

X_train_final = scaler.fit_transform(X_train_final)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

mse1 = []
mse2 = []

for l in lambdas:
    model = Ridge(alpha=l)
    model.fit(X_train_final, y_train_final)
    y_pred1 = model.predict(X_val)
    y_pred2 = model.predict(X_train_final)
    mse1.append(mean_squared_error(y_val, y_pred1))
    mse2.append(mean_squared_error(y_train_final, y_pred2))
    print(f"Lambda: {l:<10}, Val MSE: {mse1[-1]:.4f}, Train MSE: {mse2[-1]:.4f}")

plt.plot(lambdas, mse1, marker='o', label="Validation MSE")
plt.plot(lambdas, mse2, marker='s', label="Training MSE")

plt.xscale('log') # Car les lambdas sont en échelle exponentielle
plt.xlabel('Lambda (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Lambda for Ridge Regression')
plt.grid()
plt.legend()
plt.show()  

best_lambda = lambdas[mse1.index(min(mse1))]
print("Best lambda =", best_lambda)

#%%

from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path


alphas, coefs, _ = lasso_path(X_train_final, y_train_final)

for i in range(coefs.shape[0]):
    plt.plot(alphas, coefs[i], label=X.columns[i])

plt.xscale("log")
plt.gca().invert_xaxis()   # souvent on met grand -> petit
plt.xlabel("alpha")
plt.ylabel("coefficients")
plt.title("Lasso Path")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.show()