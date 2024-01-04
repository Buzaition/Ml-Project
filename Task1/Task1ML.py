import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset (replace 'path_to_dataset' with your actual file path)
data_df = pd.read_csv('/E/MLTask1/plantco_pp.csv')



# define X & Y
x = data_df.drop ( [ 'PE'], axis = 1).values
y = data_df['PE'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state=0)


#1 Initialize Linear Regression model
linear_regression_model = LinearRegression()

# Train the model
linear_regression_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_regression_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



#2 Initialize Ridge Regression model
RR_model = Ridge(alpha = 1.0)

# Train the model
RR_model.fit(X_train, y_train)

# Make predictions
y_pred = RR_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



#3 Initialize Lasso Regression model
LassoR_model = Lasso(alpha=1.0)

# Train the model
LassoR_model.fit(X_train, y_train)

# Make predictions
y_pred = LassoR_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



#4 Initialize SVR model
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SVR model
svr_model = SVR(kernel='linear')  # You can adjust the kernel as needed

# Train the model
svr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svr_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



#5 Initialize Elastic Net Regression model
EN_model = ElasticNet(alpha = 1.0,l1_ratio=0.5)

# Train the model
EN_model.fit(X_train, y_train)

# Make predictions
y_pred = En_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# Plot the results (scatter plot of predicted vs. actual values)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()