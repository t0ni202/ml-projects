#Toni Ogunkoya
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    # This transformer extracts day and hour from a datetime column and drops the original column
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['day_of_week'] = X_copy['date'].dt.dayofweek
        X_copy['trading_hour'] = X_copy['date'].dt.hour
        return X_copy.drop(columns=['date'])

trading_data_pkl = pd.read_pickle("/Users/toniogunkoya/Documents/code/ECE 310/appml-assignment1-dataset-v2.pkl")
trading_features = trading_data_pkl['X']
trading_labels = trading_data_pkl['y'].rename('CAD-pred')
trading_data_df = pd.concat([trading_features, trading_labels], axis=1)

# Split the data
train_set, test_set = train_test_split(trading_data_df, test_size=0.2, random_state=42)
X_train = train_set.copy()
X_test = test_set.copy()

y_train = X_train['CAD-pred'].copy()
X_train = train_set.drop('CAD-pred', axis=1)

y_test = X_test['CAD-pred'].copy()
X_test = X_test.drop('CAD-pred', axis=1)

trading_data_df_labels = trading_data_df["CAD-pred"].copy()
trading_data_df = trading_data_df.drop("CAD-pred", axis = 1)

# Create pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('std_scaler', StandardScaler())
])

num_attribs = trading_data_df.select_dtypes(include=[np.number]).columns.tolist()
cat_attribs = ['day_of_week', 'trading_hour']

pre_process = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

full_pipeline = Pipeline([
    ("date_parser", DateTimeExtractor()),
    ("processing", pre_process)
])

# Apply transformations using the pipeline
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)

# Train the model
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared,y_train)


# Evaluate the model
linPreds_tr=lin_reg.predict(X_train_prepared)
lin_rmse_tr=np.sqrt(mean_squared_error(y_train,linPreds_tr))
print(f"Train RMSE: {lin_rmse_tr}")


linPreds_test = lin_reg.predict(X_test_prepared)
lin_rmse_test=np.sqrt(mean_squared_error(y_test,linPreds_test))
print(f"Test RMSE: {lin_rmse_test}")

dump(full_pipeline, 'pipeline1.pkl')
dump(forest_reg, 'model1.pkl')
