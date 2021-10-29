import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

#Load the Dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Replacing missing values with monthly charge values
df.loc[df['TotalCharges']==' ','TotalCharges'] = df.loc[df['TotalCharges']==' ','MonthlyCharges']

#Converting the TotalCharges column to float64 dtype
df['TotalCharges'] = df['TotalCharges'].astype('float64')

#Dropping customerID column
df = df.drop('customerID', axis = 1)

#Feature Engineering
df['gender'] = df['gender'].apply(lambda x: 0 if x=='Female' else 1)
df['MultipleLines'] = df['MultipleLines'].apply(lambda x: 'No' if x == 'No phone service' else x)
df['InternetService'] = df['InternetService'].apply(lambda x: 0 if x == 'No' else 1).astype('int64')

no_net_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
              'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in no_net_cols:
    df[col] = df[col].apply(lambda x: 'No' if x == 'No internet service' else x)

df['loyalty'] =  df['Contract'].apply(lambda x: 0 if x == 'Month-to-month' else 1).astype('int64')
df = df.drop('Contract',axis = 1)

##Convert Yes/No columns to 1 or 0
y_n_cols = list(df.select_dtypes(include = 'object').columns)
y_n_cols.remove('PaymentMethod')

for col in y_n_cols:
    
    df[col] = df[col].apply(lambda x: 0 if x == 'No' else 1).astype('int64')

#Loading the model
model = pickle.load(open('knn_model.pkl','rb'))

#Drop TotalCharges column because of high multicolinearity with tenure and MonthlyCharges column
df = df.drop('TotalCharges', axis=1)

# Factorizing the 'Payment Method' column to make the category group names into numerical format
df['PaymentMethod'], _ = df['PaymentMethod'].factorize()


#Modeling

#Splitting the Data into X and y
X = df.drop('Churn', axis = 1)
y = df['Churn']

# Using Smote to fix Class Imbalance Issue
smote = SMOTE()

X_smote, y_smote = smote.fit_resample(X, y)

#Splitting the Data into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, random_state = 42)

#Scaling the Data
ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

#Fitting the KNN model with defined Hyper-Parameters
knn_tuned = KNeighborsClassifier(leaf_size= 10,
                                 metric = 'manhattan',
                                 n_neighbors = 97,
                                 weights = 'distance')

knn_tuned.fit(X_train_scaled, y_train)

#Pickle the model
pickle.dump(knn_tuned, open('knn_model.pkl','wb'))

#Making a prediction
print(model.predict([[0,0,1,0,28,1,1,1,0,0,1,1,1,1,1,0,104.8,0]]))
