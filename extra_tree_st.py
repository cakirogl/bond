import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor, RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from flaml import AutoML
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#plt.rcParams["text.usetex"]=True
#plt.rcParams["text.latex.preamble"]=r"\usepackage{amsmath}\boldmath"
url = 'https://raw.githubusercontent.com/ccakiroglu/bond/main/Bond2.xlsm'
df = pd.read_excel(url, header=0, sheet_name='regression')
x, y = df.iloc[:, :-1], df.iloc[:, -1]
scaler=MinMaxScaler()
#x = pipeline.fit_transform(x)
#contaminations=np.arange(0.00001,0.50001,0.02)
#contaminations=[0.00001,0.02, 0.04, 0.06, 0.08,0.1, 0.18,0.30]
contaminations=[0.04]
for c in contaminations:
    model=IsolationForest(random_state=0, contamination=float(c));
    #model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(c),max_features=1.0)
    model.fit(x)
    df['scores']=model.decision_function(x)
    df['anomaly']=model.predict(x)
    anomaly_df=df.loc[df['anomaly']==-1]
    normal_df=df.loc[df['anomaly']!=-1]
    normal_data=normal_df.values
    anomaly_index=list(anomaly_df.index)
    #print(anomaly_index)
    x_normal, y_normal = normal_data[:, :-3], normal_data[:, -3]
    x_train, x_test, y_train, y_test = train_test_split(x_normal, y_normal, test_size=0.2, random_state=0)
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    #x_train = pipeline.fit_transform(x_train)
    #x_test = pipeline.transform(x_test)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #XGBModel=XGBRegressor(random_state=0)
    #XGBModel.fit(x_train, y_train)
    #CBModel=CatBoostRegressor(random_state=0, logging_level="Silent")
    #CBModel.fit(x_train, y_train)
    LGBMModel=LGBMRegressor(random_state=0, verbose=-1)
    LGBMModel.fit(x_train,y_train)
    #RFModel=RandomForestRegressor(random_state=0, verbose=0)
    #RFModel.fit(x_train,y_train)
    automl = AutoML()
    automl_settings = {
        "estimator_list": ["catboost"],
        "metric": "r2",
        "log_training_metric": True,
        "log_type": "all",
        "model_history": True,
        "task": "regression",
        "max_iter": 50,
        "time_budget": 7000,
        "log_file_name": "logs.txt"
    }
    #automl.fit(x_train, y_train, **automl_settings)
    #yhat_test = XGBModel.predict(x_test)
    #yhat_train = XGBModel.predict(x_train)
    #yhat_test = CBModel.predict(x_test)
    #yhat_train = CBModel.predict(x_train)
    yhat_test = LGBMModel.predict(x_test)
    yhat_train = LGBMModel.predict(x_train)
    #yhat_test = RFModel.predict(x_test)
    #yhat_train = RFModel.predict(x_train)
    #yhat_test = automl.predict(x_test)
    #yhat_train = automl.predict(x_train)
    print("Contamination = ",c);
    print("The number of anomalies:", len(anomaly_index));
    print('MSE train= ',mean_squared_error(y_train, yhat_train))
    print('RMSE train= ',np.sqrt(mean_squared_error(y_train, yhat_train)))
    print('MAE train= ',mean_absolute_error(y_train, yhat_train))
    print('R2 train:',r2_score(y_train, yhat_train))
    print('MSE test= ',mean_squared_error(y_test, yhat_test))
    print('RMSE test= ',np.sqrt(mean_squared_error(y_test, yhat_test)))
    print('MAE test= ',mean_absolute_error(y_test, yhat_test))
    print('R2 test:',r2_score(y_test, yhat_test))#original
    #scores = cross_val_score(CBmodel, x_train, y_train, cv=10)
    #print(scores)
f=open("y_test.txt", "w")
for c in y_test:
    f.write(str(c)+"\n")
f.close();
fig, ax=plt.subplots()
ax.scatter(yhat_train, y_train, color='blue',label='XGBoost train')
ax.scatter(yhat_test, y_test, color='red',label='XGBoost test')
#ax.scatter(yhat_train, y_train, color='seagreen',label=r'$\mathbf{CatBoost\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='coral',label=r'$\mathbf{CatBoost\text{ }test}$')
#ax.scatter(yhat_train, y_train, color='teal', label=r'$\mathbf{LightGBM\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='fuchsia', label=r'$\mathbf{LightGBM\text{ }test}$')
#ax.scatter(yhat_train, y_train, color='royalblue',label=r'$\mathbf{Random\text{ }Forest\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='gray',label=r'$\mathbf{Random\text{ }Forest\text{ }test}$')
#ax.set_xticks([20,30,40,50,60,70])
ax.set_xlabel('f,predicted [MPa]', fontsize=14)
ax.set_ylabel('f,test [MPa]}$', fontsize=14)
xmax=50;ymax=50;
xk=[0,xmax];yk=[0,ymax];ykPlus10Perc=[0,ymax*1.1];ykMinus10Perc=[0,ymax*0.9];
ax.tick_params(axis='x',labelsize=14)
ax.tick_params(axis='y',labelsize=14)
ax.plot(xk,yk, color='black')
ax.plot(xk,ykPlus10Perc, dashes=[2,2], color='black')
ax.plot(xk,ykMinus10Perc,dashes=[2,2], color='black')
ax.grid(True)
ratio=1.0
xmin,xmax=ax.get_xlim()
ymin,ymax=ax.get_ylim()
ax.set_aspect(ratio*np.abs((xmax-xmin)/(ymax-ymin)));

def linearRegr(x,a0,a1):
    return a0+a1 * np.array(x)

coeffs, covmat=curve_fit(f=linearRegr, xdata=np.concatenate((yhat_train,yhat_test)).flatten(),ydata=np.concatenate((y_train,y_test)).flatten())
print(f"a0={coeffs[0]}, a1={coeffs[1]}")
regr=linearRegr(xk,coeffs[0], coeffs[1])
ax.plot(xk,regr, label="y=-0.07+x")
plt.legend(loc='upper left',fontsize=12)
#plt.show()
st.pyplot(fig)