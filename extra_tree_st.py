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
#st.set_page_config(layout="wide")
#st.write("### Bond strength prediction")
model_selector = st.selectbox('**Bond strength prediction model**', ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest'])
input_container = st.container()
output_container = st.container()
ic1,ic2, ic3=input_container.columns(3)
with ic1:
    wc=st.number_input("**Water to cement ratio:**",min_value=0.1,max_value=0.9,step=0.05,value=0.4)
    rca_perc=st.number_input("**Replacement percentage of RCA:**",min_value=0.0,max_value=100.0,step=10.0,value=50.0)
    rebar_type=st.selectbox("**Rebar type**", ["Mild steel", "BFRP", "GFRP", "UHSS", "CFRP"])
    if rebar_type=="Mild steel":
        rebar_type=1
    elif rebar_type=="BFRP":
        rebar_type=3
    elif rebar_type=="GFRP":
        rebar_type=2
    elif rebar_type=="UHSS":
        rebar_type=5
    elif rebar_type=="CFRP":
        rebar_type=4

    rebar_e=st.number_input("**Rebar E module [GPa]**", min_value=20.0, max_value=250.0, value=50.0, step=10.0)
    embed_len=st.number_input("**Embedment length [mm]**", min_value=20.0, max_value=500.0, value=150.0, step=10.0)
with ic2:
    abs_cap=st.number_input("**Absorption capacity [%]:**", min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    fc=st.number_input("**Compressive strength [MPa]:**", min_value=10.0, max_value=100.0, value=25.0, step=5.0)
    rebar_surface=st.selectbox("**Rebar surface**", ["Deformed", "Sand Coated", "Plain"])
    if rebar_surface=="Deformed":
        rebar_surface=2
    elif rebar_surface=="Sand Coated":
        rebar_surface=3
    elif rebar_surface=="Plain":
        rebar_surface=1
    rebar_d=st.number_input("**Rebar diameter [mm]**", min_value=8.0, max_value=25.2, value=10.0, step=1.0)
    spec_type=st.selectbox("**Specimen type**", ["Beam", "Cube", "Cylinder"])
    if spec_type=="Beam":
        spec_type=2
    elif spec_type=="Cube":
        spec_type=1
    elif spec_type=="Cylinder":
        spec_type=3
with ic3:
    age=st.number_input("**Age [Days]:**", min_value=5.0, max_value=365.0, value=28.0, step=1.0)
    cover=st.text_input("**Cover [mm]:**")
    fy=st.number_input("**Rebar yield strength [MPa]:**", min_value=200.0, max_value=3000.0, value=500.0, step=10.0)
    conf_effect=st.selectbox("**Confinement effect**", ["Yes", "No"])
    if conf_effect=="Yes":
        conf_effect=2
    elif conf_effect=="No":
        conf_effect=1
    
new_sample=np.array([[wc, rca_perc, abs_cap, fc, age, cover, rebar_type, rebar_surface, fy, rebar_e, rebar_d, conf_effect, embed_len, spec_type]],dtype=object)
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
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    if model_selector=='LightGBM':
        model=LGBMRegressor(random_state=0, verbose=-1)
        model.fit(x_train,y_train)
        train_color="teal";test_color="fuchsia";eqn="-1.21+1.07x"
    elif model_selector=='XGBoost':
        model=XGBRegressor(random_state=0)
        model.fit(x_train, y_train)
        train_color="blue";test_color="red";eqn="-0.07+x"
    elif model_selector=='CatBoost':
        model=CatBoostRegressor(random_state=0, logging_level="Silent")
        model.fit(x_train, y_train)
        train_color="seagreen";test_color="coral";eqn="-0.9+1.05x"
    elif model_selector=='Random Forest':
        model=RandomForestRegressor(random_state=0, verbose=0)
        model.fit(x_train,y_train)
        train_color="royalblue";test_color="gray";eqn="-1.57+1.09x"
    yhat_test = model.predict(x_test)
    yhat_train = model.predict(x_train)
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
    #col1, col2, col3 = output_container.columns(3)
    #with col1:
        #st.write("**Training set**")
        #st.write(f"Contamination = {c}")
        #st.write(f"The number of anomalies = {len(anomaly_index)}")
        #st.write(f"RMSE = {np.sqrt(mean_squared_error(y_train, yhat_train))}")
        #st.write(f"MAE = {mean_absolute_error(y_train, yhat_train)}")
        #st.write(f"R2 = {r2_score(y_train, yhat_train)}")
    #with col2:
        #st.write("**Test set**")
        #st.write(f"RMSE = {np.sqrt(mean_squared_error(y_test, yhat_test))}")
        #st.write(f"MAE = {mean_absolute_error(y_test, yhat_test)}")
        #st.write(f"R2 = {r2_score(y_test, yhat_test)}")
    

fig, ax=plt.subplots()
#with col3:
with ic3:
    st.write(f"**Bond strength = **{model.predict(new_sample)[0]:.2f}** MPa**")
    #ax.scatter(yhat_train, y_train, color=train_color,label=model_selector+' train')
    #ax.scatter(yhat_test, y_test, color=test_color,label=model_selector+' test')
#ax.scatter(yhat_train, y_train, color='blue',label='XGBoost train')
#ax.scatter(yhat_test, y_test, color='red',label='XGBoost test')
#ax.scatter(yhat_train, y_train, color='seagreen',label=r'$\mathbf{CatBoost\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='coral',label=r'$\mathbf{CatBoost\text{ }test}$')
#ax.scatter(yhat_train, y_train, color='teal', label=r'$\mathbf{LightGBM\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='fuchsia', label=r'$\mathbf{LightGBM\text{ }test}$')
#ax.scatter(yhat_train, y_train, color='royalblue',label=r'$\mathbf{Random\text{ }Forest\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='gray',label=r'$\mathbf{Random\text{ }Forest\text{ }test}$')
#ax.set_xticks([20,30,40,50,60,70])
    #ax.set_xlabel('f,predicted [MPa]', fontsize=14)
    #ax.set_ylabel('f,test [MPa]', fontsize=14)
    #xmax=50;ymax=50;
    #xk=[0,xmax];yk=[0,ymax];ykPlus10Perc=[0,ymax*1.1];ykMinus10Perc=[0,ymax*0.9];
    #ax.tick_params(axis='x',labelsize=14)
    #ax.tick_params(axis='y',labelsize=14)
    #ax.plot(xk,yk, color='black')
    #ax.plot(xk,ykPlus10Perc, dashes=[2,2], color='black')
    #ax.plot(xk,ykMinus10Perc,dashes=[2,2], color='black')
    #ax.grid(True)
    #ratio=1.0
    #xmin,xmax=ax.get_xlim()
    #ymin,ymax=ax.get_ylim()
    #ax.set_aspect(ratio*np.abs((xmax-xmin)/(ymax-ymin)));

    def linearRegr(x,a0,a1):
        return a0+a1 * np.array(x)

    coeffs, covmat=curve_fit(f=linearRegr, xdata=np.concatenate((yhat_train,yhat_test)).flatten(),ydata=np.concatenate((y_train,y_test)).flatten())
    print(f"a0={coeffs[0]}, a1={coeffs[1]}")
    #regr=linearRegr(xk,coeffs[0], coeffs[1])
    #ax.plot(xk,regr, label=eqn)
    plt.legend(loc='upper left',fontsize=12)
    #plt.show()
    #st.pyplot(fig)