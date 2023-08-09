import streamlit as st
import streamlit.components.v1 as components
from IPython.core.display import display, HTML
import pandas as pd
from streamlit_option_menu import option_menu 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from os import getenv
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import f1_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import IsolationForest
from  sklearn import svm
from sklearn.neural_network import MLPClassifier
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed


st.set_page_config(page_title="Fares Ghezal thesis", page_icon="https://math.bme.hu/~marcessz/img/bme_favicon.ico", layout="wide")

def hide_anchor_link():
    st.markdown(
        body="""
        <style>
            h1 > div > a {
                display: none;
            }
            h2 > div > a {
                display: none;
            }
            h5 > div > a {
                display: none;
            }
            h4 > div > a {
                display: none;
            }
            h5 > div > a {
                display: none;
            }
            h6 > div > a {
                display: none;
            }
        </style>
        """,
         unsafe_allow_html=True,
)
def page1():  
  selected2 = option_menu(
    menu_title=None,
    options=["traffic data","accident data"],
    icons=["speedometer", "car-front-fill"], 
    orientation="horizontal",
  )
  trafic_data=pd.read_parquet("traffic_data.parquet")
  incident=pd.read_csv("accident2.csv")
  if selected2 == "traffic data":
    trafic_data["timestamp"] = pd.to_datetime(trafic_data["timestamp"])
    trafic_data['month'] = trafic_data['timestamp'].dt.month
    trafic_data['day'] = trafic_data['timestamp'].dt.day
    trafic_data['hour'] = trafic_data['timestamp'].dt.hour
    trafic_data['minute'] = trafic_data['timestamp'].dt.minute
    trafic_data['dayofweek'] = trafic_data['timestamp'].dt.dayofweek 
    st.markdown('The Traffic Dataset contains 5-minute aggregation traffic data from District 3 (Sacramento area) for a one year period from January 1, 2016 to December 31, 2016. Incident data were recorded by California Highway Patrol.')
    figSize = (13,5)
    col1, col2 , col3 = st.columns(3)
    col4, col5 , col6= st.columns(3)
    col7, col8 , col9= st.columns(3)
    col01,col02 ,col03= st.columns(3)
    col1.markdown("<h5 style='text-align: center'>speed average over the monthes</h5>", unsafe_allow_html=True)
    col2.markdown("<h5 style='text-align: center'>occ average over the monthes</h5>", unsafe_allow_html=True)
    col3.markdown("<h5 style='text-align: center'>flow average over the monthes</h5>", unsafe_allow_html=True)
    col4.markdown("<h5 style='text-align: center'>speed average over days of the month</h5>", unsafe_allow_html=True)
    col5.markdown("<h5 style='text-align: center'>occ average over days of the month</h5>", unsafe_allow_html=True)
    col6.markdown("<h5 style='text-align: center'>flow average over days of the month</h5>", unsafe_allow_html=True)
    col7.markdown("<h5 style='text-align: center'>speed average over days of the week</h5>", unsafe_allow_html=True)
    col8.markdown("<h5 style='text-align: center'>occ average over days of the week</h5>", unsafe_allow_html=True)
    col9.markdown("<h5 style='text-align: center'>flow average over days of the week</h5>", unsafe_allow_html=True)
    col02.markdown("<h5 style='text-align: center'>correlation between features</h5>", unsafe_allow_html=True)

    speed_average=trafic_data.groupby(['month'])['speed'].mean().reset_index()
    fig = px.line(speed_average, x="month", y="speed")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col1.plotly_chart(fig, use_container_width=True)

    occ_average=trafic_data.groupby(['month'])['occ'].mean().reset_index()
    fig = px.line(occ_average, x="month", y="occ")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col2.plotly_chart(fig, use_container_width=True)

    flow_average=trafic_data.groupby(['month'])['flow'].mean().reset_index()
    fig = px.line(flow_average, x="month", y="flow")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col3.plotly_chart(fig, use_container_width=True)

    speed_average=trafic_data.groupby(['day'])['speed'].mean().reset_index()
    fig = px.line(speed_average, x="day", y="speed")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col4.plotly_chart(fig, use_container_width=True)

    occ_average=trafic_data.groupby(['day'])['occ'].mean().reset_index()
    fig = px.line(occ_average, x="day", y="occ")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col5.plotly_chart(fig, use_container_width=True)

    flow_average=trafic_data.groupby(['day'])['flow'].mean().reset_index()
    fig = px.line(flow_average, x="day", y="flow")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col6.plotly_chart(fig, use_container_width=True)

    speed_average=trafic_data.groupby(['dayofweek'])['speed'].mean().reset_index()
    fig = px.line(speed_average, x="dayofweek", y="speed")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col7.plotly_chart(fig, use_container_width=True)

    occ_average=trafic_data.groupby(['dayofweek'])['occ'].mean().reset_index()
    fig = px.line(occ_average, x="dayofweek", y="occ")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col8.plotly_chart(fig, use_container_width=True)

    flow_average=trafic_data.groupby(['dayofweek'])['flow'].mean().reset_index()
    fig = px.line(flow_average, x="dayofweek", y="flow")
    fig.update_layout(yaxis={'title': ""})
    fig.update_layout(xaxis={'title': ""})
    col9.plotly_chart(fig, use_container_width=True)
    x = trafic_data
    alpha = x.corr(numeric_only = True).columns
    plt.rcParams["axes.grid"] = False
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(x.corr(method='pearson'), cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(cax)
    plt.xticks(rotation='vertical')
    ax.set_xticks(np.arange(len(alpha)))
    ax.set_yticks(np.arange(len(alpha)))
    ax.set_xticklabels([' ']+alpha)
    ax.set_yticklabels([' ']+alpha)
    ax.tick_params(labelsize = 18)
    col02.pyplot(fig, use_container_width=True)
  if selected2 == "accident data":
    figSize = (13,5)
    st.markdown('The Traffic Incident Dataset contains 5-minute aggregation traffic data from District 3 (Sacramento area) for a one year period from January 1, 2016 to December 31, 2016. Incident data were recorded by California Highway Patrol.')
    col1, col2 , col3 = st.columns(3)
    col4, col5 , col6= st.columns(3)
    col7, col8 , col9= st.columns(3)
    col01,col02 ,col03= st.columns(3)
    col1.markdown("<h5 style='text-align: center'>accident by month</h5>", unsafe_allow_html=True)
    col2.markdown("<h5 style='text-align: center'>accident by day of week</h5>", unsafe_allow_html=True)
    col3.markdown("<h5 style='text-align: center'>accident by hour</h5>", unsafe_allow_html=True)
    col4.markdown("<h5 style='text-align: center'>average speed depending on type of accident</h5>", unsafe_allow_html=True)
    col5.markdown("<h5 style='text-align: center'>average occ depending on type of accident</h5>", unsafe_allow_html=True)
    col6.markdown("<h5 style='text-align: center'>average flow depending on type of accident</h5>", unsafe_allow_html=True)
    col02.markdown("<h5 style='text-align: center'>correlation between features</h5>", unsafe_allow_html=True)
    trafic_data["timestamp"] = pd.to_datetime(trafic_data["timestamp"])
    trafic_data['quarter'] = trafic_data['timestamp'].dt.to_period('Q')
    incident["timestamp"] = pd.to_datetime(incident["timestamp"])
    incident['quarter'] = incident['timestamp'].dt.to_period('Q')
    trafic_1_quarter = trafic_data[trafic_data['quarter'] == "2016Q1"]
    incident_1_quarter = incident[incident['quarter'] == "2016Q1"]
    x=trafic_1_quarter.rename(columns={"station_id": "down_id"})
    x=x.rename(columns={"speed": "down_speed"})
    x=x.rename(columns={"occ": "down_occ"})
    x=x.rename(columns={"flow": "down_flow"})
    merged = pd.merge(x, incident_1_quarter, how="left", on=["timestamp","down_id"])
    x=trafic_1_quarter.rename(columns={"station_id": "up_id"})
    x=x.rename(columns={"speed": "up_speed"})
    x=x.rename(columns={"occ": "up_occ"})
    x=x.rename(columns={"flow": "up_flow"})
    merged = pd.merge(merged, x, how="left", on=["timestamp","up_id"])
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged["up_id"] = merged["up_id"].astype("category")
    merged["down_id"] = merged["down_id"].astype("category")
    merged['month'] = merged['timestamp'].dt.month
    merged['day'] = merged['timestamp'].dt.day
    merged['hour'] = merged['timestamp'].dt.hour
    merged['minute'] = merged['timestamp'].dt.minute
    merged['dayofweek'] = merged['timestamp'].dt.dayofweek
    merged = merged.drop('quarter_x', axis=1)
    merged = merged.drop('quarter_y', axis=1)
    merged = merged.drop('quarter', axis=1)
    merged = merged.drop('timestamp', axis=1)

    x=merged.groupby(['month', 'description']).size().to_frame().reset_index()
    fig = px.bar(x, x="month", y=[0], color="description", title="")
    col1.plotly_chart(fig, use_container_width=True)

    x=merged.groupby(['dayofweek', 'description']).size().to_frame().reset_index()
    fig = px.bar(x, x="dayofweek", y=[0], color="description", title="")
    col2.plotly_chart(fig, use_container_width=True)

    x=merged.groupby(['hour', 'description']).size().to_frame().reset_index()
    fig = px.bar(x, x="hour", y=[0], color="description", title="")
    col3.plotly_chart(fig, use_container_width=True)

    merged['description'] = merged['description'].fillna("no accident")
    x=merged.groupby(['description']).mean().reset_index()
    fig = go.Figure(data=[
    go.Bar(name='down_speed',opacity=0.8,marker_color='#4C78A8',x=x["description"], y=x["down_speed"]),
    go.Bar(name='up_speed',opacity=0.8,marker_color='#F58518',x=x["description"], y=x["up_speed"]),
  ])
    fig.update_layout(
       title={
        'text': "",
        'y':0.08,
        'x':0.44,
        'xanchor': 'center',
        'yanchor': 'bottom'})
    col4.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(data=[
        go.Bar(name='down_occ',opacity=0.8,marker_color='#4C78A8',x=x["description"], y=x["down_occ"]),
        go.Bar(name='up_occ',opacity=0.8,marker_color='#F58518',x=x["description"], y=x["up_occ"]),
      ])
    fig.update_layout(
      title={
            'text': "",
            'y':0.08,
            'x':0.44,
            'xanchor': 'center',
            'yanchor': 'bottom'})
    col5.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(data=[
        go.Bar(name='down_flow',opacity=0.8,marker_color='#4C78A8',x=x["description"], y=x["down_flow"]),
        go.Bar(name='up_flow',opacity=0.8,marker_color='#F58518',x=x["description"], y=x["up_flow"]),
      ])
    fig.update_layout(
      title={
            'text': "",
            'y':0.08,
            'x':0.44,
            'xanchor': 'center',
            'yanchor': 'bottom'})
    col6.plotly_chart(fig, use_container_width=True)
    x = incident
    alpha = x.corr(numeric_only = True).columns
    plt.rcParams["axes.grid"] = False
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(x.corr(method='pearson'), cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    fig.colorbar(cax)
    plt.xticks(rotation='vertical')
    ax.set_xticks(np.arange(len(alpha)))
    ax.set_yticks(np.arange(len(alpha)))
    ax.set_xticklabels([' ']+alpha)
    ax.set_yticklabels([' ']+alpha)
    ax.tick_params(labelsize = 18)
    col02.pyplot(fig, use_container_width=True)
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

def page2():
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.markdown('this page allows you to try the various models (training and testing) on either the default or your own dataset.')
  st.markdown('the traffic data must be on csv format and include the following features: timestamp, station_id, speed,	occ and	flow')
  st.markdown('the accident data must be on csv format and include the following features: timestamp, up_id and	duration	')
  st.markdown('Sadly, Numenta htm model can not be implemented as it is not available as a pip package and including the whole repository is not cost efficient ')
  col001, col002  = st.columns(2)
  col01, col02  = st.columns(2)
  
  trafic_data=pd.read_parquet("traffic_data.parquet")
  incident=pd.read_csv("accident2.csv")
  uploaded_file = col001.file_uploader("upload traffic data",type=["csv"])
  if uploaded_file is not None:
    trafic_data = pd.read_csv(uploaded_file)
  col001.markdown(hide_table_row_index, unsafe_allow_html=True)
  col001.write(trafic_data.head())
  uploaded_file2 = col002.file_uploader("upload accident data",type=["csv"])
  if uploaded_file2 is not None:
    incident = pd.read_csv(uploaded_file)
  col002.markdown(hide_table_row_index, unsafe_allow_html=True)
  col002.write(incident.head())
  trafic_data["timestamp"] = pd.to_datetime(trafic_data["timestamp"])
  incident["timestamp"] = pd.to_datetime(incident["timestamp"])
  import re
  collect_numbers = lambda x : [int(i) for i in re.split("[^0-9]", x) if i != ""]
  numbers1 = col01.text_input("PLease enter accident indexes for training", value="31,35,36,45,50,52,55,56,60,67,68,69,76,78,84,90,91")
  acident_index_train=collect_numbers(numbers1)
  numbers2 = col02.text_input("PLease enter accident indexes for testing", value="162,166,170,171,173,174,176,177,180,181,184,187,188")
  acident_index_test=collect_numbers(numbers2)

  import time
  def model1():
    start = time.time()
    def modified_z_score(x,y):
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result["speed_modified_z_result"] =  0.6745* (result['speed'] - np.median(result['speed']))/ np.median(np.abs(result['speed']-np.median(result['speed'])))
        result.loc[np.abs(result['speed_modified_z_result']) < y, 'speed_modified_z_result'] = 0
        result.loc[np.abs(result['speed_modified_z_result']) > y, 'speed_modified_z_result'] = 1
        result["occ_modified_z_result"] =  0.6745* (result['occ'] - np.median(result['occ']))/ np.median(np.abs(result['occ']-np.median(result['occ'])))
        result.loc[np.abs(result['occ_modified_z_result']) < y, 'occ_modified_z_result'] = 0
        result.loc[np.abs(result['occ_modified_z_result']) > y, 'occ_modified_z_result'] = 1
        result["flow_modified_z_result"] =  0.6745* (result['flow'] - np.median(result['flow']))/ np.median(np.abs(result['flow']-np.median(result['flow'])))
        result.loc[np.abs(result['flow_modified_z_result']) < y, 'flow_modified_z_result'] = 0
        result.loc[np.abs(result['flow_modified_z_result']) > y, 'flow_modified_z_result'] = 1
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        return(result)
    def dr_score(list,y):
        No_of_congestions_detected=0
        No_of_true_congestions=len(list)
        for x in list:
            result=modified_z_score(x,y)
            if result.query('1== speed_modified_z_result and 1 ==current').shape[0]>1:
                No_of_congestions_detected+=1
        return(No_of_congestions_detected/No_of_true_congestions)
    col11.write(f"The DR score is {dr_score(acident_index_test,2.6)*100:.1f}%")  
    def far(list,y):
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        for x in list:
            result=modified_z_score(x,y)
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== speed_modified_z_result and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== speed_modified_z_result and 0 ==current').shape[0]
        return(No_of_false_alarms_signals/No_of_non_congestion_instances)
                
    col11.write(f"The FAR score is {far(acident_index_test,2.6)*100:.2f}%") 
    def MTTD(list,y):
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=modified_z_score(x,y)
            if result.query('1== speed_modified_z_result and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.speed_modified_z_result == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        return(sum_of_time/No_of_congestions_detected)        
    col22.write(f"The MTTD score is {MTTD(acident_index_test,2.6):.2f} minutes") 
    et = time.time()
    col22.write(f"The total execution time is {( et - start)/60 :.2f} minutes") 
    result=modified_z_score(acident_index_test[0],2.6)
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["speed_modified_z_result"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)

  def model2():
    train=pd.DataFrame()
    for x in acident_index_train:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        train = pd.concat([train, result], axis=0)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train['day'] = train['timestamp'].dt.day
    train['hour'] = train['timestamp'].dt.hour
    train['minute'] = train['timestamp'].dt.minute
    train['dayofweek'] = train['timestamp'].dt.dayofweek
    train=train.drop(['timestamp'], axis=1)
    test=pd.DataFrame()
    for x in acident_index_test:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        test = pd.concat([test, result], axis=0)
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test['day'] = test['timestamp'].dt.day
    test['hour'] = test['timestamp'].dt.hour
    test['minute'] = test['timestamp'].dt.minute
    test['dayofweek'] = test['timestamp'].dt.dayofweek
    test=test.drop(['timestamp'], axis=1)
    X_train=train[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    X_test=test[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    y_train=train[['current']]
    y_test = test[['current']] 
    df = pd.concat([test, train], axis=0)
    transform = list(df.dtypes[df.dtypes != 'object'].index.values) 
    transform.remove('current')
    numerical = Pipeline(steps=[
      
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(2))])

    column_preprocessor = ColumnTransformer(
        transformers=[
          ('num', numerical, transform)])
    clf = Pipeline(steps=[('preprocessor', column_preprocessor),
                          ('classifier', svm.SVC())])
    start = time.time()
    param_dict = { 
        'classifier__C': [ 1,1.2,1.4],
        'classifier__gamma':[1, 0.1, 0.01,0.005, 0.001, 0.0001], 
        'classifier__degree':[0, 1, 2, 3],
        'classifier__kernel': ['rbf']
    }

    grid = GridSearchCV(clf, param_dict, cv=3, verbose=0, n_jobs=-1)
    best_model = grid.fit(X_train, y_train.values.ravel())
    neigh = best_model.best_estimator_
    neigh.fit(X_train, y_train) 
    def svm_result(x):
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result['day'] = result['timestamp'].dt.day
        result['hour'] = result['timestamp'].dt.hour
        result['minute'] = result['timestamp'].dt.minute
        result['dayofweek'] = result['timestamp'].dt.dayofweek
        res = neigh.predict(result[['speed','occ','flow','station_id','day','hour','minute','dayofweek']])
        result["svm_result"] = res
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        return(result)
    def dr_score(list):
        No_of_congestions_detected=0
        No_of_true_congestions=len(list)
        for x in list:
            result=svm_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                No_of_congestions_detected+=1
        return(No_of_congestions_detected/No_of_true_congestions)
    col11.write(f"The DR score is {dr_score(acident_index_test)*100:.1f}%")
    def far(list):
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        for x in list:
            result=svm_result(x)
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== svm_result and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== svm_result and 0 ==current').shape[0]
        return(No_of_false_alarms_signals/No_of_non_congestion_instances)
    col11.write(f"The FAR score is {far(acident_index_test)*100:.2f}%") 
    def MTTD(list):
      No_of_congestions_detected=0
      sum_of_time=0
      for x in list:
          result=svm_result(x)
          if result.query('1== svm_result and 1 ==current').shape[0]>1:
              sum_of_time+=(result[(result.svm_result == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
              No_of_congestions_detected+=1
      return(sum_of_time/No_of_congestions_detected)  
    col22.write(f"The MTTD score is {MTTD(acident_index_test):.2f} minutes") 
    et = time.time()
    col22.write(f"The total execution time is {( et - start)/60 :.2f} minutes") 
    result=svm_result(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["svm_result"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)

  def model3():
    train=pd.DataFrame()
    for x in acident_index_train:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        train = pd.concat([train, result], axis=0)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train['day'] = train['timestamp'].dt.day
    train['hour'] = train['timestamp'].dt.hour
    train['minute'] = train['timestamp'].dt.minute
    train['dayofweek'] = train['timestamp'].dt.dayofweek
    train=train.drop(['timestamp'], axis=1)
    test=pd.DataFrame()
    for x in acident_index_test:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        test = pd.concat([test, result], axis=0)
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test['day'] = test['timestamp'].dt.day
    test['hour'] = test['timestamp'].dt.hour
    test['minute'] = test['timestamp'].dt.minute
    test['dayofweek'] = test['timestamp'].dt.dayofweek
    test=test.drop(['timestamp'], axis=1)
    X_train=train[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    X_test=test[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    y_train=train[['current']]
    y_test = test[['current']] 
    df = pd.concat([test, train], axis=0)
    transform = list(df.dtypes[df.dtypes != 'object'].index.values) 
    transform.remove('current')
    numerical = Pipeline(steps=[
      
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(2))])

    column_preprocessor = ColumnTransformer(
        transformers=[
          ('num', numerical, transform)])
    clf = Pipeline(steps=[('preprocessor', column_preprocessor),
                          ('classifier', RandomForestClassifier())])
    start = time.time()
    param_dict = { 
        'classifier__criterion': ['gini', 'entropy', 'log_loss'],
        'classifier__max_depth': [None, 90, 100, 110],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__n_estimators': [100, 200, 300, 1000]
    }

    grid = GridSearchCV(clf, param_dict, cv=3, verbose=0, n_jobs=-1)
    best_model = grid.fit(X_train, y_train.values.ravel())
    neigh = best_model.best_estimator_
    neigh.fit(X_train, y_train) 
    def rf_result(x):
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result['day'] = result['timestamp'].dt.day
        result['hour'] = result['timestamp'].dt.hour
        result['minute'] = result['timestamp'].dt.minute
        result['dayofweek'] = result['timestamp'].dt.dayofweek
        res = neigh.predict(result[['speed','occ','flow','station_id','day','hour','minute','dayofweek']])
        result["svm_result"] = res
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        return(result)
    def dr_score(list):
        No_of_congestions_detected=0
        No_of_true_congestions=len(list)
        for x in list:
            result=rf_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                No_of_congestions_detected+=1
        return(No_of_congestions_detected/No_of_true_congestions)
    col11.write(f"The DR score is {dr_score(acident_index_test)*100:.1f}%")
    def far(list):
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        for x in list:
            result=rf_result(x)
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== svm_result and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== svm_result and 0 ==current').shape[0]
        return(No_of_false_alarms_signals/No_of_non_congestion_instances)   
    col11.write(f"The FAR score is {far(acident_index_test)*100:.2f}%")       
    def MTTD(list):
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=rf_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.svm_result == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        return(sum_of_time/No_of_congestions_detected)  
    col22.write(f"The MTTD score is {MTTD(acident_index_test):.2f} minutes") 
    et = time.time()
    col22.write(f"The total execution time is {( et - start)/60 :.2f} minutes") 
    result=rf_result(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["svm_result"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)

  def model4():
    train=pd.DataFrame()
    for x in acident_index_train:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        train = pd.concat([train, result], axis=0)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train['day'] = train['timestamp'].dt.day
    train['hour'] = train['timestamp'].dt.hour
    train['minute'] = train['timestamp'].dt.minute
    train['dayofweek'] = train['timestamp'].dt.dayofweek
    train=train.drop(['timestamp'], axis=1)
    test=pd.DataFrame()
    for x in acident_index_test:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        test = pd.concat([test, result], axis=0)
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test['day'] = test['timestamp'].dt.day
    test['hour'] = test['timestamp'].dt.hour
    test['minute'] = test['timestamp'].dt.minute
    test['dayofweek'] = test['timestamp'].dt.dayofweek
    test=test.drop(['timestamp'], axis=1)
    X_train=train[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    X_test=test[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    y_train=train[['current']]
    y_test = test[['current']] 
    df = pd.concat([test, train], axis=0)
    transform = list(df.dtypes[df.dtypes != 'object'].index.values) 
    transform.remove('current')
    numerical = Pipeline(steps=[
      
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(2))])

    column_preprocessor = ColumnTransformer(
        transformers=[
          ('num', numerical, transform)])
    clf = Pipeline(steps=[('preprocessor', column_preprocessor),
                          ('classifier', MLPClassifier())])
    start = time.time()
    param_dict = { 
        'classifier__solver': ['adam'],
        'classifier__learning_rate_init': [0.0001],
        'classifier__max_iter': [300],
        'classifier__hidden_layer_sizes': [(500, 400, 300, 200, 100), (400, 400, 400, 400, 400), (300, 300, 300, 300, 300)],
        'classifier__activation': ['logistic', 'tanh', 'relu'],
        'classifier__alpha': [0.0001, 0.001, 0.005],
        'classifier__early_stopping': [True]
    }

    grid = GridSearchCV(clf, param_dict, cv=3, verbose=0, n_jobs=-1)
    best_model = grid.fit(X_train, y_train.values.ravel())
    neigh = best_model.best_estimator_
    neigh.fit(X_train, y_train) 
    def ann_result(x):
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result['day'] = result['timestamp'].dt.day
        result['hour'] = result['timestamp'].dt.hour
        result['minute'] = result['timestamp'].dt.minute
        result['dayofweek'] = result['timestamp'].dt.dayofweek
        res = neigh.predict(result[['speed','occ','flow','station_id','day','hour','minute','dayofweek']])
        result["svm_result"] = res
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        return(result)
    def dr_score(list):
        No_of_congestions_detected=0
        No_of_true_congestions=len(list)
        for x in list:
            result=ann_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                No_of_congestions_detected+=1
        return(No_of_congestions_detected/No_of_true_congestions)
    col11.write(f"The DR score is {dr_score(acident_index_test)*100:.1f}%")
    def far(list):
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        for x in list:
            result=ann_result(x)
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== svm_result and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== svm_result and 0 ==current').shape[0]
        return(No_of_false_alarms_signals/No_of_non_congestion_instances)   
    col11.write(f"The FAR score is {far(acident_index_test)*100:.2f}%")         
    def MTTD(list):
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=ann_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.svm_result == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        return(sum_of_time/No_of_congestions_detected)  
    col22.write(f"The MTTD score is {MTTD(acident_index_test):.2f} minutes") 
    et = time.time()
    col22.write(f"The total execution time is {( et - start)/60 :.2f} minutes") 
    result=ann_result(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["svm_result"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)

  def model5():
    train=pd.DataFrame()
    for x in acident_index_train:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        train = pd.concat([train, result], axis=0)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train['day'] = train['timestamp'].dt.day
    train['hour'] = train['timestamp'].dt.hour
    train['minute'] = train['timestamp'].dt.minute
    train['dayofweek'] = train['timestamp'].dt.dayofweek
    train=train.drop(['timestamp'], axis=1)
    test=pd.DataFrame()
    for x in acident_index_test:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        test = pd.concat([test, result], axis=0)
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test['day'] = test['timestamp'].dt.day
    test['hour'] = test['timestamp'].dt.hour
    test['minute'] = test['timestamp'].dt.minute
    test['dayofweek'] = test['timestamp'].dt.dayofweek
    test=test.drop(['timestamp'], axis=1)
    X_train=train[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    X_test=test[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    y_train=train[['current']]
    y_test = test[['current']] 
    df = pd.concat([test, train], axis=0)
    transform = list(df.dtypes[df.dtypes != 'object'].index.values) 
    transform.remove('current')
    numerical = Pipeline(steps=[
      
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(2))])

    column_preprocessor = ColumnTransformer(
        transformers=[
          ('num', numerical, transform)])
    start = time.time()
    clf1 = Pipeline(steps=[('preprocessor', column_preprocessor),
                          ('classifier', MLPClassifier(learning_rate='adaptive',alpha= 0.001,learning_rate_init= 0.0001,early_stopping=True,max_iter=300,solver= 'adam',hidden_layer_sizes=(300, 300, 300, 300, 300)))])
    clf2 = Pipeline(steps=[('preprocessor', column_preprocessor),
                          ('classifier',RandomForestClassifier(n_estimators=300))])
    clf3 = Pipeline(steps=[('preprocessor', column_preprocessor),
                          ('classifier',svm.SVC(kernel='rbf',degree=0,gamma=0.005,C=1.2 ))])
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    eclf1 = eclf1.fit(X_train, y_train.values.ravel()) 
    def vc_result(x):
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result['day'] = result['timestamp'].dt.day
        result['hour'] = result['timestamp'].dt.hour
        result['minute'] = result['timestamp'].dt.minute
        result['dayofweek'] = result['timestamp'].dt.dayofweek
        res = eclf1.predict(result[['speed','occ','flow','station_id','day','hour','minute','dayofweek']])
        result["svm_result"] = res
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        return(result)
    def dr_score(list):
        No_of_congestions_detected=0
        No_of_true_congestions=len(list)
        for x in list:
            result=vc_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                No_of_congestions_detected+=1
        return(No_of_congestions_detected/No_of_true_congestions)
    col11.write(f"The DR score is {dr_score(acident_index_test)*100:.1f}%")
    def far(list):
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        for x in list:
            result=vc_result(x)
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== svm_result and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== svm_result and 0 ==current').shape[0]
        return(No_of_false_alarms_signals/No_of_non_congestion_instances)
    col11.write(f"The FAR score is {far(acident_index_test)*100:.2f}%")   
    def MTTD(list):
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=vc_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.svm_result == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        return(sum_of_time/No_of_congestions_detected)   
    col22.write(f"The MTTD score is {MTTD(acident_index_test):.2f} minutes") 
    et = time.time()
    col22.write(f"The total execution time is {( et - start)/60 :.2f} minutes") 
    result=vc_result(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["svm_result"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)

  def model6():
    train=pd.DataFrame()
    for x in acident_index_train:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        train = pd.concat([train, result], axis=0)
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train['day'] = train['timestamp'].dt.day
    train['hour'] = train['timestamp'].dt.hour
    train['minute'] = train['timestamp'].dt.minute
    train['dayofweek'] = train['timestamp'].dt.dayofweek
    train=train.drop(['timestamp'], axis=1)
    test=pd.DataFrame()
    for x in acident_index_test:
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        test = pd.concat([test, result], axis=0)
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test['day'] = test['timestamp'].dt.day
    test['hour'] = test['timestamp'].dt.hour
    test['minute'] = test['timestamp'].dt.minute
    test['dayofweek'] = test['timestamp'].dt.dayofweek
    test=test.drop(['timestamp'], axis=1)
    start = time.time()
    X_train=train[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    X_test=test[['speed','occ','flow','station_id','day','hour','minute','dayofweek']]
    y_train=train[['current']]
    y_test = test[['current']] 
    model = XGBClassifier(use_label_encoder=False, 
                          eval_metric='mlogloss')
    def objective(trial):
        """Define the objective function"""

        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }

        # Fit the model
        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)

        # Make predictions
        y_pred = optuna_model.predict(X_test)

        # Evaluate predictions
        accuracy = recall_score(y_test, y_pred)
        return accuracy
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    params = trial.params
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    def xg_result(x):
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        result['day'] = result['timestamp'].dt.day
        result['hour'] = result['timestamp'].dt.hour
        result['minute'] = result['timestamp'].dt.minute
        result['dayofweek'] = result['timestamp'].dt.dayofweek
        res = model.predict(result[['speed','occ','flow','station_id','day','hour','minute','dayofweek']])
        result["svm_result"] = res
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        return(result)
    def dr_score(list):
        No_of_congestions_detected=0
        No_of_true_congestions=len(list)
        for x in list:
            result=xg_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                No_of_congestions_detected+=1
        return(No_of_congestions_detected/No_of_true_congestions)
    col11.write(f"The DR score is {dr_score(acident_index_test)*100:.1f}%")
    def far(list):
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        for x in list:
            result=xg_result(x)
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== svm_result and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== svm_result and 0 ==current').shape[0]
        return(No_of_false_alarms_signals/No_of_non_congestion_instances)
    col11.write(f"The FAR score is {far(acident_index_test)*100:.2f}%")  
    def MTTD(list):
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=xg_result(x)
            if result.query('1== svm_result and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.svm_result == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        return(sum_of_time/No_of_congestions_detected) 
    col22.write(f"The MTTD score is {MTTD(acident_index_test):.2f} minutes") 
    et = time.time()
    col22.write(f"The total execution time is {( et - start)/60 :.2f} minutes") 
    result=xg_result(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["svm_result"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)


  def model7():
    def RNN_result(x):
        train=pd.DataFrame()
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-10))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        train = pd.concat([train, result], axis=0)
        train["timestamp"] = pd.to_datetime(train["timestamp"])
        scaler = StandardScaler()
        scaler.fit(train[["speed"]])
        train[["speed"]] = scaler.fit_transform(train[["speed"]])

        df=train[(train['timestamp'] >=  train['timestamp'].iloc[0]) & (train['timestamp'] <= train['timestamp'].iloc[0] + timedelta(hours=4))]
        df=df[['speed','timestamp']]
        df['timestamp'] = df['timestamp'].astype(np.int64)
        train['timestamp'] = train['timestamp'].astype(np.int64)
        # Define input sequence length
        sequence_length = 30
        # Create sequences for training
        sequences = []
        for i in range(len(df) - sequence_length):
            sequence = df.iloc[i:i + sequence_length]
            sequences.append(sequence)
        # Convert sequences to NumPy arrays
        X = np.array([sequence['timestamp'].values for sequence in sequences])
        y = np.array([sequence['speed'].values[-1] for sequence in sequences])
        # Reshape input to match LSTM input shape
        X = X.reshape(X.shape[0], X.shape[1], 1)
        # Build LSTM model
        model = tf.keras.models.Sequential([
          tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                              strides=1, padding="causal",
                              activation="relu",
                              input_shape=(sequence_length, 1)),
          tf.keras.layers.SimpleRNN(64, return_sequences=True),
          tf.keras.layers.SimpleRNN(64),
          tf.keras.layers.Dense(30, activation="relu"),
          tf.keras.layers.Dense(10, activation="relu"),
          tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X, y, epochs=10, batch_size=1,verbose=0)
        result = pd.DataFrame(columns=['timestamp', 'speed', 'current_predicted'])
        i=0
        for index, row in train.iterrows():
            if row['timestamp'] in df['timestamp'].values :
                continue
            else: 
                a=pd.Series({'timestamp': row['timestamp'],
                                    'speed': row['speed']})
                df=pd.concat([df, a.to_frame().T], ignore_index=True)
                # Re-create sequences for training with the updated DataFrame
                sequences = []
                for i in range(len(df) - sequence_length):
                    sequence = df.iloc[i:i + sequence_length]
                    sequences.append(sequence)
                new_sequence = sequences[-1]
                new_X = np.array([new_sequence['timestamp'].values])
                new_X = new_X.reshape(new_X.shape[0], new_X.shape[1], 1)
                predicted_speed = model.predict(new_X)
                result = result.append({'timestamp': row['timestamp'], 'speed': row['speed'], 'current_predicted': predicted_speed[0][0]}, ignore_index=True)
                X = np.array([sequence['timestamp'].values for sequence in sequences])
                y = np.array([sequence['speed'].values[-1] for sequence in sequences])
                X = X.reshape(X.shape[0], X.shape[1], 1)
                # Re-train the model with the updated data
                i+=1
                if ((i%4)==0):
                    model.fit(X, y, epochs=2, batch_size=1,verbose=0)
        result[["speed"]] = scaler.inverse_transform(result[["speed"]])
        result[["current_predicted"]] = scaler.inverse_transform(result[["current_predicted"]])
        result['timestamp'] = pd.to_datetime(result['timestamp'], unit='ns')
        result['current_predicted'] = ((result['speed'] - result['current_predicted'])/ result['speed'])
        result['current_predicted'] =result['current_predicted'].abs()
        result['current_predicted'] = result['current_predicted'].apply(lambda x: 0 if x < 0.25 else 1)      
        train['timestamp'] = pd.to_datetime(train['timestamp'], unit='ns')
        result1 = pd.merge(result[['timestamp','current_predicted']], train, on=['timestamp'], how='inner')
        return(result1)
    start = time.time()
    def score(list):
        No_of_congestions_detected1=0
        No_of_true_congestions=len(list)
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=RNN_result(x)
            if result.query('1== current_predicted and 1 ==current').shape[0]>1:
                No_of_congestions_detected1+=1
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== current_predicted and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== current_predicted and 0 ==current').shape[0]
            if result.query('1== current_predicted and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.current_predicted == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        x=(No_of_congestions_detected1/No_of_true_congestions)
        y=(No_of_false_alarms_signals/No_of_non_congestion_instances)
        z=(sum_of_time/No_of_congestions_detected)
        return [x, y , z]
    score=score(acident_index_test)
    et = time.time()
    prediction_time = et - start
    col11.write(f"The DR score is {score[0]*100:.1f}%")
    col11.write(f"The FAR score is {score[1]*100:.2f}%")  
    col22.write(f"The MTTD score is {score[2]:.2f} minutes") 
    col22.write(f"The total execution time is {(prediction_time)/60 :.2f} minutes") 
    result=RNN_result(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["current_predicted"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)




  def model8():
    def lstm_result(x):
        train=pd.DataFrame()
        result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-10))]
        result['current']=0
        result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
        train = pd.concat([train, result], axis=0)
        train["timestamp"] = pd.to_datetime(train["timestamp"])
        scaler = StandardScaler()
        scaler.fit(train[["speed"]])
        train[["speed"]] = scaler.fit_transform(train[["speed"]])

        df=train[(train['timestamp'] >=  train['timestamp'].iloc[0]) & (train['timestamp'] <= train['timestamp'].iloc[0] + timedelta(hours=4))]
        df=df[['speed','timestamp']]
        df['timestamp'] = df['timestamp'].astype(np.int64)
        train['timestamp'] = train['timestamp'].astype(np.int64)
        # Define input sequence length
        sequence_length = 30
        # Create sequences for training
        sequences = []
        for i in range(len(df) - sequence_length):
            sequence = df.iloc[i:i + sequence_length]
            sequences.append(sequence)
        # Convert sequences to NumPy arrays
        X = np.array([sequence['timestamp'].values for sequence in sequences])
        y = np.array([sequence['speed'].values[-1] for sequence in sequences])
        # Reshape input to match LSTM input shape
        X = X.reshape(X.shape[0], X.shape[1], 1)
        # Build LSTM model
        model = tf.keras.models.Sequential([
          tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                              strides=1, padding="causal",
                              activation="relu",
                              input_shape=(sequence_length, 1)),
          tf.keras.layers.LSTM(64, return_sequences=True),
          tf.keras.layers.LSTM(64),
          tf.keras.layers.Dense(30, activation="relu"),
          tf.keras.layers.Dense(10, activation="relu"),
          tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(X, y, epochs=10, batch_size=1,verbose=0)
        result = pd.DataFrame(columns=['timestamp', 'speed', 'current_predicted'])
        i=0
        for index, row in train.iterrows():
            if row['timestamp'] in df['timestamp'].values :
                continue
            else: 
                a=pd.Series({'timestamp': row['timestamp'],
                                    'speed': row['speed']})
                df=pd.concat([df, a.to_frame().T], ignore_index=True)
                # Re-create sequences for training with the updated DataFrame
                sequences = []
                for i in range(len(df) - sequence_length):
                    sequence = df.iloc[i:i + sequence_length]
                    sequences.append(sequence)
                new_sequence = sequences[-1]
                new_X = np.array([new_sequence['timestamp'].values])
                new_X = new_X.reshape(new_X.shape[0], new_X.shape[1], 1)
                predicted_speed = model.predict(new_X)
                result = result.append({'timestamp': row['timestamp'], 'speed': row['speed'], 'current_predicted': predicted_speed[0][0]}, ignore_index=True)
                X = np.array([sequence['timestamp'].values for sequence in sequences])
                y = np.array([sequence['speed'].values[-1] for sequence in sequences])
                X = X.reshape(X.shape[0], X.shape[1], 1)
                # Re-train the model with the updated data
                i+=1
                if ((i%4)==0):
                    model.fit(X, y, epochs=2, batch_size=1,verbose=0)
        result[["speed"]] = scaler.inverse_transform(result[["speed"]])
        result[["current_predicted"]] = scaler.inverse_transform(result[["current_predicted"]])
        result['timestamp'] = pd.to_datetime(result['timestamp'], unit='ns')
        result['current_predicted'] = ((result['speed'] - result['current_predicted'])/ result['speed'])
        result['current_predicted'] =result['current_predicted'].abs()
        result['current_predicted'] = result['current_predicted'].apply(lambda x: 0 if x < 0.25 else 1)      
        train['timestamp'] = pd.to_datetime(train['timestamp'], unit='ns')
        result1 = pd.merge(result[['timestamp','current_predicted']], train, on=['timestamp'], how='inner')
        return(result1)
    start = time.time()
    def score(list):
        No_of_congestions_detected1=0
        No_of_true_congestions=len(list)
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=lstm_result(x)
            if result.query('1== current_predicted and 1 ==current').shape[0]>1:
                No_of_congestions_detected1+=1
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('1== current_predicted and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('1== current_predicted and 0 ==current').shape[0]
            if result.query('1== current_predicted and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.current_predicted == 1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        x=(No_of_congestions_detected1/No_of_true_congestions)
        y=(No_of_false_alarms_signals/No_of_non_congestion_instances)
        z=(sum_of_time/No_of_congestions_detected)
        return [x, y , z]
    score=score(acident_index_test)
    et = time.time()
    prediction_time = et - start
    col11.write(f"The DR score is {score[0]*100:.1f}%")
    col11.write(f"The FAR score is {score[1]*100:.2f}%")  
    col22.write(f"The MTTD score is {score[2]:.2f} minutes") 
    col22.write(f"The total execution time is {(prediction_time)/60 :.2f} minutes") 
    result=lstm_result(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["current_predicted"] == 1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)




  def model9():
    start = time.time()
    def Isolation_Forest(x):
          train=pd.DataFrame()
          result = trafic_data[(trafic_data['station_id'] == incident.at[x,'up_id'])&(trafic_data['timestamp'] <= incident.at[x,'timestamp'] + timedelta(hours=6))&(trafic_data['timestamp'] >= incident.at[x,'timestamp'] + timedelta(hours=-6))]
          result['current']=0
          result.loc[(result['timestamp'] >= incident.at[x,'timestamp'])&(result['timestamp'] <= incident.at[x,'timestamp']+ timedelta(minutes=incident.at[x,'duration'])), 'current'] = 1
          train = pd.concat([train, result], axis=0)
          train["timestamp"] = pd.to_datetime(train["timestamp"])
          scaler = StandardScaler()
          scaler.fit(train[["speed"]])
          train[["speed"]] = scaler.fit_transform(train[["speed"]])
          train[["occ"]] = scaler.fit_transform(train[["occ"]])
          train[["flow"]] = scaler.fit_transform(train[["flow"]])
          model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
          model.fit(train[["speed","occ","flow"]])
          train['scores']=model.decision_function(train[["speed","occ","flow"]])
          train['anomaly']=model.predict(train[["speed","occ","flow"]])

          return(train)
    def dr_score(list):
        No_of_congestions_detected=0
        No_of_true_congestions=len(list)
        for x in list:
            result=Isolation_Forest(x)
            if result.query('-1== anomaly and 1 ==current').shape[0]>1:
                No_of_congestions_detected+=1
        return(No_of_congestions_detected/No_of_true_congestions)
    col11.write(f"The DR score is {dr_score(acident_index_test)*100:.1f}%") 

    def far(list):
        No_of_false_alarms_signals=0
        No_of_non_congestion_instances=0
        for x in list:
            result=Isolation_Forest(x)
            No_of_non_congestion_instances+=result.query('0 ==current').shape[0]
            if result.query('-1== anomaly  and 0 ==current').shape[0]>1:
                No_of_false_alarms_signals+=result.query('-1== anomaly and 0 ==current').shape[0]
        return(No_of_false_alarms_signals/No_of_non_congestion_instances)
    col11.write(f"The FAR score is {far(acident_index_test)*100:.2f}%") 
    def MTTD(list):
        No_of_congestions_detected=0
        sum_of_time=0
        for x in list:
            result=Isolation_Forest(x)
            if result.query('-1== anomaly  and 1 ==current').shape[0]>1:
                sum_of_time+=(result[(result.anomaly == -1) &(result.current == 1)].iloc[0]["timestamp"]-incident.at[x,'timestamp']).total_seconds() / 60
                No_of_congestions_detected+=1
        return(sum_of_time/No_of_congestions_detected)  
    col22.write(f"The MTTD score is {MTTD(acident_index_test):.2f} minutes") 
    et = time.time()
    col22.write(f"The total execution time is {( et - start)/60 :.2f} minutes") 
    result=Isolation_Forest(acident_index_test[0])
    fig = go.Figure()
    fig.add_scattergl(x=result["timestamp"], y=result["speed"], line={'color': 'blue'})
    fig.add_scattergl(x = result["timestamp"], y = result["speed"].where(result["anomaly"] == -1), line ={'color' : 'red'})
    fig.update_traces(showlegend=False)
    fig.update_layout(
    title={
        'text': "Example",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    col2.plotly_chart(fig, use_container_width=True)
  
  st.text("")
  st.text("")
  col1, col2 = st.columns([1,2])
  col11, col22 = col2.columns([1,2])

  with col1:
     
     line1 = option_menu(
      menu_title=None,
      options=["modified Z score","support vector machine", "random forest","Feedforward neural network","voting classifier","XGBoost","recurrent neural network","long short-term memory networks", "Isolation Forest"],
      orientation="vertical",
  )

  if line1 == "modified Z score":
    model1()
  if line1 == "support vector machine":
    model2()
  if line1 == "random forest":
    model3()
  if line1 == "Feedforward neural network":
    model4()
  if line1 == "voting classifier":
    model5()
  if line1 == "XGBoost":
    model6()
  if line1 == "recurrent neural network":
    model7()
  if line1 == "long short-term memory networks":
    model8()
  if line1 == "Isolation Forest":
    model9()


def page3():
  st.markdown(hide_table_row_index, unsafe_allow_html=True)
  st.markdown('There are three standard and commonly used performance metrics that are used : the DR, the FAR and the MTTD.')
  st.markdown('• The Detection Rate (DR) is the percentage of the correctly detected congestions.')
  st.markdown('• The False Alarm Rate (FAR) is the percentage of false alarm signals to the total number of non-incident instances.')
  st.markdown('• The Mean Time to Detection (MTTD) is the average of the congestion detection delay, where this delay is the time difference between the moment when the congestion was detected by the algorithm and the moment when it appeared in reality.')
  st.markdown('• compilation time refers to the total time needed for training ,optimizing and predicting.')

  data = {' ': ['modified Z score', 'support vector machine', 'random forest', 'Feedforward neural network', 'voting classifier', 'XGBoost', 'Numenta Htm', 'recurrent neural network', 'long short-term memory networks', 'Isolation Forest'],
        'DR ': ["95%","90%", "95%", "92%", "95%", "95%", "80%", "95%", "95%", "100%"],
        'FAR': ["4.3%", "3.2%", "4.8%","5.0%", "4.4%", "4.2%", "5.6%", "4.8%", "4.7%", "4.3%"],
        'MTTD': ["2.2 min", "3.8 min", "2.1 min", "3.1 min", "3.5 min", "2.6 min", "3.4 min", "2.8 min", "2.9 min", "0.8 min"],
        'compilation time': ["0.57 min", "1.23 min", "24.49 min", "14.20 min", "1.56 min", "4.94 min", "25.9 min", "30.9 min", "58.6 min", "0.8 min"]}
  df = pd.DataFrame(data)
  st.table(df)
 



def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)





link="https://www.bme.hu/?language=en"
image_base64 = get_base64_of_bin_file('logo.png')
image_base64_2 = get_base64_of_bin_file('deb_logo.png')
link_2="https://www.vik.bme.hu/en/"
a = f'<div style="background-color:#ee605f;left: 0;top: 0;width: 100%;margin-left: 0px; margin-right: 0px;"><div class="column"style="float: left;width: 15.0%;"><a href="{link}"><img src="data:image/png;base64,{image_base64}"></a></div><div class="column"style="float: left;width: 70.0%;"><h2  style="margin: 0px 0px 0px 0px;padding: 0px 0px 50px 0px ;text-align: center;font-family:Calibri (Body);"> Traffic congestion detection algorithms <br/> by Fares Ghezal, supervised by Mohammad Bawaneh </h2></div><div class="column"style="float: left;width: 15.0%;"><a href="{link_2}"><img src="data:image/png ;base64,{image_base64_2}" width="100" height="60" style="margin: 0px 0px 0px 0px"></a></div></div>' 
st.markdown(a, unsafe_allow_html=True)

st.markdown(f'<div class="line" style=" display: inline-block;border-top: 1px solid black;width:  100%;margin-top: 0px; margin-bottom: 20px"></div>', unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=["data analysis","model implementation", "results comparison"],
    icons=["bar-chart-fill", "diagram-3", "table"], 
    orientation="horizontal",
    styles={
        "container": {"margin": "0","max-width": "100%"},
    }
)




if selected == "data analysis":
  page1()
if selected == "model implementation":
  page2()
if selected == "results comparison":
  page3()



file_ = open("./fb.png", "rb")
contents = file_.read()
fb_url = base64.b64encode(contents).decode("utf-8")
file_.close()
file_ = open("./insta.png", "rb")
contents = file_.read()
ins_url = base64.b64encode(contents).decode("utf-8")
file_.close()
file_ = open("./web.png", "rb")
contents = file_.read()
web_url = base64.b64encode(contents).decode("utf-8")
file_.close()

footer=('''<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #ee605f;
color: black;
text-align: center;
overflow: hidden;
}
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-gb8h{border-color:#ee605f;text-align:left}
.tg .tg-hdil{border-color:#ee605f;text-align:center}
.tg .tg-ik7g{border-color:#ee605f;text-align:right}

.footerimg /* lábléc ikonjai */ {
transition: 0.5s; }

.footerimg:hover {
transform: scale(0.8);
transition: 0.5s;
cursor: pointer; }

</style>

<div class="footer" style="margin-bottom:-15px">'''
+
f'''
<table class="tg" style="undefined;table-layout: fixed; width: 100%">
<colgroup>
<col style="width: 20%">
<col style="width: 48%">
<col style="width: 32%">
</colgroup>
<thead>
  <tr>
    <td class="tg-gb8h"></td>
    <td class="tg-hdil"><p style="color:white; font-family:sans-serif; font-size:16px; float:left;margin-top:3px; text-align:center">&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp Faculty of Electrical Engineering and Informatics, BME &nbsp&nbsp&nbsp<a target="_blank" href="https://www.vik.bme.hu/en/" style="margin:1px;padding:1px"><img class="footerimg" src="data:image/gif;base64,{web_url}" style="width:2.5%" ></img></a><a target="_blank" href="https://www.facebook.com/BMEVIK" style="margin:1px;padding:1px"><img class="footerimg" src="data:image/gif;base64,{fb_url}" style="width:2.5%" ></img></a><a target="_blank" href="https://www.instagram.com/bmevik/" style="margin:1px;padding:1px"><img class="footerimg" src="data:image/gif;base64,{ins_url}" style="width:2.5%" ></img></a></p>

  </tr>
</thead>
</table>
''')

st.markdown('This website is aesthetically inspired by a [similar site of the BME MI ](https://research.math.bme.hu), Personally developed under the [HSDSlab](https://hsdslab.math.bme.hu/en.html). We acknowledge the similarity and are thankfull for the inspiration.')
st.markdown(footer,unsafe_allow_html=True)

