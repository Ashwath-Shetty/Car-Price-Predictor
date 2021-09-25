import sklearn
from sklearn.ensemble import RandomForestClassifier
import streamlit
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import joblib
from sklearn.compose import make_column_transformer
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
#import matplotlib.pyplot as plt


import warnings
warnings.simplefilter("ignore")

def preprocess(train,mode=None):
    train_c=train.copy(deep=True)
    if mode=='train':
        train_c.drop_duplicates(inplace=True)
        train_c.reset_index(inplace=True)
        train_c.drop('index',axis=1,inplace=True)
    train_c['Car_age'] = 2021 - train_c['Prod. year']
    train_c['category_processed'] = train_c['Category'].apply(lambda x: 'rare' if x == 'Goods wagon' or x == 'Pickup'
                                              or x == 'Cabriolet' or x == 'Limousine' else x)
    train_c['fuel_type_processed'] = train_c['Fuel type'].apply(lambda x: 'Hybrid-hydrogen' if x == 'Plug-in Hybrid' or x == 'Hydrogen' else x)
    train_c['Turbo_engine'] = train_c['Engine volume'].apply(lambda x: 'Yes' if x.split(" ")[-1] == 'Turbo' else 'No')
    train_c['Engine volume'] = train_c['Engine volume'].apply(lambda x: x.split(" ")[0]).astype(float)
    
    train_c['doors_processed']=train_c.Doors.replace(to_replace={'04-May':4,'02-Mar':2,'>5':6})
    train_c['doors_processed']=train_c['doors_processed'].astype('int')
    train_c['is_levy_missing']=False
    train_c.is_levy_missing[(train_c['Levy']=='-' )]=True
    train_c['levy_processed']=train_c['Levy'].replace("-",np.nan)
    train_c['levy_processed']=train_c['levy_processed'].fillna(0)
    train_c['levy_processed']=train_c['levy_processed'].astype('float')
    train_c['mileage_processed']=train_c['Mileage'].str.split(n=1,expand=True)[0:][0]
    train_c['mileage_processed']=train_c['mileage_processed'].astype('float')
    
    return train_c

def pipeline(train_data,test_data):
    train_data=preprocess(train_data,'train')
    test_data=preprocess(test_data)
    continuous=['levy_processed','mileage_processed','Car_age']
    categories=['Model','Prod. year','Engine volume','Color','Airbags',
           'Manufacturer','category_processed','fuel_type_processed','Turbo_engine','doors_processed']
    column_trans = make_column_transformer((preprocessing.MinMaxScaler(), continuous),(preprocessing.OrdinalEncoder(),categories),remainder='passthrough')
    X_train=train_data[continuous+categories]
    X_test=test_data[continuous+categories]
    combined_df=pd.concat([X_train,X_test])
    transf=column_trans.fit(combined_df)
    X_train_transformed=transf.transform(X_train)
    X_test_transformed=transf.transform(X_test)
    y_train=train_data['Price']
    return X_train_transformed,X_test_transformed,y_train

def UI():
    st.title('Car Price Predictor')
    data=pd.read_csv("Data/train.csv")
    all_features=pd.DataFrame()
    manufacturer_all=data['Manufacturer'].unique()
    manufacturer_selected = st.selectbox('Select the Manufacturer',manufacturer_all)
    all_features['Manufacturer']=[manufacturer_selected]
    model_all=data.loc[data["Manufacturer"]==manufacturer_selected,'Model'].unique()
    model_selected = st.selectbox('Selct the Model',model_all)
    all_features['Model']=[model_selected]
    production_year_all=data['Prod. year'].unique()
    production_year_selected = st.selectbox('Select the Production Year',production_year_all)
    all_features['Prod. year']=production_year_selected
    category_all=data.loc[(data["Manufacturer"]==manufacturer_selected) & (data['Model']==model_selected),'Category'].unique()
    category_selected = st.selectbox('Select the Category',category_all)
    all_features['Category']=[category_selected]
    fuel_all=data['Fuel type'].unique()
    fuel_selected = st.selectbox('select the Fuel Type',fuel_all)
    all_features['Fuel type']=fuel_selected
    Engine_volume_all=data.loc[(data["Manufacturer"]==manufacturer_selected) & (data['Model']==model_selected),'Engine volume'].unique()
    Engine_volume_selected = st.selectbox('select the Engine Volume',Engine_volume_all)
    all_features['Engine volume']=Engine_volume_selected
    data['doors_processed']=data.Doors.replace(to_replace={'04-May':4,'02-Mar':2,'>5':'other'})
    doors_all=data['doors_processed'].unique()
    data['doors_processed']=data.doors_processed.replace(to_replace={'other':6})
    data['doors_processed']=data['doors_processed'].astype('int')
    doors_selected = st.selectbox('select the Number of Doors',doors_all)
    if doors_selected=='other':
        doors_selected=6
    all_features['Doors']=doors_selected
    color_all=data.loc[(data["Manufacturer"]==manufacturer_selected) & (data['Model']==model_selected),'Color'].unique()
    color_selected = st.selectbox('select the colors',color_all)
    all_features['Color']=color_selected
    airbags_all=data.loc[(data["Manufacturer"]==manufacturer_selected) & (data['Model']==model_selected),'Airbags'].unique()
    airbags_selected = st.selectbox('Select the Airbags',airbags_all)
    all_features['Airbags']=airbags_selected
    levy_all=['No','yes']
    levy_selected = st.selectbox('Do you know the Levy?',levy_all)
    all_features['is_levy_missing']=levy_selected
    if levy_selected=='No':
        all_features['Levy']='-'
    if levy_selected=='yes':
        levy = st.number_input("Enter the levy.",min_value=None,key='1')
        all_features['Levy']=str(levy)
    mileage= st.number_input("Enter the Mileage(odometer reading)", 0,key='2')
    all_features['Mileage']=str(mileage)
    x_tr,x_t,y_t=pipeline(data,all_features)
    continuous=['levy_processed','mileage_processed','Car_age']
    categories=['Model','Prod. year','Engine volume','Color','Airbags',
           'Manufacturer','category_processed','fuel_type_processed','Turbo_engine','doors_processed']
    collected_data=pd.DataFrame(x_t,columns=continuous+categories)
    return collected_data

def prediction(collected_data):
    filename = 'Model.pkl'
    loaded_model = joblib.load(filename, mmap_mode=None)
    if (st.button('predict')):
        pred=loaded_model.predict(collected_data)
        return st.success("price in USD {}".format(pred[0]))
        

def main():
    collected_data=UI()
    prediction(collected_data)


if __name__=='__main__':
    main()
