import sklearn
from sklearn.ensemble import RandomForestClassifier
import streamlit
import pandas as pd
import numpy as np
import streamlit as st
import pickle


st.title('My first app')
st.write("Here's our first attempt at using data to create a table:")
data=pd.read_csv("train.csv")
print("version",pickle.format_version)
print('sk',sklearn.__version__)
all_features=pd.DataFrame()
#manufacturer_all={'HYUNDAI':1,'TOYOTA':2,'MERCEDES-BENZ':3,'FORD':4,'CHEVROLET':5,'BMW':6,'HONDA':7,'LEXUS':8,'NISSAN':9,'VOLKSWAGEN':10,'SSANGYONG':11,'KIA':12,'OPEL':13,'MITSUBISHI':14,'SUBARU':15,'AUDI':16,'MAZDA':17,'JEEP':18}
manufacturer_all=data['Manufacturer'].unique()
manufacturer_selected = st.selectbox(
    'select the manufacturer',manufacturer_all)
'You selected: ', manufacturer_selected
all_features['Manufacturer']=[manufacturer_selected]



model_all=data.loc[data["Manufacturer"]==manufacturer_selected,'Model'].unique()
model_selected = st.selectbox('selct the model',model_all)
'You selected: ', model_selected
all_features['Model']=[model_selected]


production_year_all=data['Prod. year'].unique()
production_year_selected = st.selectbox('select the production year',production_year_all)
'You selected: ', production_year_selected
all_features['Prod. year']=production_year_selected


category_all=data['Category'].unique()
category_selected = st.selectbox('select the category',category_all)
'You selected: ', category_selected
all_features['Category']=[category_selected]

leather_all=data['Leather interior'].unique()
leather_selected = st.selectbox('select the category',leather_all)
'You selected: ', leather_selected
all_features['Leather interior']=leather_selected

fuel_all=data['Fuel type'].unique()
fuel_selected = st.selectbox('select the fuel type',fuel_all)
'You selected: ', fuel_selected
all_features['Fuel type']=fuel_selected

Engine_volume_all=data['Engine volume'].unique()
Engine_volume_selected = st.selectbox('select the engine volume',Engine_volume_all)
'You selected: ', Engine_volume_selected
all_features['Engine volume']=Engine_volume_selected

cylinders_all=data['Cylinders'].unique()
cylinders_selected = st.selectbox('select the cylinders',cylinders_all)
'You selected: ', cylinders_selected
all_features['Cylinders']=cylinders_selected

gear_box_all=data['Gear box type'].unique()
gear_box_selected = st.selectbox('select the gear box type',gear_box_all)
'You selected: ', gear_box_selected
all_features['Gear box type']=gear_box_selected

drive_wheels_all=data['Drive wheels'].unique()
drive_wheels_selected = st.selectbox('select the drive wheels',drive_wheels_all)
'You selected: ', drive_wheels_selected
all_features['Drive wheels']=drive_wheels_selected


data['doors_processed']=data.Doors.replace(to_replace={'04-May':4,'02-Mar':2,'>5':'other'})
doors_all=data['doors_processed'].unique()
data['doors_processed']=data.doors_processed.replace(to_replace={'other':6})
data['doors_processed']=data['doors_processed'].astype('int')
doors_selected = st.selectbox('select the doors',doors_all)
if doors_selected=='other':
    doors_selected=6
'You selected: ', doors_selected
all_features['Doors']=doors_selected

wheel_all=data['Wheel'].unique()
wheel_selected = st.selectbox('select the wheels',wheel_all)
'You selected: ', wheel_selected
all_features['Wheel']=wheel_selected

color_all=data['Color'].unique()
color_selected = st.selectbox('select the colors',color_all)
'You selected: ', color_selected
all_features['Color']=color_selected

airbags_all=data['Airbags'].unique()
airbags_selected = st.selectbox('select the airbags',airbags_all)
'You selected: ', airbags_selected
all_features['Airbags']=airbags_selected

levy_all=['No','yes']
levy_selected = st.selectbox('do you know the levy',levy_all)
'You selected: ', levy_selected
all_features['is_levy_missing']=levy_selected
if levy_selected=='No':
    all_features['Levy']='-'


if levy_selected=='yes':
    levy = st.number_input("i/p the levy. if levy not known keep it empty",min_value=None,key='1')
    'You selected: ', levy
    all_features['Levy']=str(levy)

mileage= st.number_input("i/p the mileage", 10,key='2')
'You selected: ', mileage
all_features['Mileage']=str(mileage)
# print(len(all_features.columns))
# print("out",all_features)
'You selected: ', all_features['Mileage']

from sklearn.compose import make_column_transformer
from sklearn import preprocessing
# st.button("predict")
# st.success("yo")
from sklearn.ensemble import ExtraTreesRegressor

print("-------")
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
    categories=['Model','Prod. year','Leather interior','Engine volume','Cylinders','Gear box type','Drive wheels','Wheel','Color','Airbags','is_levy_missing',
           'Manufacturer','category_processed','fuel_type_processed','Turbo_engine','doors_processed']
    column_trans = make_column_transformer((preprocessing.MinMaxScaler(), continuous),(preprocessing.OrdinalEncoder(),categories),remainder='passthrough')
    X_train=train_data[continuous+categories]
    X_test=test_data[continuous+categories]
    combined_df=pd.concat([X_train,X_test])
    transf=column_trans.fit(combined_df)
    X_train_transformed=transf.transform(X_train)
    X_test_transformed=transf.transform(X_test)
    y_train=train_data['Price']
    #y_test=test_data['Price']
    return X_train_transformed,X_test_transformed,y_train
    
x_tr,x_t,y_t=pipeline(data,all_features)
continuous=['levy_processed','mileage_processed','Car_age']
categories=['Model','Prod. year','Leather interior','Engine volume','Cylinders','Gear box type','Drive wheels','Wheel','Color','Airbags','is_levy_missing',
           'Manufacturer','category_processed','fuel_type_processed','Turbo_engine','doors_processed']
   
k=pd.DataFrame(x_t,columns=continuous+categories)
st.write(k)
filename = 'finalized_rfmodel.sav'
loaded_model = pickle.load(open(filename, 'rb'))
if (st.button('predict')):
    pred=loaded_model.predict(x_t)
    st.success("price in USD {}".format(pred[0]))
    print(pred)
print("pickle version",pickle.format_version)
