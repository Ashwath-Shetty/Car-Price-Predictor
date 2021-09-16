def preprocess(train,mode=None):
    train_c=train.copy(deep=True)
    #test_c=test.copy(deep=True)
    if mode=='train':
        train_c.drop_duplicates(inplace=True)
        train_c.reset_index(inplace=True)
        train_c.drop('index',axis=1,inplace=True)
    train_c['Manufacturer_processed']=train_c['Manufacturer']
    man_cat={'HYUNDAI':1,'TOYOTA':2,'MERCEDES-BENZ':3,'FORD':4,'CHEVROLET':5,'BMW':6,'HONDA':7,'LEXUS':8,'NISSAN':9,'VOLKSWAGEN':10,'SSANGYONG':11,'KIA':12,'OPEL':13,'MITSUBISHI':14,'SUBARU':15,'AUDI':16,'MAZDA':17,'JEEP':18}
    train_c['Manufacturer_processed'].replace(man_cat,inplace=True)
    train_c.Manufacturer_processed[~train_c['Manufacturer_processed'].isin(man_cat.values())]=19
    train_c['Manufacturer_processed']=pd.to_numeric(train_c['Manufacturer_processed'])
#     cat=train_c.Model.value_counts()>=100
#     cat=cat[cat].index
#     model_cat={}
#     for i in range(len(cat)):
#         model_cat[cat[i]]=i
#     train_c['Model_processed']=train_c.Model
#     train_c['Model_processed'].replace(model_cat,inplace=True)
#     train_c.Model_processed[~train_c['Model_processed'].isin(model_cat.values())]=38
#     train_c['Model_processed']=pd.to_numeric(train_c['Model_processed'])
    train_c['Car_age'] = 2021 - train_c['Prod. year']
    train_c['category_processed'] = train_c['Category'].apply(lambda x: 'rare' if x == 'Goods wagon' or x == 'Pickup'
                                              or x == 'Cabriolet' or x == 'Limousine' else x)
    train_c['fuel_type_processed'] = train_c['Fuel type'].apply(lambda x: 'Hybrid-hydrogen' if x == 'Plug-in Hybrid' or x == 'Hydrogen' else x)
    train_c['Turbo_engine'] = train_c['Engine volume'].apply(lambda x: 'Yes' if x.split(" ")[-1] == 'Turbo' else 'No')
    train_c['Engine volume'] = train_c['Engine volume'].apply(lambda x: x.split(" ")[0]).astype(float)
    
#     train_c['doors_processed']=train_c.Doors.replace(to_replace={'04-May':4,'02-Mar':2,'>5':6})
#     train_c['doors_processed']=train_c['doors_processed'].astype('int')
    train_c['is_levy_missing']=False
#     train_c.is_levy_missing[train_c['Levy']=='-']=True
#     train_c['levy_processed']=train_c['Levy'].replace("-",np.nan)
# #print("number of missing values in levy is",train_c.levy_processed.isna().sum())
#     train_c['Levy_cor']=train_c['levy_processed']
#     train_c['Levy_cor']= train_c.groupby(['Engine volume', 'Cylinders'])['Levy_cor'].transform(
#     lambda grp: grp.fillna(grp.median()))
#     train_c['Levy_cor']= train_c.groupby(['Engine volume'])['Levy_cor'].transform(
#     lambda grp: grp.fillna(grp.median()))
    
#     train_c['Levy_cor']= train_c.groupby(['Cylinders'])['Levy_cor'].transform(
#     lambda grp: grp.fillna(grp.median()))
#     train_c['Levy_cor']=train_c['Levy_cor'].astype('int')
#     train_c['mileage_processed']=train_c['Mileage'].str.split(n=1,expand=True)[0:][0]
#     train_c['mileage_processed']=train_c['mileage_processed'].astype('int')
    train_c['mileage_processed']=train_c['mileage_processed'].replace(0,np.nan)
    train_c['mileage_processed']= train_c.groupby('Prod. year')['mileage_processed'].transform(
    lambda grp: grp.fillna(grp.median()))
# Fill remaining na's(if any) with median of entire data
    train_c['mileage_processed']= train_c['mileage_processed'].fillna(train_c['mileage_processed'].median())
    train_c['mileage_processed']= train_c['mileage_processed'].astype('int')
    
    return train_c
    
    

