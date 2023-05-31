#Notebook Imports
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st


import warnings
warnings.filterwarnings('ignore')

#Create a pandas dataframe
data = pd.read_csv('Housing.csv')

#converting/removing categorical data
data['basement']=pd.get_dummies(data['basement'], drop_first=True)
data['mainroad']=pd.get_dummies(data['mainroad'], drop_first=True)
data['prefarea']=pd.get_dummies(data['prefarea'], drop_first=True)
data['hotwaterheating']=pd.get_dummies(data['hotwaterheating'], drop_first=True)
data['airconditioning']=pd.get_dummies(data['airconditioning'], drop_first=True)
data['guestroom']=pd.get_dummies(data['guestroom'], drop_first=True)
data=data.drop('furnishingstatus', axis='columns')
features=data.drop(['bedrooms', 'price'], axis=1)

log_prices=np.log(data['price'])
target=pd.DataFrame(log_prices, columns=['price'])

area_index=0
bathrooms_index=1
stories_index=2
mainroad_index=3
guestroom_index=4
basement_index=5
hotwaterheating_index=6
airconditioning_index=7
parking_index=8
prefarea_index=9

#property_stats=np.ndarray(shape=(1,11))
#property_stats[0][price_index]=0.02
property_stats=features.mean().values.reshape(1,10)

regression=LinearRegression().fit(features, target)
fitted_values=regression.predict(features)

MSE=mean_squared_error(target,fitted_values)
RMSE=np.sqrt(MSE)

def get_log_estimate(area, bathrooms, stories, parking, mainroad=True, guestroom=False, 
                     basement=True, hotwater=True, aircondition=True, prefarea=False, high_confidence=True):
    #configure property
    property_stats[0][area_index]=area
    property_stats[0][bathrooms_index]=bathrooms
    property_stats[0][stories_index]=stories
    property_stats[0][parking_index]=parking
    
    if mainroad:
        property_stats[0][mainroad_index]=1
    else:
        property_stats[0][mainroad_index]=0
        
    if guestroom:
        property_stats[0][guestroom_index]=1
    else:
        property_stats[0][guestroom_index]=0
        
    if basement:
        property_stats[0][basement_index]=1
    else:
        property_stats[0][basement_index]=0
        
    if hotwater:
        property_stats[0][hotwaterheating_index]=1
    else:
        property_stats[0][hotwaterheating_index]=0
        
    if aircondition:
        property_stats[0][airconditioning_index]=1
    else:
        property_stats[0][airconditioning_index]=0
        
    if prefarea:
        property_stats[0][prefarea_index]=1
    else:
        property_stats[0][prefarea_index]=0
    
    #make prediction
    log_estimate=regression.predict(property_stats)[0][0]
    
    #range
    if high_confidence:
        upper_bound=log_estimate+2*RMSE
        lower_bound=log_estimate-2*RMSE
        interval=95
    else:
        upper_bound=log_estimate+RMSE
        lower_bound=log_estimate-RMSE
        interval=68
    return log_estimate, upper_bound, lower_bound, interval
	
	

def get_dollar_estimate(area, bathrooms, stories, parking, mainroad=True, guestroom=False, basement=True, 
                        hotwater=True, aircondition=True, prefarea=False, high_confidence=True):
    
    
    log_estimate, upper, lower, confidence = get_log_estimate(area, bathrooms, stories, parking, mainroad, guestroom, basement, hotwater, aircondition, prefarea, high_confidence)

    #convert from log
    dollar_estimate=np.e**log_estimate
    dollar_high=np.e**upper
    dollar_low=np.e**lower

    #round to nearest thousand
    rounded_estimate=np.around(dollar_estimate, -3)
    rounded_high=np.around(dollar_high, -3)
    rounded_low=np.around(dollar_low, -3)

    return print(f'The estimated property value is {rounded_estimate}.\n At {confidence}% confidence the valuation range is\n USD {rounded_low} at the low end to USD {rounded_high} at the high end.')


#print(get_dollar_estimate(3000, 3, 3, 2))

st.title('House valuation tool')

st.sidebar.header('User Input Parameters')

def user_input_features():
    area=st.sidebar.slider('area', 1650, 16200, 7275)
    bathrooms=st.sidebar.slider('bathrooms', 1, 4, 2)
    stories=st.sidebar.slider('stories',1, 4, 2)
    parking=st.sidebar.slider('parking', 0, 3, 1)
    mainroad=st.sidebar.select_slider('mainroad', options=['True', 'False'])
    guestroom=st.sidebar.select_slider('guestroom', options=['True', 'False'])
    basement=st.sidebar.select_slider('basement', options=['True', 'False'])
    hotwater=st.sidebar.select_slider('hotwater', options=['True', 'False'])
    aircondition=st.sidebar.select_slider('aircondition', options=['True', 'False'])
    prefarea=st.sidebar.select_slider('prefarea', options=['True', 'False'])
    high_confidence=st.sidebar.select_slider('high_confidence', options=['True', 'False'])
    input_data = {'area':area,
                  'bathrooms':bathrooms,
                  'stories':stories,
                  'parking':parking,
                  'mainroad':mainroad,
                  'guestroom':guestroom,
                  'basement':basement,
                  'hotwater':hotwater,
                  'aircondition':aircondition,
                  'prefarea':prefarea,
                  'high_confidence':high_confidence}
    input_df= pd.DataFrame(data, index=[0])
    return area, bathrooms, stories, parking, mainroad, guestroom, basement, hotwater, aircondition, prefarea, high_confidence

df=user_input_features()
area=df[0]
bathrooms=df[1]
stories=df[2]
parking=df[3]
mainroad=df[4]
guestroom=df[5]
basement=df[6]
hotwater=df[7]
aircondition=df[8]
prefarea=df[9]
high_confidence=df[10]

prediction=get_dollar_estimate(area, bathrooms, stories, parking, mainroad, guestroom, basement, hotwater, aircondition, prefarea, high_confidence)

st.subheader('User Input Parameters')
st.write()

st.subheader('Prediction')
st.write(prediction)