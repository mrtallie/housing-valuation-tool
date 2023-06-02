#Notebook Imports
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


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
features=data.drop(['price'], axis=1)

log_prices=np.log(data['price'])
target=pd.DataFrame(log_prices, columns=['price'])

area_index=0
bedrooms_index=1
bathrooms_index=2
stories_index=3
mainroad_index=4
guestroom_index=5
basement_index=6
hotwaterheating_index=7
airconditioning_index=8
parking_index=9
prefarea_index=10

#property_stats=np.ndarray(shape=(1,11))
#property_stats[0][price_index]=0.02
property_stats=features.mean().values.reshape(1,11)

regression=LinearRegression().fit(features, target)
fitted_values=regression.predict(features)

MSE=mean_squared_error(target,fitted_values)
RMSE=np.sqrt(MSE)

def get_log_estimate(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, 
                     basement, hotwater, aircondition, prefarea, high_confidence):
    #configure property
    property_stats[0][area_index]=area
    property_stats[0][bathrooms_index]=bathrooms
    property_stats[0][stories_index]=stories
    property_stats[0][parking_index]=parking

    
    if mainroad.any()==1:
        property_stats[0][mainroad_index]=1
    else:
        property_stats[0][mainroad_index]=0
        
    if guestroom.any():
        property_stats[0][guestroom_index]=1
    else:
        property_stats[0][guestroom_index]=0
        
    if basement.any():
        property_stats[0][basement_index]=1
    else:
        property_stats[0][basement_index]=0
        
    if hotwater.any():
        property_stats[0][hotwaterheating_index]=1
    else:
        property_stats[0][hotwaterheating_index]=0
        
    if aircondition.any():
        property_stats[0][airconditioning_index]=1
    else:
        property_stats[0][airconditioning_index]=0
        
    if prefarea.any():
        property_stats[0][prefarea_index]=1
    else:
        property_stats[0][prefarea_index]=0
    
    #make prediction
    log_estimate=regression.predict(property_stats)[0][0]
    
    #range
    if high_confidence.any():
        upper_bound=log_estimate+2*RMSE
        lower_bound=log_estimate-2*RMSE
        interval=95
    else:
        upper_bound=log_estimate+RMSE
        lower_bound=log_estimate-RMSE
        interval=68

    print(interval)
    return log_estimate, upper_bound, lower_bound, interval
	
	

def get_dollar_estimate(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, 
                        hotwater, aircondition, prefarea, high_confidence):
    
    
    log_estimate, upper, lower, confidence = get_log_estimate(area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwater, aircondition, prefarea, high_confidence)

    #convert from log
    dollar_estimate=np.e**log_estimate / 10
    dollar_high=np.e**upper / 10
    dollar_low=np.e**lower / 10

    #round to nearest thousand
    rounded_estimate=np.around(dollar_estimate, -3)
    rounded_high=np.around(dollar_high, -3)
    rounded_low=np.around(dollar_low, -3)


    prediction=f'The estimated property value is {rounded_estimate}.\n At {confidence}% confidence the valuation range is\n USD {rounded_low} at the low end to USD {rounded_high} at the high end.'

    return prediction


#print(get_dollar_estimate(3000, 3, 3, 2))

st.title('House valuation tool')

st.sidebar.header('User Input Parameters')

def user_input_features():
    area=st.sidebar.slider('area', 1650, 16200, 7275)
    bedrooms=st.sidebar.slider('bedrooms', 1, 6, 3)
    bathrooms=st.sidebar.slider('bathrooms', 1, 4, 2)
    stories=st.sidebar.slider('stories',1, 4, 2)
    parking=st.sidebar.slider('parking', 0, 3, 1)
    mainroad=st.sidebar.select_slider('mainroad', options=['True', 'False'])
    if mainroad=='True':
        mainroad_bool=1
    else:
        mainroad_bool=0
    guestroom=st.sidebar.select_slider('guestroom', options=['True', 'False'])
    if guestroom=='True':
        guestroom_bool=1
    else:
        guestroom_bool=0
    basement=st.sidebar.select_slider('basement', options=['True', 'False'])
    if basement=='True':
        basement_bool=1
    else:
        basement_bool=0
    hotwater=st.sidebar.select_slider('hotwater', options=['True', 'False'])
    if hotwater=='True':
        hotwater_bool=1
    else:
        hotwater_bool=0
    aircondition=st.sidebar.select_slider('aircondition', options=['True', 'False'])
    if aircondition=='True':
        aircondition_bool=1
    else:
        aircondition_bool=0
    prefarea=st.sidebar.select_slider('prefarea', options=['True', 'False'])
    if prefarea=='True':
        prefarea_bool=1
    else:
        prefarea_bool=0
    high_confidence=st.sidebar.select_slider('high_confidence', options=['True', 'False'])
    if high_confidence=='True':
        high_confidence_bool=1
    else:
        high_confidence_bool=0
    input_data = {'area':area,
                  'bedrooms':bedrooms,
                  'bathrooms':bathrooms,
                  'stories':stories,
                  'parking':parking,
                  'mainroad':mainroad_bool,
                  'guestroom':guestroom_bool,
                  'basement':basement_bool,
                  'hotwater':hotwater_bool,
                  'aircondition':aircondition_bool,
                  'prefarea':prefarea_bool,
                  'high_confidence':high_confidence_bool}
    input_df=pd.DataFrame(input_data, index=[0])
    return input_df


df=user_input_features()

print(df)

prediction=get_dollar_estimate(df.area, df.bedrooms, df.bathrooms, df.stories, df.parking, df.mainroad, df.guestroom, df.basement, df.hotwater, df.aircondition, df.prefarea, df.high_confidence)

print(prediction)

#st.subheader('User Input Parameters')
#st.write()


st.subheader('Prediction')
st.write(prediction)

#visual aids
#create a heat map
st.subheader('Visulization of prediction model')
mask = np.zeros_like(data.corr(numeric_only=True))
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(16,10))
plt.title('Heatmap of Attributes')
sns.heatmap(data.corr(numeric_only=True), annot=True, mask=mask)
st.pyplot(bbox_inches='tight')
st.set_option('deprecation.showPyplotGlobalUse', False)

#distribution of residuals (log prices) - checking for normality
#scatter plot of price vs area
nox_dis_corr = round(data['price'].corr(data['area']), 3)
plt.figure(figsize=(16,10))
plt.scatter(x=data['price'], y=data['area'], alpha=0.6, s=80, color='#6200ea')
plt.title(f'price vs area (correlation {nox_dis_corr})', fontsize=10)
plt.xlabel('price', fontsize=10)
plt.ylabel('area', fontsize=10)
st.pyplot(bbox_inches='tight')
st.set_option('deprecation.showPyplotGlobalUse', False)

#log price with skew
