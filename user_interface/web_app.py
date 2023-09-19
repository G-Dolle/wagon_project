import streamlit as st
import pandas as pd
import requests
import datetime
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Web app",
    page_icon="🌍",
    layout="centered",  # wide
    initial_sidebar_state="auto")

#Title of the app
st.title("THIS WEB APP PREDICTS THE AVAILABILITY OF \U0001F6B2 'S FOR DIVVY OVER THE NEXT FIVE DAYS")

st.header("Select the date \U0001F4C5 and time \U0001F551 to explore Divvy's stations traffic")

today=datetime.date.today()
max=today+datetime.timedelta(days=5)

now=datetime.datetime(2023,1,8,0,0,0).time()
delta=datetime.timedelta(hours=23)
max_hour=(datetime.datetime.combine(today,now)+delta).time()

#Obtain the user inputs
with st.form('Please provide the some inputs'):
    departure_date=st.slider('Select a date',
                             min_value=today,
                             max_value=max,
                             step=datetime.timedelta(hours=3),
                             value=datetime.datetime(2023,1,9,0,0,0).date())
    departure_time=st.slider('Select a time',
                             min_value=now,
                             max_value=max_hour,
                             step=datetime.timedelta(hours=3),
                             value=datetime.datetime(2023,1,9,9,0,0).time())
    st.form_submit_button("Let's explore Divvy's stations traffic")
dep_datetime=f'{departure_date} {departure_time}'
user_input= datetime.datetime.strptime(dep_datetime,'%Y-%m-%d %H:%M:%S')
if user_input>datetime.datetime.now():
    st.success("All is set")
    st.write(dep_datetime)
else:
    st.error("Let's pick a date/time in the future")



#Executing the API
url="https://bikespredv2-z2pwa4emra-ew.a.run.app/predict"
params = {
'year_input':str(user_input.year),
'month_input':str(user_input.month),
'day_input':str(user_input.day),
'hour_input' : str(user_input.hour),
'min_input' : str(user_input.minute),
'sec_input' : str(user_input.second)
}

response=requests.get(url=url, params=params)
outcome=response.json()

#Extracting info from API response to streamlit
stations=pd.DataFrame.from_dict(response.json())
#summer_stations=pd.DataFrame.from_dict(outcome['summer_stations'])

st.header("These are divvy stations' availability on the selected date")
px.set_mapbox_access_token('pk.eyJ1Ijoia3Jva3JvYiIsImEiOiJja2YzcmcyNDkwNXVpMnRtZGwxb2MzNWtvIn0.69leM_6Roh26Ju7Lqb2pwQ')
fig=fig=px.scatter_mapbox(stations,
                      lat="lat", lon="lon",
                      color="prediction",
                      #size="traffic",
                      hover_data=["name"],
                      color_discrete_sequence=px.colors.qualitative.Alphabet,
                      zoom=10,
                      height=770)
st.plotly_chart(fig)
