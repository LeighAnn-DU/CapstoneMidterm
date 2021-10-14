#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:38:54 2021

@author: leighannkudloff
"""

# Midterm--Kudloff

import streamlit            as st
import geopandas as gpd
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_pkl(file):
    my_pkl=open(file,'rb')
    python_object = pickle.load(my_pkl)
    my_pkl.close()
    return(python_object)

airports = load_pkl("cityunique.pkl")
data2019 = pd.read_csv("Data2019.csv")
Flights=pd.Series(list(data2019["CityOrigin"])+list(data2019["CityDest"])).value_counts().reset_index().rename({"index": "Airport", 0: "Flights"}, axis=1)                  
CityDepDelays=data2019.groupby("CityOrigin").aggregate(depdelay=("DEP_DEL15", "sum")).reset_index().rename({"CityOrigin": "Airport"}, axis=1)
CityArrDelays = data2019.groupby("CityDest").aggregate(arrdelay=("ARR_DEL15", "sum")).reset_index().rename({"CityDest": "Airport"}, axis=1)
CityDelays = pd.merge(CityDepDelays, CityArrDelays, how ="inner", on="Airport")
CityDelays["Delays"]=CityDelays["depdelay"]+CityDelays["arrdelay"]
Airports = pd.DataFrame.from_dict(airports, orient="index").reset_index().rename({"index": "Airport", 
                                                                                  0: "Lat", 
                                                                                  1: "Long", 
                                                                                  2: "Time Zone"}, axis=1)
Airports=pd.merge(Airports, Flights, how="inner", on="Airport")
Airports = pd.merge(Airports, CityDelays, how="inner", on="Airport")
timezonecolors = {k: px.colors.qualitative.Dark24[i] for i, k in enumerate(Airports["Time Zone"].unique())}

Airports["Time_Colors"] = Airports["Time Zone"].replace(timezonecolors)
dropvars = ["DEP_DEL15", "ARR_DEL15","ORIGIN", "DEST", "TimeArr", "TimeDep", 
            "CityOrigin", "CityDest", "UTCDep", "UTCArr",  "TAIL_NUM", "ARR_DOM"]
catcols = ["OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "DIVERTED", "CANCELLED", "DELAY",
           "DAY_OF_WEEK", "DEP_TIME_BLK", "TZOrg", "TZDest"] 
numcols=[col for col in data2019.columns if col not in catcols+dropvars]
data2019[catcols]=data2019[catcols].astype("category")
data2019ML=data2019[catcols + numcols + ["DELAY"]].copy()
data2019ML["DELAY"]=data2019ML["DELAY"].astype(float)
data2019ML=data2019ML.replace({np.nan: 0})
data2019DESC = data2019.copy()
citymapdf= data2019DESC.copy()
citymapdf[["CANCELLED", "DELAY"]]=citymapdf[["CANCELLED", "DELAY"]].astype(float)
counts = citymapdf.groupby(["CityOrigin", "CityDest", "LatOrg", "LongOrg", "LatDest", "LongDest"])    ["TAIL_NUM"].count().reset_index().rename({"TAIL_NUM": "Number of Flights"}, axis=1)
citymapdf = citymapdf.groupby(["CityOrigin", "CityDest", "LatOrg", "LongOrg", "LatDest", "LongDest"])[["DELAY", "CANCELLED"]].aggregate(DELAY=("DELAY","sum"), CANCELLED=("CANCELLED","sum")).reset_index()
citymapdf=pd.merge(citymapdf,counts, how="inner", on=["CityOrigin", "CityDest", "LatOrg", "LongOrg", "LatDest", "LongDest"])
USTmap=gpd.read_file("http://code.highcharts.com/mapdata/countries/us/custom/us-all-territories.geo.json")
bigdelay = citymapdf.loc[citymapdf["DELAY"]>65].copy()
bigdelay.reset_index(drop=True, inplace=True)
bigcancel = citymapdf.loc[citymapdf["CANCELLED"]>8].copy()
bigcancel.reset_index(drop=True, inplace=True)
scaler = MinMaxScaler()
bigcancel["Line_Width"]=scaler.fit_transform(bigcancel[["CANCELLED"]])
bigcancel["Line_Width"] = bigcancel["Line_Width"].replace({0: 0.005})
X_train, X_test, Y_train, Y_test = train_test_split(data2019ML.drop("DELAY", axis = 1), 
                                                    data2019ML["DELAY"], test_size=.2, 
                                                    random_state=38, stratify = data2019ML["DELAY"])
pipeSSDT=load_pkl("FinalDT.pkl")
pipenumcols=pipeSSDT.named_steps["prep"].transformers[0][2]
pipecatcols=pipeSSDT.named_steps["prep"].transformers[1][2]
onehot=pipeSSDT.named_steps["prep"].transformers[1][1].fit(X_train[pipecatcols]).categories_                                                 
                                                 
def create_onehot_dummy_cols(cat_cols, onehot):
    '''returns new column with original columns followed by category level'''
    new_cols = []
    for i, code_array in enumerate(onehot):
        col = cat_cols[i]
        for code in code_array:
            new_cols.append(f'{col}_{code}')
    return(new_cols)
dummycategorycolumns = create_onehot_dummy_cols(pipecatcols, onehot)
featimportcolumns = pipenumcols + dummycategorycolumns
pipeSSRF = load_pkl("FinalRF.pkl")
data2019CORR = data2019ML.copy()
data2019CORR["DELAY"]= data2019CORR["DELAY"].astype(float)
timeblockcounts=data2019DESC["DEP_TIME_BLK"].value_counts().reset_index().rename({"index": "DEP_TIME_BLK", 
                                                                    "DEP_TIME_BLK": "Count"}, axis = 1)
timeblockcounts.reset_index(inplace = True)
airlineIDcounts = data2019DESC["OP_UNIQUE_CARRIER"].value_counts().reset_index().rename({"index": "Airline ID", 
                                                                    "OP_UNIQUE_CARRIER": "Count"}, axis = 1)
airlineIDcounts.reset_index(inplace = True)
DOWcounts = data2019DESC["DAY_OF_WEEK"].value_counts().reset_index().rename({"index": "Day of Week", 
                                                                    "DAY_OF_WEEK": "Count"}, axis = 1)
DOWcounts.reset_index(inplace = True)

# Visualizations
#st.sidebar.title("Slides")
#option = st.sidebar.selectbox("", ("Introduction", "Correlations", "Time Blocks", "Flights by Airline", "Flights by Day of Week", "Flights by Day of Month", "Airport with Most Flights--Pacific View", "Busiest US Airports", "Delay Box", "Routes with Most Delayed Flights--Pacific View", "Routes with Most Delayed Flights", "Cancelled-Box", "Routes with Most Cancelled Flights--Pacific View", "Routes with Most Cancelled Flights", "Airports with Most Delays", "Delayed Flights", "Data Analysis", "Looking at the Results", "Feature Importance-Decision Tree", "Feature Importance-Random Forest"))


st.markdown("# Predicting Flight Delays")
st.markdown("## Capstone Project--Fall 2021")
st.markdown("## Leigh Ann Kudloff")
st.markdown("")
st.markdown("## The Dataset")
st.markdown("A Kaggle dataset of US flights in January 2019")
st.markdown("Over 500,000 flights")
st.markdown("Features of interest include:")
st.markdown("* Date of Flight and Day of Week")
st.markdown("* Airport of Origin and Destination Airport")
st.markdown("* Airline")
st.markdown("* Time of Departure and Time of Arrival")
st.markdown("* Departure Delay or Arrival Delay (over 15 minutes)")
st.markdown("* Flight Cancelled or Diverted")
st.markdown("* Tail Number of Aircraft")
st.markdown("* Distance of Flight")
st.markdown("")
st.markdown("")
st.markdown("## Data Preparation")
st.markdown("* Dropped columns that were irrelevant or redundant (like airport and airline id #s)")
st.markdown("* Changed airline id number to names of airlines")
st.markdown("* Changed numerical days of week to names of days of week")
st.markdown("* Added a column for any delay to use as a target variable")
st.markdown("* Used IATA aircodes to find city/airport names")
st.markdown("* Added latitude and longitude for each airport and found time zone")
st.markdown("* Converted times to UTC time to calculate airtime")
st.markdown("* Created a column for time in air for each flight")
st.markdown("* Converted negative times to accommodate International Date Line")
st.markdown("* Deleted flights with unusual flight times and that started at midnight on January 31st")
st.markdown("* Created a data frame for visualizations and another for data analysis")
    
st.markdown("## Correlations")
fig1 = px.imshow(data2019CORR.corr(), color_continuous_scale=px.colors.diverging.Portland)
st.plotly_chart(fig1)
    
st.markdown("## Time Blocks")
fig2 = px.bar(timeblockcounts, x="index", y="Count", 
             labels={"index": "Departure Time Block"}, title = "Flights by Time Block")
fig2.update_xaxes(tickvals = timeblockcounts['index'],
                           ticktext = timeblockcounts["DEP_TIME_BLK"], tickfont_family="Arial Black")
fig2.update_layout(title_x=0.5)
st.plotly_chart(fig2)

st.markdown("## Flights by Airline")
fig3 = px.bar(airlineIDcounts, x="index", y="Count", labels={"index": "Airline ID"}, 
              title = "Flights per Airline")
fig3.update_xaxes(tickvals = airlineIDcounts['index'],
                           ticktext = airlineIDcounts["Airline ID"], tickfont_family="Arial Black")
fig3.update_layout(title_x=0.5)
st.plotly_chart(fig3)

st.markdown("## Flights by Day of Week")
fig4 = px.bar(DOWcounts, x="index", y="Count", labels={"index": "Day of Week"}, title = "Day of Week Counts", 
         text = "Count")
fig4.update_xaxes(tickvals = DOWcounts['index'], ticktext = DOWcounts["Day of Week"], 
             tickfont_family="Arial Black")
fig4.update_layout(title_x=0.5)
st.plotly_chart(fig4)

st.markdown("## Flights by Day of Month")
DOMcounts = data2019DESC["DAY_OF_MONTH"].value_counts().reset_index().rename({"index": "Day of Month", 
                                                                "DAY_OF_MONTH": "Count"}, axis = 1)
DOMcounts = DOMcounts.sort_values("Day of Month").reset_index(drop = True)
DOMcounts["Day of Week"] = (["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"]*5)[:31]
fig5 = px.bar(DOMcounts, x="Day of Month", y="Count", title = "Day of Month Counts", 
         text = "Count", color = "Day of Week")
fig5.update_layout(title_x=0.5)
st.plotly_chart(fig5)

st.markdown("## Airport with Most Flights--Pacific View")
fig6 = px.scatter_geo(Airports, lat="Lat", lon="Long", 
           color = "Time Zone", color_discrete_sequence=px.colors.qualitative.Dark24, hover_name="Airport", 
           geojson = USTmap, size = "Flights", width=800, height=500,
           title = "Airports with Most Flights", projection="natural earth")
fig6.update_traces(marker_sizemin=3)
st.plotly_chart(fig6)

st.markdown("## Busiest US Airports")
fig7 = px.scatter_geo(Airports, lat="Lat", lon="Long", 
           color = "Time Zone", color_discrete_sequence=px.colors.qualitative.Dark24, hover_name="Airport", 
           scope = "usa", size = "Flights",title = "Busiest Airports")
fig7.update_traces(marker_sizemin=3)
st.plotly_chart(fig7)

st.markdown("## Delay Box")
fig8 = px.box(citymapdf, x="DELAY")
st.plotly_chart(fig8)

st.markdown("## Routes with Most Delayed Flights--Pacific View")
fig9 = px.scatter_geo(Airports, lat="Lat", lon="Long", 
           color = "Time Zone", color_discrete_sequence=px.colors.qualitative.Dark24, hover_name="Airport", 
           geojson = USTmap, size = "Flights", width=800, height=500,
           title = "Routes with Most Delayed Flights", projection="natural earth")
for i in range(len(bigdelay)):
    fig9.add_trace(
    go.Scattergeo(
        lon=[bigdelay['LongOrg'][i], bigdelay['LongDest'][i]],
        lat=[bigdelay['LatOrg'][i], bigdelay['LatDest'][i]],
        mode='lines',
        line=dict(width=1),
        #opacity=float(citymapdf['airline_count'][i])/float(edge_df['airline_count'].max()),
        hoverinfo='text', 
        text=bigdelay['CityOrigin'][i] + ' --> ' + bigdelay['CityDest'][i] + "<br>" + "Delayed Flights: " + str(bigdelay["DELAY"][i])
    )
)
fig9.update_traces(marker_sizemin=3) 
st.plotly_chart(fig9)

st.markdown("## Routes with Most Delayed Flights")
fig10 = px.scatter_geo(Airports, lat="Lat", lon="Long", 
           color = "Time Zone", color_discrete_sequence=px.colors.qualitative.Dark24, hover_name="Airport", 
           scope = "usa", size = "Flights",title = "Routes with Most Delayed Flights")
for i in range(len(bigdelay)):
    fig10.add_trace(
    go.Scattergeo(
        lon=[bigdelay['LongOrg'][i], bigdelay['LongDest'][i]],
        lat=[bigdelay['LatOrg'][i], bigdelay['LatDest'][i]],
        mode='lines',
        line=dict(width=1),
        #opacity=float(citymapdf['airline_count'][i])/float(edge_df['airline_count'].max()),
        hoverinfo='text', 
        text=bigdelay['CityOrigin'][i] + ' --> ' + bigdelay['CityDest'][i] + "<br>" + "Delayed Flights: " + str(bigdelay["DELAY"][i])
    )
)
fig10.update_traces(marker_sizemin=3)
st.plotly_chart(fig10)

st.markdown("## Cancelled-Box")
fig11 = px.box(citymapdf, x="CANCELLED")
st.plotly_chart(fig11)

st.markdown("## Routes with Most Cancelled Flights--Pacific View")
fig12 = px.scatter_geo(Airports, lat="Lat", lon="Long", 
           color = "Time Zone", color_discrete_sequence=px.colors.qualitative.Dark24, hover_name="Airport", 
           geojson = USTmap, size = "Flights", title = "Routes with Most Cancelled Flights")
for i in range(len(bigcancel)):
    fig12.add_trace(
    go.Scattergeo(
        lon=[bigcancel['LongOrg'][i], bigcancel['LongDest'][i]],
        lat=[bigcancel['LatOrg'][i], bigcancel['LatDest'][i]],
        mode='lines',
        line=dict(width=1),
        #line=dict(width=bigcancel["Line_Width"][i]*5),
        #opacity=bigcancel["Line_Width"][i],
        hoverinfo='text', 
        text=bigcancel['CityOrigin'][i] + ' --> ' + bigcancel['CityDest'][i] + "<br>" + "Cancelled Flights: " + str(bigcancel["CANCELLED"][i])
    )
)
fig12.update_traces(marker_sizemin=3)
st.plotly_chart(fig12)

st.markdown("## Routes with Most Cancelled Flights")
fig13 = px.scatter_geo(Airports, lat="Lat", lon="Long", 
           color = "Time Zone", color_discrete_sequence=px.colors.qualitative.Dark24, hover_name="Airport", 
           scope = "usa", size = "Flights", title = "Routes with Most Cancelled Flights")
for i in range(len(bigcancel)):
    fig13.add_trace(
    go.Scattergeo(
        lon=[bigcancel['LongOrg'][i], bigcancel['LongDest'][i]],
        lat=[bigcancel['LatOrg'][i], bigcancel['LatDest'][i]],
        mode='lines',
        line=dict(width=1),
        #line=dict(width=bigcancel["Line_Width"][i]*5),
        #opacity=bigcancel["Line_Width"][i],
        hoverinfo='text', 
        text=bigcancel['CityOrigin'][i] + ' --> ' + bigcancel['CityDest'][i] + "<br>" + "Cancelled Flights: " + str(bigcancel["CANCELLED"][i])
    )
)
fig13.update_traces(marker_sizemin=3)
st.plotly_chart(fig13)

st.markdown("## Airports with Most Delays")
fig14 = px.scatter_geo(Airports, lat="Lat", lon="Long", 
           color = "Time Zone", color_discrete_sequence=px.colors.qualitative.Dark24, hover_name="Airport", 
           scope = "usa", size = "Delays",title = "Airports with Most Delays")
fig14.update_traces(marker_sizemin=3)
st.plotly_chart(fig14)

st.markdown("## Delayed Flights")
DelayData=data2019DESC["DELAY"].value_counts().reset_index()
DelayData["index"]=["On Time", "Delayed"]
fig15 = px.bar(DelayData, x="index", y="DELAY", title = "Delayed Flights", 
   labels = {"index": "Flight Status", "DELAY": "Number of Flights"}, 
   text = "DELAY", color = "index")
st.plotly_chart(fig15)

st.markdown("## Data Analysis")
st.markdown(" ")
st.image("Classifier Comparison.png")
st.markdown("* The Random Forest model was able to predict flight delays with 93% accuracy.")

st.markdown("## Looking at the Results")
st.markdown("* A similar dataset of flights in January 2020 was cleaned and prepared.")
st.markdown("* The three best models were used to analyze this new data and the results can be seen in this table:")
st.image("Comparing 2019 to 2020 Models.png")


st.markdown("## Feature Importance-Decision Tree")
featimport = pipeSSDT.named_steps["clf"].feature_importances_
importance_df = pd.DataFrame(zip(featimportcolumns, featimport),
                         columns=['Features','Importance'])
##Order from largest to smallest
importance_df = importance_df.sort_values('Importance',ascending = False).reset_index(drop=True)
#Add ranks column
importance_df['Rank'] = np.arange(1,len(importance_df)+1)
#Round the weights so they visualize better
importance_df['Importance'] = importance_df['Importance'].apply(lambda x: np.round(x,4))
fig16 = px.bar(importance_df.head(30), x="Importance",y="Features",color="Features",text="Importance",
                        title = 'Ranked Features',
                        color_discrete_sequence=px.colors.qualitative.Safe,width=1000,height=1000,
                       labels = {'Features':"",'Importance':'Feature Importance'})
fig16.update_layout(showlegend=False,
                             yaxis = dict(dtick=1,tickfont = dict(size = 14,color = 'black')),
                            xaxis = dict(titlefont = dict(size=14,color='black')),
                            font=dict(size=14))
st.plotly_chart(fig16)

st.markdown("## Feature Importance-Random Forest")
featimport = pipeSSRF.named_steps["clf"].feature_importances_
importance_df = pd.DataFrame(zip(featimportcolumns, featimport),
                         columns=['Features','Importance'])
importance_df = importance_df.sort_values('Importance',ascending = False).reset_index(drop=True)
importance_df['Rank'] = np.arange(1,len(importance_df)+1)
importance_df['Importance'] = importance_df['Importance'].apply(lambda x: np.round(x,4))
fig17 = px.bar(importance_df.head(20), x = "Importance",y="Features",color="Features",text="Importance",
                        title = 'Ranked Features',
                        color_discrete_sequence=px.colors.qualitative.Safe,width=1000,height=1000,
                       labels = {'Features':"",'Importance':'Feature Importance'})
fig17.update_layout(showlegend=False,
                             yaxis = dict(dtick=1,tickfont = dict(size = 14,color = 'black')),
                            xaxis = dict(titlefont = dict(size=14,color='black')),
                            font=dict(size=14))
st.plotly_chart(fig17)
    
    
    
    
     
    



















