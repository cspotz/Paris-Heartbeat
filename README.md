# Paris' heartbeat
In order to gear up my data science skills, I recently became interested in Vélib bike-sharing data. Biking in Paris is a very common practice, at a point that the mayor Anne Hidalgo was maybe a bit too quick to [mention](https://www.leparisien.fr/international/hidalgo-suggere-plus-de-velos-a-kiev-les-dessous-dune-phrase-maladroite-01-12-2022-PNK3JE7ZHNFBZA5WTYDV65LD3U.php) (back in 2022 !!) bikes to Kiev for post-war reconstruction 🤨. 
To put the figures, 0.5 millions rides of Vélib occur per day, to be compared to 4 million daily rides on the metro and 1.1 million daily car trips within the 2.1 million inhabitants city.

While Vélib data is only a biased tracer of Paris's total motion (it is restricted to a non-representative subset of users), it still provides very insightful clues about urban dynamics. This journey through Parisian data is the perfect excuse to learn and practice key technical skills, including:
*    **Pandas** for data management and manipulation.
*    **GeoPandas** for manipulating geographical data and producing maps.
*    **Matplotlib** and **Seaborn** for creating visualizations and graphs.
*    **Requests** and **APScheduler** for querying various APIs to collect live data and  for automating regular tasks.
*    **Tslearn**, **sklearn** and **xgboost** for clustering and machine learning.

## Data collection
Vélib data is available via a live API but offers no history. I built a pipeline to collect it every 10 minutes during 15 days.

**Choice of frequency:** The choice of a 10-minute interval is directly related to the characteristic duration of a bike ride (20-30 minutes). This allows the capture of meaningful information about individual bike displacements, while avoiding redundant data in the dataset.. 
### Key Steps:
1.  **API Request:** Data is fetched from two endpoints: station information (static) and status (dynamic).
```python
#URLs of APIs Vélib'
URL_INFO = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
URL_STATUS = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
# Fetch data from both endpoints
answer_info = requests.get(URL_INFO).json()
answer_status = requests.get(URL_STATUS).json()
# Convert to DataFrames
df_info = pd.DataFrame(answer_info["data"]["stations"])
df_status = pd.DataFrame(answer_status["data"]["stations"])
```  
2.  **Data Cleaning:** Unnecessary columns are dropped using Pandas.
```python
# Clean static data
df_info.drop(columns=["stationCode", "rental_methods"], inplace=True)
# Clean dynamic data and add timestamp
columns_to_drop = ["numBikesAvailable", "num_bikes_available_types", "numDocksAvailable", 
                   "is_installed", "is_returning", "is_renting", "last_reported"]
df_status.drop(columns=columns_to_drop, inplace=True)
df_status["time_stamp"] = pd.Timestamp.now()  # Create time series
```
3.  **Storage:** Data is saved to an SQLite database, with status data appended to create a time series.
```python
# Save to SQLite database
conn = sqlite3.connect("velib_data.db")
df_info.to_sql("localisation", conn, if_exists="replace", index=False)  # Static reference
df_status.to_sql("stations", conn, if_exists="append", index=False)     # Time series log
conn.close()
```
4.  **Automation & Scheduling:** The entire process is automated using APScheduler to run periodically.
```python
# Configure the scheduler
sched = BlockingScheduler()
start_time = datetime.now()
end_time = start_time + timedelta(hours=360)  # 15 days

# Schedule the job to run every 10 minutes
@sched.scheduled_job("interval", minutes=10)
def scheduled_job():
    job_velib()  # Function that executes steps 1-3
    
    # Automatic shutdown after 15 days
    if datetime.now() >= end_time:
        sched.shutdown()
# Start the automated data collection
sched.start()
```
In my case, I run it every 10 minutes from 7th september to 22th september 2025.
## Data inspection
Let's first look at the raw data ``df_status`` that was downloaded in point 1 above.

| station_id   | num_bikes_available | numBikesAvailable | num_bikes_available_types          | num_docks_available | numDocksAvailable | is_installed | is_returning | is_renting | last_reported | stationCode | station_opening_hours |
|-------------|---------------------|-------------------|------------------------------------|---------------------|-------------------|-------------|-------------|-----------|--------------|------------|----------------------|
| 213688169   | 3                   | 3                 | [{'mechanical': 1}, {'ebike': 2}]  | 32                  | 32                | 1           | 1           | 1         | 1757540462   | 16107      | None                 |
| 19179944124 | 9                   | 9                 | [{'mechanical': 7}, {'ebike': 2}]  | 16                  | 16                | 1           | 1           | 1         | 1757540775   | 40001      | None                 |
| 36255       | 5                   | 5                 | [{'mechanical': 5}, {'ebike': 0}]  | 16                  | 16                | 1           | 1           | 1         | 1757540598   | 9020       | None                 |

Some data are redundant and some other fields are not relevant for this project, so I cleaned them in point 2. Regarding data privacy, note that there is no mention of user names nor specific bike trajectories (e.g., routes from point A to point B)—only the number of bikes available at each station. This design represents a balance between data privacy and open data policy.
For a comprehensive description of the quantities in the table, the [doc](https://www.velib-metropole.fr/donnees-open-data-gbfs-du-service-velib-metropole) is a good place to go.

## Station-Level Time Series Analysis
I chose Saint-Sulpice station (out of 1,469) as an example to visualize how bike availability evolves over time.
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/station_availability.png)
<p align="center"><em>Bike and dock availability at Saint-Sulpice station over time</em></p>

- Clear patterns emerge—the advertised _heart beats_ of Paris is visible. 
- Occasionally, the station is completely empty of bikes (bad luck for the next user! 🤯). Let's see if we can predict that!
- On September 10th, the amplitude of the “heartbeat” decreased significantly as it was raining cats and dogs that afternoon in Paris.
## Visualizing Station Occupancy and Anomaly detection
Using ``GeoPandas`` and ``OpenStreetMap``, it is possible to visualize, the availability of the bikes in the stations. 
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/Velib_availability.png)
<p align="center"><em>Geospatial Analysis of Station Availability</em></p>
Green means a lot of bikes and red few bikes, while a cross indicate no bikes at all 🤯
This visual inspection is worth it before delving into more involved data analysis. 

Using ``Isolation Forest``, an algorithm to detect "anomalies" in a given dataset, I could identify 147 (out of 1469) atypical stations including 16 station always full (over-utilization) and 5 stations always empty (under-utilization).
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/Anomaly.png)
<p align="center"><em>Flies in the ointment? An analysis of anomalous pattern in the Vélib data</em></p>
I analyzed the full datasets with all the timeframes, so the results is a good tip for Vélib users 😉, though it remains to be checked whether for instance the altitude of the station impacts my claim of good tip 🥵.

## Beyond individual stations: sorting data by districts
The previous maps may look a bit cluttered and adopting a coarser point of view of the data will prove insightful. The official list of Paris' districts can be found [here](https://opendata.paris.fr/explore/dataset/quartier_paris/information/), and I also added it to the GitHub repo.
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/evoBYdistrict.png)
<p align="center"><em>Time evolution of Vélib availability in some representative districts of Paris</em></p>
Some pattern clearly emerge : some stations are filling up during the daytime (Bercy, Champs-Elysée), while some others are filling during the night (Belleville, Père Lachaise). The heavy rainfall already mentionned in the afternoon of 10th september impact sighly this picture : Bercy did not empty as much as usual resulting that Père-Lachaise became emptier as previous (somewhat) sunnier days. We will not see how it is possible to define clusters for each district of Paris, and if you live in Paris, you can check in which type of district you live according to my velib data !

## What Vélib tells us about districts: Residential, Business, Tourism?
To sort the date of each district, I used the k-means clustering algorithm to sort each time series of each district into one (out of k) cluster. I normalize my data using ``TimeSeriesScalerMeanVariance`` to ensure that each series has a mean of 0 and a variance of 1. This step is very important because it allows us to eliminate many biases, such as the absolute number of bikes in each district or the proportion of residents using the network. The times series sorted in each cluster (black on my figure) will be "close" from the mean value of the cluster (colored in my figure). After this unsupervized training, I have checked that k=3 offers a good physical interpretation and is backed up by criterions to choose such as Silhouette score and Elbow method.
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/TimeEVOcluster.png)
<p align="center"><em>Time evolution of Vélib availability in 3 cluster identified by unsupervized learning</em></p>

As announced, the machine managed to identify three main type, one cluster which fills during the night that we will dub "residential" and two which fill during the day that we will dub "business" (as it shuts down its activity during the week end) and another "tourism" as it is active every day. As studied before, indeed Belleville and Père Lachaise belong to residential cluster. Art-et-Métier and Champs-Elysée belongs to buisiness, while Bercy to tourism. Of course, the difference between tourism and business is more subtle than with residential. To sum up my findings, I propose you a map of Paris with each district classified with its Vélib activity.

![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/ALLcluster.png)
<p align="center"><em>Interpretation of each cluster identified with Vélib datag</em></p>
So do you agree with this unsupervized training? In my case, my home is indeed in a "residential" era 🥳.

## Predicting Vélib availability: Can we predict how many bikes will be available at a given station at a given time?
After monitoring the heartbeats of Paris through Vélib stations and classifying districts into “Residential”, “Business”, and “Tourism”, a natural question is whether we can predict the future behavior of Vélib traffic. To titiller Betteridge, the answer is _yes_, we can… with a little help from machine learning, weather, and time. The Vélib traffic depends on several parameters, for that simple project, I selected some :
1. **Temporal information**: Bike usage is extremely time-dependent. Commuters swarm the streets during morning and evening peaks, while weekends follow a more relaxed rhythm. I extracted the hour of the day and the day of the week as features for the model.
2. **Spatial information**: Each station belongs to a district. I encoded these districts numerically for the model and also added the cluster type (“Residential”, “Business”, “Tourism”) we identified earlier. Now the model knows whether it’s dealing with a sleepy residential area or a tourist hotspot buzzing with bikes.
3. **External information**: As we saw before, a rainy day can ruin your morning commute and empty a lot of stations… or fill them with stranded bikes. I fetched hourly weather ☀️🌧️💨 for Paris (temperature, precipitation, wind) with ``meteostat``:
```python
location = Point(48.8566, 2.3522) #Paris, note there is only one meteo station in Paris (Montsouris)
start = occupation_data['hour_floor'].min()
end = occupation_data['hour_floor'].max()
weather = Hourly(location, start, end).fetch()[['temp','prcp','wspd']]
```
and merged it with our Vélib data by the hour. Now, each observation knows what the sky looked like when the bikes were counted. 

### Training the machine 🤖 
I trained the machine using the popular choice ``XGBoost`` in these situation: a gradient boosting algorithm that handles complex interactions very well. The model learns patterns across the different inputs simultaneously. I wanted to check the impact of the district and of the weather so I performed three runs with the following input parameters: 

### Testing the crystal ball 🔮

After training, I let the model make predictions on unseen data, here is a sample of the predictions:

| district       | type        | hour | dayofweek | temperature | precip | wind_speed | y_true | y_pred      |
|----------------|------------|------|-----------|------------|--------|------------|--------|------------|
| Arts-et-Métiers| Business    | 2    | 0         | 16.5       | 0.0    | 5.5        | 38     | 52.539246  |
| Belleville     | Residential | 3    | 6         | 15.6       | 0.0    | 3.7        | 30     | 79.792419  |
| Bercy          | Tourism     | 6    | 6         | 15.5       | 0.0    | 10.4       | 249    | 229.692734 |
| Champs-Elysées | Business    | 12   | 6         | 25.7       | 0.0    | 8.3        | 156    | 117.166687 |
| Père-Lachaise  | Residential | 0    | 6         | 16.8       | 0.0    | 9.7        | 125    | 129.630432 |


I then measured the accuracy of the prediction using RMSE (the typical error in number of bikes), R² (how well the model explains the variability):


| Run | Features                                          | RMSE | R²                                             |
| --- | ------------------------------------------------- | ---- | ---------------------------------------------- |
| 1   | District code + hour + dayofweek                  |   49.6   |    0.82                                            |
| 2   | District code + hour + dayofweek + weather        |   49.2   | 0.82 | 
| 3   | District code + type + hour + dayofweek + weather | <span style="background-color:#d4f7d4">36.4</span>     | <span style="background-color:#d4f7d4">0.90</span> |

Ok, [state of the art](https://www.20minutes.fr/paris/1767487-20160118-paris-bike-predict-application-lit-avenir-stations-velib) 10 years ago seemed to be 98% accurancy for the next 45 minutes using more than 80 features, so of course R²=0.9 is more than perfectible, but I was already happy to see that adding the type of district helped a lot, while temperature doesn't seems to have much effect, it makes sense as except from the heavy rainfull of 10th september, the weather was pretty uniform during the time I fetch the Vélib data. Of course, with more data, this will help. To get a more concrete sense of with feature matter, I plotted a diagram of feature importance along with the prediction of the model vs the actual data :
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/resFIT.png)
<p align="center"><em>Contribution of each feature to the final result</em></p>
In the notebook, I also add aditional plot including a heatmap to see if the input features were correlated.

All in all, I have a fun time playing around those bikes data, if I were to improve my model, I would add more features, such as altitude of the stations, holidays, strikes...I would also move to a larger computer than my laptop, and of course use a larger data set. My background of physicist makes me note than as usual in those data driven project, little car is given to the error bars (at that is what I chose to do here), and I guess adding them (for instance in the classification of the district would refine the game).



