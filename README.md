# Paris' heartbeat
In order to gear up my data science skills, I recently became interested in Vélib bike-sharing data. Biking in Paris is a very common practice, at a point that the mayor Anne Hidalgo was maybe a bit too quick to [mention](https://www.leparisien.fr/international/hidalgo-suggere-plus-de-velos-a-kiev-les-dessous-dune-phrase-maladroite-01-12-2022-PNK3JE7ZHNFBZA5WTYDV65LD3U.php) (back in 2022 !!) bikes to Kiev for post-war reconstruction 🤨. 
To put the figures, 0.5 millions rides of Vélib occur per day, to be compared to 4 million daily rides on the metro and 1.1 million daily car trips within the 2.1 million inhabitants city.

While Vélib data is only a biased tracer of Paris's total motion (it is restricted to a non-representative subset of users), it still provides very insightful clues about urban dynamics. This journey through Parisian data is the perfect excuse to learn and practice key technical skills, including:
*    **Pandas** for data management and manipulation, **GeoPandas** for manipulating geographical data and producing maps.
*    **Matplotlib** and **Seaborn** for creating visualizations and graphs.
*    **Requests** and **APScheduler** for querying various APIs to collect live data and  for automating regular tasks.
*    **Tslearn**, **Sklearn**, **Xgboost**  and **Optuna** for clustering and machine learning

## Data collection
Vélib data is available via a live API but offers no history. I built a pipeline to collect it every 10 minutes during 15 days.

**Choice of frequency:** The choice of a 10-minute interval is directly related to the characteristic duration of a bike ride (20-30 minutes). This allows the capture of meaningful information about individual bike displacements, while avoiding redundant data in the dataset.. 
### Key Steps:
1.  **API request:** Data is fetched from two endpoints: station information (static) and status (dynamic).
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
2.  **Data cleaning:** Unnecessary columns are dropped using Pandas.
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
In my case, I use the decorator ``@sched.scheduled_job("interval", minutes=10)`` to fetch data every 10 minutes from 7th september to 22nd september 2025.
## Data inspection
Let's first look at the raw data ``df_status`` that was downloaded in step 1 above.

| station_id   | num_bikes_available | numBikesAvailable | num_bikes_available_types          | num_docks_available | numDocksAvailable | is_installed | is_returning | is_renting | last_reported | stationCode | station_opening_hours |
|-------------|---------------------|-------------------|------------------------------------|---------------------|-------------------|-------------|-------------|-----------|--------------|------------|----------------------|
| 213688169   | 3                   | 3                 | [{'mechanical': 1}, {'ebike': 2}]  | 32                  | 32                | 1           | 1           | 1         | 1757540462   | 16107      | None                 |
| 19179944124 | 9                   | 9                 | [{'mechanical': 7}, {'ebike': 2}]  | 16                  | 16                | 1           | 1           | 1         | 1757540775   | 40001      | None                 |
| 36255       | 5                   | 5                 | [{'mechanical': 5}, {'ebike': 0}]  | 16                  | 16                | 1           | 1           | 1         | 1757540598   | 9020       | None                 |

Some data are redundant and other fields are not relevant for this project, so I cleaned them in point 2. Regarding data privacy, note that there is no mention of user names nor specific bike trajectories (e.g., routes from point A to point B)—only the number of bikes available at each station. This design represents a balance between data privacy and open data policy.
For a comprehensive description of the fields in the table, the [doc](https://www.velib-metropole.fr/donnees-open-data-gbfs-du-service-velib-metropole) is a good place to go.

## Station-level time series analysis
I chose Saint-Sulpice station (out of 1,469) as an example to visualize how bike availability evolves over time.
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/station_availability.png)
<p align="center"><em>Bike and dock availability at Saint-Sulpice station over time</em></p>

- Clear patterns emerge—the advertised _heartbeat_ of Paris is visible. 
- Occasionally, the station is completely empty of bikes (bad luck for the next user! 🤯). Let's see if we can predict that!
- On September 10th, the amplitude of the “heartbeat” decreased significantly as it was raining cats and dogs that afternoon in Paris.
## Visualizing station occupancy and anomaly detection
Using ``GeoPandas`` and ``OpenStreetMap``, we can visualize bike availability across stations.
![Vélib Station Availability Map](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/Velib_availability.png)
<p align="center"><em>Geospatial Analysis of Station Availability</em></p>
Green indicates many available bikes, red indicates few bikes, and a cross (❌) marks stations with no bikes at all 🤯.
This visual inspection is worth it before delving into more involved data analysis. 

Using ``Isolation Forest``, an algorithm designed to detect "anomalies" in a given dataset, I could identify 150 (out of 1469) atypical stations including 32 station always full (over-utilization) and 22 stations always empty (under-utilization).
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/Anomaly.png)
<p align="center"><em>Flies in the ointment? An analysis of anomalous pattern in the Vélib data</em></p>
I analyzed the full datasets with all the timeframes, so the findings are a good tip for Vélib users 😉, though it remains to be checked whether, for instance, the altitude of the station impacts my claim of good tip 🥵.

## Beyond individual stations: sorting data by districts
The previous maps may look a bit cluttered and adopting a coarser point of view of the data will prove insightful. The official list of Paris districts can be found [here](https://opendata.paris.fr/explore/dataset/quartier_paris/information/), and is also available in this repository for reproducibility.
![Vélib District Availability](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/evoBYdistrict.png)
<p align="center"><em>Time evolution of Vélib availability in representative parisian districts</em></p>
Clear patterns emerge from this aggregated view: some stations fill up during the daytime (Bercy, Champs-Élysées), while some others are filling during the night (Belleville, Père-Lachaise). The heavy rainfall already mentionned in the afternoon of 10th september subtly altered these patterns. Bercy did not empty as much as usual, which consequently resulted in Père-Lachaise becoming emptier compared to sunnier days. Next, we will explore how to define clusters for all districts in Paris based on their usage patterns. If you live in Paris, you can discover which type of district you reside in according to the Vélib data!

## What Vélib tells us about districts: Residential, Business, or Tourism?
To categorize the temporal patterns of each district, I applied the k-means clustering algorithm to group each district's time series into one of k clusters. The data was first normalized using ``TimeSeriesScalerMeanVariance`` to standardize each series to a mean of 0 and a variance of 1. This crucial step eliminates biases related to absolute scale, such as the total number of bikes in a district or its overall usage rate, allowing the algorithm to focus on the shape of the usage patterns rather than their magnitude. The times series sorted in each cluster (black on my figure) will be "close" from the mean value of the cluster (colored in my figure).
The individual time series assigned to each cluster (shown in black) are "close" to their cluster's centroid (shown in color). After unsupervised training, I determined that k=3 provides the most physically interpretable results, a choice supported by quantitative criteria like the Silhouette score and the Elbow method.
![Time evolution of Vélib availability in 3 cluster identified by unsupervized learning](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/TimeEVOcluster.png)
<p align="center"><em>Time evolution of Vélib availability in 3 cluster identified by unsupervized learning</em></p>

As announced, the machine successfully identified three main types:
* **Residential**: A cluster that fills during the night.
* **Business**: A cluster that fills during the day but shows reduced activity on weekends.
* **Tourism**: A cluster that is consistently active every day of the week.

This classification aligns with our earlier observations: Belleville and Père-Lachaise belong to the residential cluster, while Bercy and Champs-Elysées are business-oriented. Arts et Métiers, with its museums and historical landmarks, was classified as tourism. The distinction between business and tourism is indeed more nuanced than the clear day/night pattern of residential areas.

To summarize my findings, I created a map of Paris with each district classified according to its Vélib activity pattern.
![Map of clusters in Paris](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/ALLcluster.png)
<p align="center"><em>Spatial distribution of district clusters identified from Vélib data</em></p>
So, do you agree with this unsupervized classification? In my case, my home is indeed in a "residential" era 🥳.

## Predicting Vélib availability: Can we predict how many bikes will be available at a given station at a given time?
After monitoring the heartbeats of Paris through Vélib stations and classifying districts into “Residential”, “Business”, and “Tourism”, a natural question is whether we can predict the future behavior of Vélib traffic. Against Betteridge's law, the answer is a tentative _yes_—with a little help from machine learning. Vélib traffic depends on several parameters, for this project, I selected the following key features:
1. **Temporal information**: Bike usage is extremely time-dependent. Commuters swarm the streets during morning and evening peaks, while weekends follow a more relaxed rhythm. I extracted the hour of the day and the day of the week as core  temporal features for the model.
2. **Spatial information**: Each station belongs to a specific district. I encoded these districts numerically and, crucially, incorporated the cluster type (“Residential”, “Business”, “Tourism”) we identified earlier. This allows the model to distinguish between a quiet residential area and a bustling tourist hub.
3. **External information**:As we saw previously, weather significantly impacts bike usage. A rainy day can disrupt commutes, emptying some stations while filling others with stranded bikes. I fetched hourly weather data ☀️🌧️💨 for Paris (temperature, precipitation, wind speed) using the ``meteostat`` library:
```python
location = Point(48.8566, 2.3522) #Paris, note there is only one meteo station in Paris (Montsouris)
start = occupation_data['hour_floor'].min()
end = occupation_data['hour_floor'].max()
weather = Hourly(location, start, end).fetch()[['temp','prcp','wspd']]
```
and merged it with our Vélib data by the hour. Now, each observation knows what the sky looked like when the bikes were counted. 

### Training the machine 🤖 
I used ``XGBoost`` a gradient boosting algorithm that builds an ensemble of decision trees sequentially, optimizing for the residual errors of the previous trees. This approach is particularly well-suited for capturing complex, non-linear interactions between features. The model learns to identify patterns across all input variables simultaneously. Its core call is:
```python
# Train XGBoost
model = xgb.XGBRegressor(
  n_estimators=300,
  learning_rate=0.05,
  max_depth=3,
  subsample=0.8,
  colsample_bytree=0.8,
  random_state=42,
  n_jobs=-1,
  tree_method="hist"
        )
```
where the hyperparameters of the model (``n_estimators``: number of decision trees, more trees increase capacity but also slow training and risk overfitting ; ``max_depth``: maximum depth of trees, controls model complexity ; ``learning_rate``: step size per tree, balances speed vs precision ; ``subsample``: fraction of data per tree, adds randomness, reduces overfit ; ``colsample_bytree``: fraction of features per tree, improves robustness) were chosen somewhat arbitrarily. In subsequent runs, I optimized the hyperparameters using ``optuna``.

To check the impact of the district and of the weather so I performed three runs with the following features : {District code + hour + dayofweek} , {District code + hour + dayofweek + weather } and {District code  + hour + dayofweek + weather + type}.

**Training Strategy** As my data are time-ordered, I employed a 5-fold ``TimeSeriesSplit``, training the model on progressively larger portions of the data (starting from ~1/6 of the dataset and growing to ~5/6), always testing on the subsequent chronological segment. This ensures the model is evaluated on future data relative to its training set, avoiding any look-ahead bias. This training strategy is optimal to have robust predictions 2-3 days in the future. Depending on the business case, the training strategy may be adapted. For instance, if the goal is customer-oriented (e.g., predicting bike availability 45 minutes in the future) or company-oriented (e.g., deciding how bikes should be redistributed to optimize availability), the training strategy may differ slightly (e.g., different folds, inclusion of lag features, etc.).

### Testing the crystal ball 🔮

After training, the model makes predictions on unseen data—time periods that come immediately after each training fold—ensuring it is always tested on future data it hasn’t encountered. I then measured the accuracy of the prediction using the mean of the RMSE (the typical error in number of bikes) and R² (how well the model explains the variability) over the 5 time folding:


| Run | Features                                          | RMSE | R²                                             |
| --- | ------------------------------------------------- | ---- | ---------------------------------------------- |
| 1   | District code + hour + dayofweek                  |   59.6   |    0.73                                          |
| 2   | District code + hour + dayofweek + weather        |   59.7   | 0.73 | 
| 3   | District code + hour + dayofweek + weather + type | <span style="background-color:#d4f7d4">55.7</span>     | <span style="background-color:#d4f7d4">0.76</span> |

This first try shows that adding the type of district helped a lot, while temperature doesn't seem to have much effect. It makes sense as except for the heavy rainfall of 10th september, the weather was pretty uniform during the time I fetched the Vélib data. 

### Tuning the hyperparameters with optuna
The hyperparameters left fixed in the previous section are now varied using ``optuna`` which allows for an optimal search of the ideal parameters. After 30 trials, it selected :  ``n_estimators=189,  learning_rate=0.11,  max_depth=5,
  subsample=0.97,  colsample_bytree=0.98`` as the best model with RMSE=50.1 and R²=0.91.


Here is a sample of the predictions:

| district       | type        | hour | dayofweek | temperature | precip | wind_speed | y_true | y_pred      |
|----------------|------------|------|-----------|------------|--------|------------|--------|------------|
| Arts-et-Métiers| Business    | 2    | 0         | 16.5       | 0.0    | 5.5        | 38     | 52.539246  |
| Belleville     | Residential | 3    | 6         | 15.6       | 0.0    | 3.7        | 30     | 79.792419  |
| Bercy          | Tourism     | 6    | 6         | 15.5       | 0.0    | 10.4       | 249    | 229.692734 |
| Champs-Elysées | Business    | 12   | 6         | 25.7       | 0.0    | 8.3        | 156    | 117.166687 |
| Père-Lachaise  | Residential | 0    | 6         | 16.8       | 0.0    | 9.7        | 125    | 129.630432 |

To get a more concrete sense of which features matter, I plotted a diagram of feature importance along with the prediction of the model vs the actual data :

![Performance of the model](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/resFIT.png)
<p align="center"><em>Contribution of each feature to the final result</em></p>

Ok, [state of the art](https://www.20minutes.fr/paris/1767487-20160118-paris-bike-predict-application-lit-avenir-stations-velib) a decade ago seemed to be 98% accurancy for the next 45 minutes using more than 80 features, so of course R²=0.9 is certainly perfectible. In the notebook, I added additional plots including time evolution of the residutes and a heatmap to see if the input features were correlated. 


All in all, I have had a fun time playing around this bikes data. [That](https://pierreauclair.org/blog/velibs.html) blog post was a good source of inspiration for the begining of this project. If I were to improve my model, I would incorporate additional features like station altitude, public holidays, and strike days, use a more powerful machine than my laptop, and—most importantly—train on a much larger dataset (for instance [that one](https://github.com/lovasoa/historique-velib-opendata)). I also [read](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) lag features (like the number of bikes available in the previous hour or the same hour on previous days) and rolling statistics (moving averages or rolling standard deviations) o capture persistence in bike usage. Essentially, by feeding the model both “what just happened” and “what has been happening,” it becomes much better at anticipating Paris’ heartbeats 🚴‍♂️💓, whether on a sunny weekday or a rainy afternoon. Coming from a physics background, I noted the common data science practice of often overlooking proper error propagation and uncertainty quantification (which I also omitted here); incorporating these, for instance in the district classification, would undoubtedly refine the results.



