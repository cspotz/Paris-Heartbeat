# Paris' heartbeat
In order to gear up my data science skills, I recently became interested in Vélib bike-sharing data. Biking in Paris is a very common practice, at a point that the mayor Anne Hidalgo was maybe a bit too quick to [mention](https://www.leparisien.fr/international/hidalgo-suggere-plus-de-velos-a-kiev-les-dessous-dune-phrase-maladroite-01-12-2022-PNK3JE7ZHNFBZA5WTYDV65LD3U.php) (back in 2022 !!) bikes to Kiev for post-war reconstruction 🤨. 
To put the figures, 0.5 millions rides of Vélib occur per day, to be compared to 4 million daily rides on the metro and 1.1 million daily car trips within the 2.1 million inhabitants city.

While Vélib data is only a biased tracer of Paris's total motion (it is restricted to a non-representative subset of users), it still provides very insightful clues about urban dynamics. This journey through Parisian data is the perfect excuse to learn and practice key technical skills, including:
*    **Pandas** for data management and manipulation.
*    **Requests** and **APScheduler** for querying various APIs to collect live data and  for automating regular tasks.
*    **GeoPandas** for manipulating geographical data and producing maps.
*    **Matplotlib** and **Seaborn** for creating visualizations and graphs.
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
- On September 10th, the amplitude of the “heartbeat” decreased significantly due to heavy rainfall in Paris.
## Visualizing Station Occupancy and Anomaly detection
Using GeoPandas and OpenStreetMap, it is possible to visualize, the availability of the bikes in the stations. 
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/Velib_availability.png)
<p align="center"><em>Geospatial Analysis of Station Availability</em></p>
Green means a lot of bikes and red few bikes, while a cross indicate no bikes at all 🤯
This visual inspection is worth it before delving into more involved data analysis. 

Using Isolation Forest, an algorithm to detect "anomalies" in a given dataset, I could identify 147 (out of 1469) atypical stations including 16 station always full (over-utilization) and 5 stations always empty (under-utilization). A good tip for Vélib users 😉, though it remains to be checked whether the altitude of the station impacts all of my analysis.
![Vélib Station Availability Chart](https://github.com/cspotz/Paris-Heartbeat/blob/main/images/Velib_availability.png)
<p align="center"><em>Bike and dock availability at Saint-Sulpice station over time</em></p>
