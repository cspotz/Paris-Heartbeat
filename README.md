# Paris-Heartbeat
In order to gear up my data science skills, I recently became interested in Vélib bike-sharing data. Biking in Paris is a very common practice, at a point that the mayor Anne Hidalgo was maybe a bit too quick to [mention](https://www.leparisien.fr/international/hidalgo-suggere-plus-de-velos-a-kiev-les-dessous-dune-phrase-maladroite-01-12-2022-PNK3JE7ZHNFBZA5WTYDV65LD3U.php) (back in 2022 !!) bikes to Kiev for post-war reconstruction. To put the figures, 0.5 millions rides of Vélib occur per day, to be compared to 4 million daily rides on the metro and 1.1 million daily car trips within the 2.1 million inhabitants city.

While Vélib data is only a biased tracer of Paris's total motion (it is restricted to a non-representative subset of users), it still provides very insightful clues about urban dynamics. This journey through Parisian data is the perfect excuse to learn and practice key technical skills, including:
*    **Pandas** for data management and manipulation.
*    **Requests** and **APScheduler** for querying various APIs to collect live data and  for automating regular tasks.
*    **GeoPandas** for manipulating geographical data and producing maps.
*    **Matplotlib** and **Seaborn** for creating visualizations and graphs.
*    **Tslearn**, **sklearn** and **xgboost** for clustering and machine learning.

## Data collection
Vélib data is available via a live API but offers no history. I built a pipeline to collect it every 10 minutes during 15 days. The choice of a 10-minute interval is directly related to the characteristic duration of a bike ride (20-30 minutes). This allows the capture of meaningful information about individual bike displacements, while avoiding redundant data in the dataset.. 
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
In my case, I run it every 10 minutes from 7th september to 22th september 2025.
# Start the automated data collection
sched.start()
```
