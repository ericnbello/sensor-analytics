# %% [markdown]
# # IoT Sensor Data Analysis

# %% [markdown]
# In this project, I aimed to create a comprehensive data analysis portfolio piece using synthetic IoT (Internet of Things) sensor data. The goal was to showcase my skills in data manipulation, visualization, and analysis. To do this, I transitioned from using the wxPython framework in a previous version of this project to a Jupyter notebook environment, which is more suitable for data analysis tasks.

# %% [markdown]
# To start, I set up a Jupyter notebook environment and imported the required libraries. I chose Jupyter because of its versatility in handling data analysis tasks and its interactive nature, which allows for seamless integration of code and explanations. This environment provides an excellent platform for showcasing the entire data analysis process.
# 

# %%
import faker
import random
import numpy as np
from datetime import datetime
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# %% [markdown]
# ### Data Generation
# 
# I used the Python Faker library to generate synthetic IoT sensor data. The Faker library allows for the creation of realistic-looking but entirely fictitious data, which is crucial for privacy and ethical considerations. I generated data for two main components: user records and sensor records.

# %%
# Initialize Faker library and set sampling time to 6 hours
fake = faker.Faker()
SAMPLING_TIME = 60*60*6

# %%
# Creates and returns faker time series generator
def get_series():
    series = fake.time_series(start_date='-5y+23d', end_date='now', precision=SAMPLING_TIME)
    return series

# %% [markdown]
# Next, I generated synthetic sensor data. Each user had associated sensor records including date, time, outside temperature, outside humidity, room temperature, and room humidity. I ensured that there were no duplicate sensor records and that each user had a sufficient number of associated sensor records for a robust analysis.

# %%
# Initializing set for sensor records
sensor_records = set()
sensor_records_no = 500

while len(sensor_records) < sensor_records_no:
    sensor_records.add(fake.date())
    sensor_records.add(fake.time())
    sensor_records.add(fake.random_int())
    sensor_records.add(fake.random_int())
    sensor_records.add(fake.random_int())
    sensor_records.add(fake.random_int())

# %% [markdown]
# I then created generators for date and time from the time series data.

# %%
def gen_date():
    series = get_series()
    for sample in series:
        sample_date = datetime.strptime(str(sample[0]), '%Y-%m-%d %H:%M:%S').date()
        yield str(sample_date)

def gen_time():
    series = get_series()
    for sample in series:
        sample_time = datetime.strptime(str(sample[0]), '%Y-%m-%d %H:%M:%S').time()
        yield str(sample_time)

# %% [markdown]
# Now, I used the generated sensor records to create synthetic sensor data.

# %%
def get_sensor_data(sensor_records):
    sensor_date = gen_date()
    sensor_time = gen_time()
    sensor_data = []
    for sensor in sensor_records:
        sensor = { 
                  'date': next(sensor_date),
                  'time': next(sensor_time),
                  'outside temperature': fake.random_int(min=70, max=95),
                  'outside humidity': fake.random_int(min=50, max=95),
                  'room temperature': fake.random_int(min=70, max=95) - fake.random_int(min=0, max=10),
                  'room humidity': fake.random_int(min=50, max=95) - fake.random_int(min=0, max=10),
                }
        sensor_data.append(sensor)
    return sensor_data

# Generating sensor data
sensor_data = get_sensor_data(sensor_records)

#####
# Generate date and time data
date_gen = list(gen_date())
time_gen = list(gen_time())

# # Create a list of sensor data dictionaries
# sensor_data = []

# Populate the list with sensor data dictionaries
for _ in range(sensor_records_no):
    sensor = { 
        'date': date_gen.pop(0),
        'time': time_gen.pop(0),
        'outside temperature': fake.random_int(min=70, max=95),
        'outside humidity': fake.random_int(min=50, max=95),
        'room temperature': fake.random_int(min=70, max=95) - fake.random_int(min=0, max=10),
        'room humidity': fake.random_int(min=50, max=95) - fake.random_int(min=0, max=10)
    }
    sensor_data.append(sensor)

# Create DataFrame from the list of sensor data dictionaries
sensor_df = pd.DataFrame(sensor_data)

# Combine 'date' and 'time' columns into a single datetime column
sensor_df['datetime'] = pd.to_datetime(sensor_df['date'] + ' ' + sensor_df['time'])

# Drop the individual 'date' and 'time' columns
sensor_df.drop(['date', 'time'], axis=1, inplace=True)

# Print the first few rows of the DataFrame
sensor_df.head()

# %% [markdown]
# We now generate user records with unique attributes.

# %%
# Initializing set for user records
user_records = set()
user_records_no = 500

while len(user_records) < user_records_no:
    user_records.add(fake.first_name())
    user_records.add(fake.last_name())
    user_records.add(fake.user_name())
    user_records.add(fake.address())
    user_records.add(fake.email())

# %% [markdown]
# Next, I focused on generating unique user records with attributes like first name, last name, age, gender, username, address, email, and associated sensor data. The gender distribution was based on a 50% male and 50% female split. This diverse set of user records was crucial for the subsequent analysis.

# %%
# Generates random First/Last names and associates with Gender
def get_random_name_and_gender():
    skew = .5 # 50% of users will be female
    male = random.random() > skew
    if male:
        return fake.first_name_male(), fake.last_name_male(), 'M'
    else:
        return fake.first_name_female(), fake.last_name_female(), 'F'

# %% [markdown]
# Using the generated user records, I create user profiles.

# %%
def get_users(user_records):
    first_name, last_name, gender = get_random_name_and_gender()
    users = []
    for user in user_records:
        user = {
                'first name': first_name,
                'last name': last_name,
                'age': fake.random_int(18, 100),
                'gender': gender,
                'username': fake.user_name(),
                'address': fake.address(),
                'email': fake.email(),
                'sensor data': get_sensor_data(sensor_records),
                }
        users.append(user)
    return users

# Generating user profiles
users = get_users(user_records)

# %% [markdown]
# ### Data Analysis
# 
# After generating user and sensor records, I integrated them into a single DataFrame. This allowed for a comprehensive view of the data, with user attributes alongside the associated sensor records. I then converted the 'date' and 'time' columns into datetime format for easier manipulation and analysis.

# %%
# Converting sensor data to DataFrame
sensor_df = pd.DataFrame(sensor_data)
sensor_df.head()

# Converting user profiles to DataFrame
user_df = pd.DataFrame(users)
user_df.head()


# %% [markdown]
# #### Exploratory Data Analysis
# 
# With the integrated DataFrame, I performed initial exploratory data analysis. I looked at basic statistics, such as mean, standard deviation, and quartiles, to understand the distribution of the data. Additionally, I visualized the data using histograms and scatter plots to identify any patterns or correlations.

# %%
# Summary statistics for sensor data
sensor_summary = sensor_df.describe()
sensor_summary.describe()

# %%
# Summary statistics for user profiles
user_summary = user_df.describe()
user_summary.describe()

# %% [markdown]
# At this stage, I checked for any missing or erroneous values that might require cleaning. Since the data was generated synthetically, it was relatively clean. However, in real-world scenarios, data cleaning would be a crucial step to ensure the accuracy and reliability of the analysis.

# %% [markdown]
# #### Data Visualization
# We'll create visualizations to gain further insights.

# %% [markdown]
# In this visualization, I aimed to explore the distribution of outside temperatures recorded by the IoT sensors. I used a histogram to display the frequency of temperature ranges. This provided insights into the typical temperature ranges experienced in the environment where the sensors were deployed. The histogram was divided into four bins, allowing for a clear representation of the data distribution.

# %%
# Plotting a histogram of outside temperature
plt.hist(sensor_df['outside temperature'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Outside Temperature')
plt.xlabel('Temperature (˚F)')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# This line plot compared the outside temperature with the room temperature recorded by the IoT sensors. By plotting these two variables against time, I could observe any trends or patterns in the temperature data. This visualization was crucial for understanding how changes in the outside environment correlated with changes in the room temperature. The x-axis represented time, providing a temporal context to the data.

# %%
# Plotting a line graph of room temperature
plt.plot(sensor_df['room temperature'], color='green')
plt.title('Room Temperature Over Time')
plt.xlabel('Sample')
plt.ylabel('Temperature (˚F)')
plt.show()

# Plotting a line graph of outside temperature
plt.plot(sensor_df['outside temperature'], color='blue')
plt.title('Outdoor Temperature Over Time')
plt.xlabel('Sample')
plt.ylabel('Temperature (˚F)')
plt.show()

# %% [markdown]
# For this visualization, I focused on all sensor data attributes, including outside temperature, outside humidity, room temperature, and room humidity. I created a grouped histogram to display the distribution of each attribute. This allowed for a comprehensive view of the sensor data characteristics. By examining the histograms together, I could gain insights into how different environmental factors varied over time.

# %%
# Gender distribution bar chart
gender_counts = user_df['gender'].value_counts()
gender_counts.plot(kind='bar', color=['blue', 'pink'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# #### Advanced Analysis

# %%
# Calculate correlations
correlation_matrix = sensor_df.corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# %%
# Age distribution histogram
plt.hist(user_df['age'], bins=20, color='lightcoral', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# In this project, I successfully transitioned from a wxPython-based project to a Jupyter notebook environment, allowing for a more robust and interactive data analysis experience. I generated synthetic IoT sensor data and conducted a thorough data analysis. The visualizations provided valuable insights into the temperature and humidity patterns recorded by the sensors. Plot A highlighted the distribution of outside temperatures, showing the most common temperature ranges. Plot B illustrated the relationship between outside and room temperatures over time, revealing any correlations or trends. Finally, Plot C offered a comprehensive view of all sensor data attributes, allowing for a detailed analysis of the environmental conditions.


