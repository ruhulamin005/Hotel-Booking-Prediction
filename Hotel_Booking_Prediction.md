# Hotel Booking Prediction
### Md Ruhul Amin
##### mdruhul.amin@northsouth.edu
#### https://github.com/ruhulamin005/
#### Project Link: https://github.com/ruhulamin005/Hotel-Booking-Prediction


```python
#importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
#Reading dataset
df = pd.read_csv('hotel_bookings.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
df.shape
```




    (119390, 32)




```python
#Missing Values
df.isna().sum()
```




    hotel                                  0
    is_canceled                            0
    lead_time                              0
    arrival_date_year                      0
    arrival_date_month                     0
    arrival_date_week_number               0
    arrival_date_day_of_month              0
    stays_in_weekend_nights                0
    stays_in_week_nights                   0
    adults                                 0
    children                               4
    babies                                 0
    meal                                   0
    country                              488
    market_segment                         0
    distribution_channel                   0
    is_repeated_guest                      0
    previous_cancellations                 0
    previous_bookings_not_canceled         0
    reserved_room_type                     0
    assigned_room_type                     0
    booking_changes                        0
    deposit_type                           0
    agent                              16340
    company                           112593
    days_in_waiting_list                   0
    customer_type                          0
    adr                                    0
    required_car_parking_spaces            0
    total_of_special_requests              0
    reservation_status                     0
    reservation_status_date                0
    dtype: int64




```python
#filling missing value with 0 and updating data frame
def data_clean(df):
    df.fillna(0,inplace=True)
    print(df.isnull().sum())
```


```python
data_clean(df)
```

    hotel                             0
    is_canceled                       0
    lead_time                         0
    arrival_date_year                 0
    arrival_date_month                0
    arrival_date_week_number          0
    arrival_date_day_of_month         0
    stays_in_weekend_nights           0
    stays_in_week_nights              0
    adults                            0
    children                          0
    babies                            0
    meal                              0
    country                           0
    market_segment                    0
    distribution_channel              0
    is_repeated_guest                 0
    previous_cancellations            0
    previous_bookings_not_canceled    0
    reserved_room_type                0
    assigned_room_type                0
    booking_changes                   0
    deposit_type                      0
    agent                             0
    company                           0
    days_in_waiting_list              0
    customer_type                     0
    adr                               0
    required_car_parking_spaces       0
    total_of_special_requests         0
    reservation_status                0
    reservation_status_date           0
    dtype: int64
    


```python
df.columns
```




    Index(['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
           'arrival_date_month', 'arrival_date_week_number',
           'arrival_date_day_of_month', 'stays_in_weekend_nights',
           'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
           'country', 'market_segment', 'distribution_channel',
           'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'reserved_room_type',
           'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
           'company', 'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'reservation_status', 'reservation_status_date'],
          dtype='object')




```python
list=['adults', 'children', 'babies']
for i in list:
    print('{} has uniqie values as {}'.format(i,df[i].unique()))
```

    adults has uniqie values as [ 2  1  3  4 40 26 50 27 55  0 20  6  5 10]
    children has uniqie values as [ 0.  1.  2. 10.  3.]
    babies has uniqie values as [ 0  1  2 10  9]
    


```python
#creating a filter 
filter = (df['children']==0) & (df['adults']==0) & (df['babies']== 0)
df[filter]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2224</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2015</td>
      <td>October</td>
      <td>41</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>174.0</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>10/6/2015</td>
    </tr>
    <tr>
      <th>2409</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>0</td>
      <td>2015</td>
      <td>October</td>
      <td>42</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>174.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>10/12/2015</td>
    </tr>
    <tr>
      <th>3181</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>36</td>
      <td>2015</td>
      <td>November</td>
      <td>47</td>
      <td>20</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>11/23/2015</td>
    </tr>
    <tr>
      <th>3684</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>165</td>
      <td>2015</td>
      <td>December</td>
      <td>53</td>
      <td>30</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>122</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>1/4/2016</td>
    </tr>
    <tr>
      <th>3708</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>165</td>
      <td>2015</td>
      <td>December</td>
      <td>53</td>
      <td>30</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>308.0</td>
      <td>0.0</td>
      <td>122</td>
      <td>Transient-Party</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>1/5/2016</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115029</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>107</td>
      <td>2017</td>
      <td>June</td>
      <td>26</td>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>100.80</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>6/30/2017</td>
    </tr>
    <tr>
      <th>115091</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>1</td>
      <td>2017</td>
      <td>June</td>
      <td>26</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/1/2017</td>
    </tr>
    <tr>
      <th>116251</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>44</td>
      <td>2017</td>
      <td>July</td>
      <td>28</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>425.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>73.80</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/17/2017</td>
    </tr>
    <tr>
      <th>116534</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>2</td>
      <td>2017</td>
      <td>July</td>
      <td>28</td>
      <td>15</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient-Party</td>
      <td>22.86</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/22/2017</td>
    </tr>
    <tr>
      <th>117087</th>
      <td>City Hotel</td>
      <td>0</td>
      <td>170</td>
      <td>2017</td>
      <td>July</td>
      <td>30</td>
      <td>27</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/29/2017</td>
    </tr>
  </tbody>
</table>
<p>180 rows × 32 columns</p>
</div>




```python
#negation for final cleaned data
data = df[~filter]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>deposit_type</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>342</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>737</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/1/2015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>7</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>13</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>304.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>7/2/2015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel</td>
      <td>0</td>
      <td>14</td>
      <td>2015</td>
      <td>July</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>No Deposit</td>
      <td>240.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>7/3/2015</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



## Prtoblem Statement
#### 1. Where Do the Guests come from?
#### 2. How Much do guest pay for a night?


```python

#### 1. Where Do the Guests come from?

```


```python
#before spatial analysis 
data[data['is_canceled']==0]['country'].value_counts().reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PRT</td>
      <td>20977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GBR</td>
      <td>9668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FRA</td>
      <td>8468</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ESP</td>
      <td>6383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DEU</td>
      <td>6067</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>161</th>
      <td>BHR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>162</th>
      <td>BDI</td>
      <td>1</td>
    </tr>
    <tr>
      <th>163</th>
      <td>BWA</td>
      <td>1</td>
    </tr>
    <tr>
      <th>164</th>
      <td>PLW</td>
      <td>1</td>
    </tr>
    <tr>
      <th>165</th>
      <td>TJK</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>166 rows × 2 columns</p>
</div>




```python
country_wise_data = data[data['is_canceled']==0]['country'].value_counts().reset_index()
```


```python
#REnaming coloumn name
country_wise_data.columns = ['Country','No of Guests']
country_wise_data # This is for 1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>No of Guests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PRT</td>
      <td>20977</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GBR</td>
      <td>9668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FRA</td>
      <td>8468</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ESP</td>
      <td>6383</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DEU</td>
      <td>6067</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>161</th>
      <td>BHR</td>
      <td>1</td>
    </tr>
    <tr>
      <th>162</th>
      <td>BDI</td>
      <td>1</td>
    </tr>
    <tr>
      <th>163</th>
      <td>BWA</td>
      <td>1</td>
    </tr>
    <tr>
      <th>164</th>
      <td>PLW</td>
      <td>1</td>
    </tr>
    <tr>
      <th>165</th>
      <td>TJK</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>166 rows × 2 columns</p>
</div>




```python
#####Now we need folium for a MAP
#!pip install folium
```


```python
import folium
from folium.plugins import HeatMap
```


```python
basemap = folium.Map()
```


```python
#installing plotly
#!pip install plotly
```

    Collecting plotly
      Downloading plotly-4.14.3-py2.py3-none-any.whl (13.2 MB)
    Collecting retrying>=1.3.3
      Downloading retrying-1.3.3.tar.gz (10 kB)
    Requirement already satisfied: six in c:\users\user\anaconda3\lib\site-packages (from plotly) (1.14.0)
    Building wheels for collected packages: retrying
      Building wheel for retrying (setup.py): started
      Building wheel for retrying (setup.py): finished with status 'done'
      Created wheel for retrying: filename=retrying-1.3.3-py3-none-any.whl size=11435 sha256=c2de1fb090a749ca6abbcc37ca6fb7bd05cda648f77ab6abcb882c324ba61af9
      Stored in directory: c:\users\user\appdata\local\pip\cache\wheels\f9\8d\8d\f6af3f7f9eea3553bc2fe6d53e4b287dad18b06a861ac56ddf
    Successfully built retrying
    Installing collected packages: retrying, plotly
    Successfully installed plotly-4.14.3 retrying-1.3.3
    


```python
import plotly.express as px
```


```python
map_guest = px.choropleth(country_wise_data,
             locations = country_wise_data['Country'],
             color = country_wise_data['No of Guests'],
              hover_name = country_wise_data['Country'],
              title = 'Home Country of Guests'
             )
map_guest.show()
```


<div>                            <div id="e50b9b87-53a7-405d-b10d-2e9bb6b2d0d4" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e50b9b87-53a7-405d-b10d-2e9bb6b2d0d4")) {                    Plotly.newPlot(                        "e50b9b87-53a7-405d-b10d-2e9bb6b2d0d4",                        [{"coloraxis": "coloraxis", "geo": "geo", "hovertemplate": "<b>%{hovertext}</b><br><br>Country=%{location}<br>No of Guests=%{z}<extra></extra>", "hovertext": ["PRT", "GBR", "FRA", "ESP", "DEU", "IRL", "ITA", "BEL", "NLD", "USA", "BRA", "CHE", "AUT", "CN", "SWE", "POL", "CHN", "ISR", "NOR", 0, "RUS", "FIN", "ROU", "DNK", "AUS", "LUX", "JPN", "ARG", "AGO", "HUN", "MAR", "TUR", "CZE", "IND", "SRB", "GRC", "DZA", "KOR", "MEX", "HRV", "LTU", "NZL", "EST", "BGR", "IRN", "ISL", "ZAF", "CHL", "COL", "UKR", "MOZ", "LVA", "SVK", "SVN", "THA", "CYP", "TWN", "MYS", "PER", "URY", "SGP", "LBN", "EGY", "TUN", "ECU", "CRI", "JOR", "BLR", "PHL", "SAU", "VEN", "KAZ", "OMN", "IRQ", "NGA", "MLT", "CPV", "IDN", "KWT", "PRI", "BIH", "CMR", "ALB", "BOL", "PAN", "LBY", "MKD", "CUB", "AZE", "ARE", "GNB", "GEO", "GIB", "LKA", "JAM", "ARM", "VNM", "DOM", "MUS", "CAF", "PAK", "SUR", "CIV", "QAT", "PRY", "BRB", "KEN", "GTM", "MCO", "MNE", "SYR", "SEN", "MDV", "BGD", "HKG", "LIE", "AND", "ABW", "TGO", "SLV", "GAB", "STP", "COM", "UZB", "RWA", "UGA", "ZWE", "MWI", "LAO", "TZA", "ETH", "GHA", "ATA", "KNA", "TMP", "BHS", "ATF", "MDG", "SMR", "FRO", "CYM", "MAC", "DMA", "LCA", "NAM", "NPL", "KIR", "SLE", "SYC", "ZMB", "PYF", "NCL", "MMR", "BFA", "SDN", "MLI", "MRT", "AIA", "DJI", "ASM", "GUY", "BHR", "BDI", "BWA", "PLW", "TJK"], "locations": ["PRT", "GBR", "FRA", "ESP", "DEU", "IRL", "ITA", "BEL", "NLD", "USA", "BRA", "CHE", "AUT", "CN", "SWE", "POL", "CHN", "ISR", "NOR", 0, "RUS", "FIN", "ROU", "DNK", "AUS", "LUX", "JPN", "ARG", "AGO", "HUN", "MAR", "TUR", "CZE", "IND", "SRB", "GRC", "DZA", "KOR", "MEX", "HRV", "LTU", "NZL", "EST", "BGR", "IRN", "ISL", "ZAF", "CHL", "COL", "UKR", "MOZ", "LVA", "SVK", "SVN", "THA", "CYP", "TWN", "MYS", "PER", "URY", "SGP", "LBN", "EGY", "TUN", "ECU", "CRI", "JOR", "BLR", "PHL", "SAU", "VEN", "KAZ", "OMN", "IRQ", "NGA", "MLT", "CPV", "IDN", "KWT", "PRI", "BIH", "CMR", "ALB", "BOL", "PAN", "LBY", "MKD", "CUB", "AZE", "ARE", "GNB", "GEO", "GIB", "LKA", "JAM", "ARM", "VNM", "DOM", "MUS", "CAF", "PAK", "SUR", "CIV", "QAT", "PRY", "BRB", "KEN", "GTM", "MCO", "MNE", "SYR", "SEN", "MDV", "BGD", "HKG", "LIE", "AND", "ABW", "TGO", "SLV", "GAB", "STP", "COM", "UZB", "RWA", "UGA", "ZWE", "MWI", "LAO", "TZA", "ETH", "GHA", "ATA", "KNA", "TMP", "BHS", "ATF", "MDG", "SMR", "FRO", "CYM", "MAC", "DMA", "LCA", "NAM", "NPL", "KIR", "SLE", "SYC", "ZMB", "PYF", "NCL", "MMR", "BFA", "SDN", "MLI", "MRT", "AIA", "DJI", "ASM", "GUY", "BHR", "BDI", "BWA", "PLW", "TJK"], "name": "", "type": "choropleth", "z": [20977, 9668, 8468, 6383, 6067, 2542, 2428, 1868, 1716, 1592, 1392, 1298, 1033, 1025, 793, 703, 537, 500, 426, 421, 391, 377, 366, 326, 319, 177, 169, 160, 157, 153, 150, 146, 134, 116, 98, 93, 82, 78, 75, 75, 74, 68, 65, 63, 59, 53, 49, 49, 48, 48, 48, 46, 41, 41, 41, 40, 37, 25, 23, 23, 22, 22, 21, 20, 19, 18, 18, 17, 15, 15, 14, 14, 14, 14, 13, 13, 12, 11, 10, 10, 10, 10, 10, 10, 9, 8, 8, 8, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}],                        {"coloraxis": {"colorbar": {"title": {"text": "No of Guests"}}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "geo": {"center": {}, "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}, "legend": {"tracegroupgap": 0}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Home Country of Guests"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e50b9b87-53a7-405d-b10d-2e9bb6b2d0d4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
#### 2. How Much do guest pay for a night?

```


```python
#new data processing
data2 = data[data['is_canceled']==0]
```


```python
data2.columns
```




    Index(['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
           'arrival_date_month', 'arrival_date_week_number',
           'arrival_date_day_of_month', 'stays_in_weekend_nights',
           'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
           'country', 'market_segment', 'distribution_channel',
           'is_repeated_guest', 'previous_cancellations',
           'previous_bookings_not_canceled', 'reserved_room_type',
           'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
           'company', 'days_in_waiting_list', 'customer_type', 'adr',
           'required_car_parking_spaces', 'total_of_special_requests',
           'reservation_status', 'reservation_status_date'],
          dtype='object')




```python
#Need price distribution
plt.figure(figsize = (12,8))
sns.boxplot(x='reserved_room_type',y='adr',hue='hotel',data=data2)
plt.title('Price of room types per night & per person')
plt.xlabel('Room Type')
plt.ylabel('Price in Euro')
plt.legend()
plt.show()
```


![png](output_25_0.png)



```python

```
