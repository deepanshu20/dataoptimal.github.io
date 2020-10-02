# Cohort Analysis with Python

What is cohort analysis?
A cohort is a group of users who share something in common, be it their sign-up date, first purchase month, birth date, acquisition channel, etc. Cohort analysis is the method by which these groups are tracked over time, helping you spot trends, understand repeat behaviors (purchases, engagement, amount spent, etc.), and monitor your customer and revenue retention.

It’s common for cohorts to be created based on a customer’s first usage of the platform, where "usage" is dependent on your business’ key metrics. For Uber or Lyft, usage would be booking a trip through one of their apps. For GrubHub, it’s ordering some food. For AirBnB, it’s booking a stay.
With these companies, a purchase is at their core, be it taking a trip or ordering dinner — their revenues are tied to their users’ purchase behavior.

In others, a purchase is not central to the business model and the business is more interested in "engagement" with the platform. Facebook and Twitter are examples of this - are you visiting their sites every day? Are you performing some action on them - maybe a "like" on Facebook or a "favorite" on a tweet?1

When building a cohort analysis, it’s important to consider the relationship between the event or interaction you’re tracking and its relationship to your busines

# Why is it valuable?
Cohort analysis can be helpful when it comes to understanding your business’ health and "stickiness" - the loyalty of your customers. Stickiness is critical since it’s far cheaper and easier to keep a current customer than to acquire a new one. For startups, it’s also a key indicator of product-market fit.

Additionally, your product evolves over time. New features are added and removed, the design changes, etc. Observing individual groups over time is a starting point to understanding how these changes affect user behavior.

It’s also a good way to visualize your user retention/churn as well as formulating a basic understanding of their lifetime value.

# Cohort Analysis (Retention over User & Product Lifetime)

![](http://blog.appsee.com/wp-content/uploads/2018/06/action-cohort-analysis-4.png)

![](http://d35fo82fjcw0y8.cloudfront.net/2016/03/03210554/table1a2.png)

**A cohort** is a group of subjects who share a defining characteristic. We can observe how a cohort behaves across time and compare it to other cohorts. Cohorts are used in medicine, psychology, econometrics, ecology and many other areas to perform a cross-section (compare difference across subjects) at intervals through time. 

**Types of cohorts: 
**
- Time Cohorts are customers who signed up for a product or service during a particular time frame. Analyzing these cohorts shows the customers’ behavior depending on the time they started using the company’s products or services. The time may be monthly or quarterly even daily.
- Behaovior cohorts are customers who purchased a product or subscribed to a service in the past. It groups customers by the type of product or service they signed up. Customers who signed up for basic level services might have different needs than those who signed up for advanced services. Understaning the needs of the various cohorts can help a company design custom-made services or products for particular segments.
- Size cohorts refer to the various sizes of customers who purchase company’s products or services. This categorization can be based on the amount of spending in some periodic time after acquisition or the product type that the customer spent most of their order amount in some period of time.


**Import Libraries and DataSet **


```python
# import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

#For Data  Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#For Machine Learning Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df = pd.read_excel("Online Retail.xlsx")


```

**Explore + Clean the data **

# Variables Description
**InvoiceNo Invoice number. Nominal, a six digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation

**StockCode Product (item) code. Nominal, a five digit integral number uniquely assigned to each distinct product

**Description Product (item) name. Nominal

**Quantity The quantities of each product (item) per transaction. Numeric

**InvoiceDate Invoice Date and time. Numeric, the day and time when each transaction was generated

**UnitPrice Unit price. Numeric, product price per unit in sterling

**CustomerID Customer number. Nominal, a six digit integral number uniquely assigned to each customer

**Country Country name. Nominal, the name of the country where each customer resides


```python
df.head(5)
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536365</td>
      <td>84029G</td>
      <td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
    <tr>
      <th>4</th>
      <td>536365</td>
      <td>84029E</td>
      <td>RED WOOLLY HOTTIE WHITE HEART.</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541909 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    541909 non-null  object        
     1   StockCode    541909 non-null  object        
     2   Description  540455 non-null  object        
     3   Quantity     541909 non-null  int64         
     4   InvoiceDate  541909 non-null  datetime64[ns]
     5   UnitPrice    541909 non-null  float64       
     6   CustomerID   406829 non-null  float64       
     7   Country      541909 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 33.1+ MB
    

**Nots that: There are Missing Data in Description and The Customer ID Columns , let's check that**

Check and Clean Missing Data 


```python
df.isnull().sum()
```




    InvoiceNo      0
    StockCode      0
    Description    0
    Quantity       0
    InvoiceDate    0
    UnitPrice      0
    CustomerID     0
    Country        0
    dtype: int64




```python
df= df.dropna(subset=['CustomerID'])
```


```python
df.isnull().sum().sum()
```




    0



Check & Clean Duplicates Data


```python
df.duplicated().sum()

```




    5225




```python
df = df.drop_duplicates()
```


```python
df.duplicated().sum()

```




    0






```python
df.describe()
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
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>401604.000000</td>
      <td>401604.000000</td>
      <td>401604.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.183273</td>
      <td>3.474064</td>
      <td>15281.160818</td>
    </tr>
    <tr>
      <th>std</th>
      <td>250.283037</td>
      <td>69.764035</td>
      <td>1714.006089</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-80995.000000</td>
      <td>0.000000</td>
      <td>12346.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.250000</td>
      <td>13939.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.000000</td>
      <td>1.950000</td>
      <td>15145.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.000000</td>
      <td>3.750000</td>
      <td>16784.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>38970.000000</td>
      <td>18287.000000</td>
    </tr>
  </tbody>
</table>
</div>



Note That : The min for unit price = 0 and the min for Quantity with negative value 


```python
df=df[(df['Quantity']>0) & (df['UnitPrice']>0)]
df.describe() 
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
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>392692.000000</td>
      <td>392692.000000</td>
      <td>392692.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.119702</td>
      <td>3.125914</td>
      <td>15287.843865</td>
    </tr>
    <tr>
      <th>std</th>
      <td>180.492832</td>
      <td>22.241836</td>
      <td>1713.539549</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.001000</td>
      <td>12346.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.250000</td>
      <td>13955.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>1.950000</td>
      <td>15150.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.000000</td>
      <td>3.750000</td>
      <td>16791.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>8142.750000</td>
      <td>18287.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (392692, 8)




```python
# Let's create a feature with total cost of the transactions
df['Total_cost'] = df.Quantity * df.UnitPrice
```

# EDA
Now let's do some Exploratory Data Analysis on the processed dataset


```python
# Check the oldest and latest date in the dataset.
print(f'Oldest date is - {df.InvoiceDate.min()}\n')
print(f'Latest date is - {df.InvoiceDate.max()}')
```

    Oldest date is - 2010-12-01 08:26:00
    
    Latest date is - 2011-12-09 12:50:00
    


```python
 # Count of transactions in different years
df.InvoiceDate.dt.year.value_counts(sort=False).plot(kind='bar', rot=45);
```


![png](output_30_0.png)



```python
# Let's visualize some top products from the whole range.
top_products = df['Description'].value_counts()[:20]
plt.figure(figsize=(10,6))
sns.set_context("paper", font_scale=1.5)
sns.barplot(y = top_products.index,
            x = top_products.values)
plt.title("Top selling products")
plt.show();
```


![png](output_31_0.png)



```python
 #Count of transactions in different months within 2011 year.
df[df.InvoiceDate.dt.year==2011].InvoiceDate.dt.month.value_counts(sort=False).plot(kind='bar');
```


![png](output_32_0.png)


An increasing pattern can be observed month by month wise with a sharp decline in the month of December. That is evident because only first 8-9 days of December 2011 month is available in the dataset i.e. around 70% of the month transactions are not considered. Due to this fact, sales figure looks legitimate.


```python
# Let's visualize the top grossing months
monthly_gross = df[df.InvoiceDate.dt.year==2011].groupby(df.InvoiceDate.dt.month).Total_cost.sum()
plt.figure(figsize=(10,5))
sns.lineplot(y=monthly_gross.values,x=monthly_gross.index, marker='o');
plt.xticks(range(1,13))
plt.show()
```


![png](output_34_0.png)



```python
# Let's visualize the Unit price distribution
plt.figure(figsize=(16,4))
sns.boxplot(y='UnitPrice', data=df, orient='h');
```


![png](output_35_0.png)


# Let's Make Cohort Analysis

**For cohort analysis, there are a few labels that we have to create:**
- Invoice period: A string representation of the year and month of a single transaction/invoice.
- Cohort group: A string representation of the the year and month of a customer’s first purchase. This label is common across all invoices for a particular customer.
- Cohort period / Cohort Index: A integer representation a customer’s stage in its “lifetime”. The number represents the number of months passed since the first purchase.




```python
def get_month(x) : return dt.datetime(x.year,x.month,1)
df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)
grouping = df.groupby('CustomerID')['InvoiceMonth']
df['CohortMonth'] = grouping.transform('min')
df.tail()
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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>InvoiceMonth</th>
      <th>CohortMonth</th>
      <th>CohortIndex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>541904</th>
      <td>581587</td>
      <td>22613</td>
      <td>PACK OF 20 SPACEBOY NAPKINS</td>
      <td>12</td>
      <td>2011-12-09 12:50:00</td>
      <td>0.85</td>
      <td>12680.0</td>
      <td>France</td>
      <td>2011-12-01</td>
      <td>2011-08-01</td>
      <td>5</td>
    </tr>
    <tr>
      <th>541905</th>
      <td>581587</td>
      <td>22899</td>
      <td>CHILDREN'S APRON DOLLY GIRL</td>
      <td>6</td>
      <td>2011-12-09 12:50:00</td>
      <td>2.10</td>
      <td>12680.0</td>
      <td>France</td>
      <td>2011-12-01</td>
      <td>2011-08-01</td>
      <td>5</td>
    </tr>
    <tr>
      <th>541906</th>
      <td>581587</td>
      <td>23254</td>
      <td>CHILDRENS CUTLERY DOLLY GIRL</td>
      <td>4</td>
      <td>2011-12-09 12:50:00</td>
      <td>4.15</td>
      <td>12680.0</td>
      <td>France</td>
      <td>2011-12-01</td>
      <td>2011-08-01</td>
      <td>5</td>
    </tr>
    <tr>
      <th>541907</th>
      <td>581587</td>
      <td>23255</td>
      <td>CHILDRENS CUTLERY CIRCUS PARADE</td>
      <td>4</td>
      <td>2011-12-09 12:50:00</td>
      <td>4.15</td>
      <td>12680.0</td>
      <td>France</td>
      <td>2011-12-01</td>
      <td>2011-08-01</td>
      <td>5</td>
    </tr>
    <tr>
      <th>541908</th>
      <td>581587</td>
      <td>22138</td>
      <td>BAKING SET 9 PIECE RETROSPOT</td>
      <td>3</td>
      <td>2011-12-09 12:50:00</td>
      <td>4.95</td>
      <td>12680.0</td>
      <td>France</td>
      <td>2011-12-01</td>
      <td>2011-08-01</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



# Calculate time offset in months
Calculating time offset for each transaction allows you to report the metrics for each cohort in a comparable fashion.

First, we will create some variables that capture the integer value of years and months for Invoice and Cohort Date using the get_date_int() function


```python
def get_month_int (dframe,column):
    year = dframe[column].dt.year
    month = dframe[column].dt.month
    day = dframe[column].dt.day
    return year, month , day 

# Get the integers for date parts from the `InvoiceMonth` column

invoice_year,invoice_month,_ = get_month_int(df,'InvoiceMonth')

# Get the integers for date parts from the `CohortMonth` column

cohort_year,cohort_month,_ = get_month_int(df,'CohortMonth')

# Calculate difference in months

year_diff = invoice_year - cohort_year 
month_diff = invoice_month - cohort_month 

# Extract the difference in months from all previous values

df['CohortIndex'] = year_diff * 12 + month_diff + 1 
```

# Calculate retention rate
Customer retention is a very useful metric to understand how many of all the customers are still active. It gives you the percentage of active customers compared to the total number of customers


```python
#Count monthly active customers from each cohort
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)
# Return number of unique elements in the object.
cohort_data = cohort_data.reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='CustomerID')
cohort_counts

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
      <th>CohortIndex</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
    </tr>
    <tr>
      <th>CohortMonth</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-12-01</th>
      <td>948.0</td>
      <td>362.0</td>
      <td>317.0</td>
      <td>367.0</td>
      <td>341.0</td>
      <td>376.0</td>
      <td>360.0</td>
      <td>336.0</td>
      <td>336.0</td>
      <td>374.0</td>
      <td>354.0</td>
      <td>474.0</td>
      <td>260.0</td>
    </tr>
    <tr>
      <th>2011-01-01</th>
      <td>421.0</td>
      <td>101.0</td>
      <td>119.0</td>
      <td>102.0</td>
      <td>138.0</td>
      <td>126.0</td>
      <td>110.0</td>
      <td>108.0</td>
      <td>131.0</td>
      <td>146.0</td>
      <td>155.0</td>
      <td>63.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-02-01</th>
      <td>380.0</td>
      <td>94.0</td>
      <td>73.0</td>
      <td>106.0</td>
      <td>102.0</td>
      <td>94.0</td>
      <td>97.0</td>
      <td>107.0</td>
      <td>98.0</td>
      <td>119.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-03-01</th>
      <td>440.0</td>
      <td>84.0</td>
      <td>112.0</td>
      <td>96.0</td>
      <td>102.0</td>
      <td>78.0</td>
      <td>116.0</td>
      <td>105.0</td>
      <td>127.0</td>
      <td>39.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-04-01</th>
      <td>299.0</td>
      <td>68.0</td>
      <td>66.0</td>
      <td>63.0</td>
      <td>62.0</td>
      <td>71.0</td>
      <td>69.0</td>
      <td>78.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-05-01</th>
      <td>279.0</td>
      <td>66.0</td>
      <td>48.0</td>
      <td>48.0</td>
      <td>60.0</td>
      <td>68.0</td>
      <td>74.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-06-01</th>
      <td>235.0</td>
      <td>49.0</td>
      <td>44.0</td>
      <td>64.0</td>
      <td>58.0</td>
      <td>79.0</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-07-01</th>
      <td>191.0</td>
      <td>40.0</td>
      <td>39.0</td>
      <td>44.0</td>
      <td>52.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-08-01</th>
      <td>167.0</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-09-01</th>
      <td>298.0</td>
      <td>89.0</td>
      <td>97.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-10-01</th>
      <td>352.0</td>
      <td>93.0</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-11-01</th>
      <td>321.0</td>
      <td>43.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-12-01</th>
      <td>41.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Retention Rate Table **


```python
# Retention table
cohort_size = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_size,axis=0) #axis=0 to ensure the divide along the row axis 
retention.round(3) * 100 #to show the number as percentage 
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
      <th>CohortIndex</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
    </tr>
    <tr>
      <th>CohortMonth</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-12-01</th>
      <td>100.0</td>
      <td>36.6</td>
      <td>32.3</td>
      <td>38.4</td>
      <td>36.3</td>
      <td>39.8</td>
      <td>36.3</td>
      <td>34.9</td>
      <td>35.4</td>
      <td>39.5</td>
      <td>37.4</td>
      <td>50.3</td>
      <td>26.6</td>
    </tr>
    <tr>
      <th>2011-01-01</th>
      <td>100.0</td>
      <td>22.1</td>
      <td>26.6</td>
      <td>23.0</td>
      <td>32.1</td>
      <td>28.8</td>
      <td>24.7</td>
      <td>24.2</td>
      <td>30.0</td>
      <td>32.6</td>
      <td>36.5</td>
      <td>11.8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-02-01</th>
      <td>100.0</td>
      <td>18.7</td>
      <td>18.7</td>
      <td>28.4</td>
      <td>27.1</td>
      <td>24.7</td>
      <td>25.3</td>
      <td>27.9</td>
      <td>24.7</td>
      <td>30.5</td>
      <td>6.8</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-03-01</th>
      <td>100.0</td>
      <td>15.0</td>
      <td>25.2</td>
      <td>19.9</td>
      <td>22.3</td>
      <td>16.8</td>
      <td>26.8</td>
      <td>23.0</td>
      <td>27.9</td>
      <td>8.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-04-01</th>
      <td>100.0</td>
      <td>21.3</td>
      <td>20.3</td>
      <td>21.0</td>
      <td>19.7</td>
      <td>22.7</td>
      <td>21.7</td>
      <td>26.0</td>
      <td>7.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-05-01</th>
      <td>100.0</td>
      <td>19.0</td>
      <td>17.3</td>
      <td>17.3</td>
      <td>20.8</td>
      <td>23.2</td>
      <td>26.4</td>
      <td>9.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-06-01</th>
      <td>100.0</td>
      <td>17.4</td>
      <td>15.7</td>
      <td>26.4</td>
      <td>23.1</td>
      <td>33.5</td>
      <td>9.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-07-01</th>
      <td>100.0</td>
      <td>18.1</td>
      <td>20.7</td>
      <td>22.3</td>
      <td>27.1</td>
      <td>11.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-08-01</th>
      <td>100.0</td>
      <td>20.7</td>
      <td>24.9</td>
      <td>24.3</td>
      <td>12.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-09-01</th>
      <td>100.0</td>
      <td>23.4</td>
      <td>30.1</td>
      <td>11.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-10-01</th>
      <td>100.0</td>
      <td>24.0</td>
      <td>11.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-11-01</th>
      <td>100.0</td>
      <td>11.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2011-12-01</th>
      <td>100.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Build the heatmap
plt.figure(figsize=(15, 8))
plt.title('Retention rates')
sns.heatmap(data=retention,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap="BuPu_r")
plt.show()
```


![png](output_45_0.png)


** Note That: Customer retention is a very useful metric to understand how many of the all customers are still active.Retention gives you the percentage of active customers compared to the total number of customers.**

**Average quantity for each cohort **


```python
#Average quantity for each cohort
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['Quantity'].mean()
cohort_data = cohort_data.reset_index()
average_quantity = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='Quantity')
average_quantity.round(1)
average_quantity.index = average_quantity.index.date

#Build the heatmap
plt.figure(figsize=(15, 8))
plt.title('Average quantity for each cohort')
sns.heatmap(data=average_quantity,annot = True,vmin = 0.0,vmax =20,cmap="BuGn_r")
plt.show()
```


![png](output_48_0.png)



```python

ax = Retention_rates.T.mean().plot(figsize=(11,6), marker='s')
plt.title("Retention rate (%) per CohortGroup", fontname='Ubuntu', fontsize=20, fontweight='bold')

plt.xticks(np.arange(1, 16.1, 1), fontsize=10)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
ax.set_xlabel("CohortPeriod", fontsize=10)
ax.set_ylabel("Retention(%)", fontsize=10)
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-52-f89e89d88b09> in <module>
    ----> 1 ax = Retention_rates.T.mean().plot(figsize=(11,6), marker='s')
          2 plt.title("Retention rate (%) per CohortGroup", fontname='Ubuntu', fontsize=20, fontweight='bold')
          3 
          4 plt.xticks(np.arange(1, 16.1, 1), fontsize=10)
          5 plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)
    

    NameError: name 'Retention_rates' is not defined


# Recency, Frequency and Monetary Value calculation

![](http://www.omniconvert.com/blog/wp-content/uploads/2016/03/feature-img-marketizator-rfm.png)

**What is RFM?
**
- **RFM** is an acronym of recency, frequency and monetary. Recency is about when was the last order of a customer. It means the number of days since a customer made the last purchase. If it’s a case for a website or an app, this could be interpreted as the last visit day or the last login time.

- **Frequency** is about the number of purchase in a given period. It could be 3 months, 6 months or 1 year. So we can understand this value as for how often or how many a customer used the product of a company. The bigger the value is, the more engaged the customers are. Could we say them as our VIP? Not necessary. Cause we also have to think about how much they actually paid for each purchase, which means monetary value.

- **Monetary** is the total amount of money a customer spent in that given period. Therefore big spenders will be differentiated with other customers such as MVP or VIP.

![](http://d35fo82fjcw0y8.cloudfront.net/2018/03/01013508/Incontent_image.png)

**The RFM values can be grouped in several ways: **

**1.Percentiles e.g. quantiles **

**2.Pareto 80/20 cut**

**3.Custom - based on business knowledge**

**We are going to implement percentile-based grouping.**

**Process of calculating percentiles:**
1. Sort customers based on that metric
2. Break customers into a pre-defined number of groups of equal size
3. Assign a label to each group


```python
#New Total Sum Column  
df['TotalSum'] = df['UnitPrice']* df['Quantity']

#Data preparation steps
print('Min Invoice Date:',df.InvoiceDate.dt.date.min(),'max Invoice Date:',
       df.InvoiceDate.dt.date.max())

df.head(3)
```

    Min Invoice Date: 2010-12-01 max Invoice Date: 2011-12-09
    




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
      <th>InvoiceNo</th>
      <th>StockCode</th>
      <th>Description</th>
      <th>Quantity</th>
      <th>InvoiceDate</th>
      <th>UnitPrice</th>
      <th>CustomerID</th>
      <th>Country</th>
      <th>InvoiceMonth</th>
      <th>CohortMonth</th>
      <th>CohortIndex</th>
      <th>Total_cost</th>
      <th>TotalSum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>536365</td>
      <td>85123A</td>
      <td>WHITE HANGING HEART T-LIGHT HOLDER</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.55</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>2010-12-01</td>
      <td>2010-12-01</td>
      <td>1</td>
      <td>15.30</td>
      <td>15.30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>536365</td>
      <td>71053</td>
      <td>WHITE METAL LANTERN</td>
      <td>6</td>
      <td>2010-12-01 08:26:00</td>
      <td>3.39</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>2010-12-01</td>
      <td>2010-12-01</td>
      <td>1</td>
      <td>20.34</td>
      <td>20.34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>536365</td>
      <td>84406B</td>
      <td>CREAM CUPID HEARTS COAT HANGER</td>
      <td>8</td>
      <td>2010-12-01 08:26:00</td>
      <td>2.75</td>
      <td>17850.0</td>
      <td>United Kingdom</td>
      <td>2010-12-01</td>
      <td>2010-12-01</td>
      <td>1</td>
      <td>22.00</td>
      <td>22.00</td>
    </tr>
  </tbody>
</table>
</div>



In the real world, we would be working with the most recent snapshot of the data of today or yesterday


```python
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
snapshot_date
#The last day of purchase in total is 09 DEC, 2011. To calculate the day periods, 
#let's set one day after the last one,or 
#10 DEC as a snapshot_date. We will cound the diff days with snapshot_date.

```




    Timestamp('2011-12-10 12:50:00')




```python
# Calculate RFM metrics
rfm = df.groupby(['CustomerID']).agg({'InvoiceDate': lambda x : (snapshot_date - x.max()).days,
                                      'InvoiceNo':'count','TotalSum': 'sum'})
#Function Lambdea: it gives the number of days between hypothetical today and the last transaction

#Rename columns
rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalSum':'MonetaryValue'}
           ,inplace= True)

#Final RFM values
rfm.head()

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
      <th>Recency</th>
      <th>Frequency</th>
      <th>MonetaryValue</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>326</td>
      <td>1</td>
      <td>77183.60</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>2</td>
      <td>182</td>
      <td>4310.00</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>75</td>
      <td>31</td>
      <td>1797.24</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>19</td>
      <td>73</td>
      <td>1757.55</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>310</td>
      <td>17</td>
      <td>334.40</td>
    </tr>
  </tbody>
</table>
</div>



**Note That : **

**#We will rate "Recency" customer who have been active more recently better than the less recent customer,because each company wants its customers to be recent ** 

**#We will rate "Frequency" and "Monetary Value" higher label because we want Customer to spend more money and visit more often(that is  different order than recency). **


```python
#Building RFM segments
r_labels =range(4,0,-1)
f_labels=range(1,5)
m_labels=range(1,5)
r_quartiles = pd.qcut(rfm['Recency'], q=4, labels = r_labels)
f_quartiles = pd.qcut(rfm['Frequency'],q=4, labels = f_labels)
m_quartiles = pd.qcut(rfm['MonetaryValue'],q=4,labels = m_labels)
rfm = rfm.assign(R=r_quartiles,F=f_quartiles,M=m_quartiles)

# Build RFM Segment and RFM Score
def add_rfm(x) : return str(x['R']) + str(x['F']) + str(x['M'])
rfm['RFM_Segment'] = rfm.apply(add_rfm,axis=1 )
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)

rfm.head()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>MonetaryValue</th>
      <th>R</th>
      <th>F</th>
      <th>M</th>
      <th>RFM_Segment</th>
      <th>RFM_Score</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12346.0</th>
      <td>326</td>
      <td>1</td>
      <td>77183.60</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>114</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>12347.0</th>
      <td>2</td>
      <td>182</td>
      <td>4310.00</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>444</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>12348.0</th>
      <td>75</td>
      <td>31</td>
      <td>1797.24</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>224</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>12349.0</th>
      <td>19</td>
      <td>73</td>
      <td>1757.55</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>334</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>12350.0</th>
      <td>310</td>
      <td>17</td>
      <td>334.40</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>112</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



**The Result is a Table which has a row for each customer with their RFM **

# Analyzing RFM Segments

**Largest RFM segments **
**It is always the best practice to investigate the size of the segments before you use them for targeting or other business Application.**


```python
rfm.groupby(['RFM_Segment']).size().sort_values(ascending=False)[:5]
```




    RFM_Segment
    444    450
    111    381
    344    217
    122    206
    211    179
    dtype: int64



**Filtering on RFM segments **


```python
#Select bottom RFM segment "111" and view top 5 rows
rfm[rfm['RFM_Segment']=='111'].head()
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
      <th>Recency</th>
      <th>Frequency</th>
      <th>MonetaryValue</th>
      <th>R</th>
      <th>F</th>
      <th>M</th>
      <th>RFM_Segment</th>
      <th>RFM_Score</th>
    </tr>
    <tr>
      <th>CustomerID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12353.0</th>
      <td>204</td>
      <td>4</td>
      <td>89.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12361.0</th>
      <td>287</td>
      <td>10</td>
      <td>189.90</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12401.0</th>
      <td>303</td>
      <td>5</td>
      <td>84.30</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12402.0</th>
      <td>323</td>
      <td>11</td>
      <td>225.60</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>12441.0</th>
      <td>367</td>
      <td>11</td>
      <td>173.55</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>111</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



**Summary metrics per RFM Score **


```python
rfm.groupby('RFM_Score').agg({'Recency': 'mean','Frequency': 'mean',
                             'MonetaryValue': ['mean', 'count'] }).round(1)


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Recency</th>
      <th>Frequency</th>
      <th colspan="2" halign="left">MonetaryValue</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>RFM_Score</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3.0</th>
      <td>260.7</td>
      <td>8.2</td>
      <td>157.4</td>
      <td>381</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>177.2</td>
      <td>13.6</td>
      <td>240.0</td>
      <td>388</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>152.9</td>
      <td>21.2</td>
      <td>366.6</td>
      <td>518</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>95.9</td>
      <td>27.9</td>
      <td>820.8</td>
      <td>457</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>79.6</td>
      <td>38.0</td>
      <td>758.1</td>
      <td>463</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>64.1</td>
      <td>56.0</td>
      <td>987.3</td>
      <td>454</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>45.9</td>
      <td>78.7</td>
      <td>1795.1</td>
      <td>414</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>32.4</td>
      <td>110.5</td>
      <td>2056.4</td>
      <td>426</td>
    </tr>
    <tr>
      <th>11.0</th>
      <td>21.3</td>
      <td>186.9</td>
      <td>4062.0</td>
      <td>387</td>
    </tr>
    <tr>
      <th>12.0</th>
      <td>7.2</td>
      <td>367.8</td>
      <td>9285.9</td>
      <td>450</td>
    </tr>
  </tbody>
</table>
</div>



**Use RFM score to group customers into Gold, Silver and Bronze segments:**


```python
def segments(df):
    if df['RFM_Score'] > 9 :
        return 'Gold'
    elif (df['RFM_Score'] > 5) and (df['RFM_Score'] <= 9 ):
        return 'Sliver'
    else:  
        return 'Bronze'

rfm['General_Segment'] = rfm.apply(segments,axis=1)

rfm.groupby('General_Segment').agg({'Recency':'mean','Frequency':'mean',
                                    'MonetaryValue':['mean','count']}).round(1)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Recency</th>
      <th>Frequency</th>
      <th colspan="2" halign="left">MonetaryValue</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>General_Segment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bronze</th>
      <td>192.2</td>
      <td>15.1</td>
      <td>266.5</td>
      <td>1287</td>
    </tr>
    <tr>
      <th>Gold</th>
      <td>20.1</td>
      <td>225.6</td>
      <td>5246.8</td>
      <td>1263</td>
    </tr>
    <tr>
      <th>Sliver</th>
      <td>72.0</td>
      <td>49.4</td>
      <td>1072.4</td>
      <td>1788</td>
    </tr>
  </tbody>
</table>
</div>



# Data Pre-Processing for Kmeans Clustering

K-Means clustering is one type of unsupervised learning algorithms, which makes groups based on the distance between the points. How? There are two concepts of distance in K-Means clustering. Within Cluster Sums of Squares (WSS) and Between Cluster Sums of Squares (BSS).

**We must check these Key k-means assumptions before we implement our Kmeans Clustering Mode**
- Symmetric distribution of variables (not skewed)
- Variables with same average values
- Variables with same variance


```python
rfm_rfm = rfm[['Recency','Frequency','MonetaryValue']]
print(rfm_rfm.describe())


```

               Recency    Frequency  MonetaryValue
    count  4338.000000  4338.000000    4338.000000
    mean     92.536422    90.523744    2048.688081
    std     100.014169   225.506968    8985.230220
    min       1.000000     1.000000       3.750000
    25%      18.000000    17.000000     306.482500
    50%      51.000000    41.000000     668.570000
    75%     142.000000    98.000000    1660.597500
    max     374.000000  7676.000000  280206.020000
    

**From this table, we find this Problem: Mean and Variance are not Equal**

**Soluation: Scaling variables by using a scaler from scikit-learn library**


```python
# plot the distribution of RFM values
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()

```

    /opt/conda/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](output_76_1.png)


**Also, there is another Problem: UnSymmetric distribution of variables (data skewed)**

**Soluation:Logarithmic transformation (positive values only) will manage skewness**

**We use these Sequence of structuring pre-processing steps: **
**1. Unskew the data - log transformation **

**2. Standardize to the same average values **

**3. Scale to the same standard deviation **

**4. Store as a separate array to be used for clustering**
_______________________________

**Why the sequence matters?**
- Log transformation only works with positive data
- Normalization forces data to have negative values and log will not work


```python
#Unskew the data with log transformation
rfm_log = rfm[['Recency', 'Frequency', 'MonetaryValue']].apply(np.log, axis = 1).round(3)
#or rfm_log = np.log(rfm_rfm)


# plot the distribution of RFM values
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm_log.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_log.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_log.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()

```

    /opt/conda/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](output_79_1.png)


# Implementation of K-Means Clustering

**Key steps**
1. Data pre-processing
2. Choosing a number of clusters
3. Running k-means clustering on pre-processed data
4. Analyzing average RFM values of each cluster

**** 1. Data Pre-Processing****



```python
#Normalize the variables with StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(rfm_log)
#Store it separately for clustering
rfm_normalized= scaler.transform(rfm_log)
```

**2. Choosing a Number of Clusters**

**Methods to define the number of clusters**
- Visual methods - elbow criterion
- Mathematical methods - silhouette coefficient
- Experimentation and interpretation

**Elbow criterion method ** 
- Plot the number of clusters against within-cluster sum-of-squared-errors (SSE) - sum of squared distances from every data point to their cluster center
- Identify an "elbow" in the plot
- Elbow - a point representing an "optimal" number of clusters


```python
from sklearn.cluster import KMeans

#First : Get the Best KMeans 
ks = range(1,8)
inertias=[]
for k in ks :
    # Create a KMeans clusters
    kc = KMeans(n_clusters=k,random_state=1)
    kc.fit(rfm_normalized)
    inertias.append(kc.inertia_)

# Plot ks vs inertias
f, ax = plt.subplots(figsize=(15, 8))
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.style.use('ggplot')
plt.title('What is the Best Number for KMeans ?')
plt.show()


```


![png](output_86_0.png)


**Note Theat: We Choose No.KMeans = 3 **


```python
# clustering
kc = KMeans(n_clusters= 3, random_state=1)
kc.fit(rfm_normalized)

#Create a cluster label column in the original DataFrame
cluster_labels = kc.labels_

#Calculate average RFM values and size for each cluster:
rfm_rfm_k3 = rfm_rfm.assign(K_Cluster = cluster_labels)

#Calculate average RFM values and sizes for each cluster:
rfm_rfm_k3.groupby('K_Cluster').agg({'Recency': 'mean','Frequency': 'mean',
                                         'MonetaryValue': ['mean', 'count'],}).round(0)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Recency</th>
      <th>Frequency</th>
      <th colspan="2" halign="left">MonetaryValue</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>mean</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>K_Cluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>171.0</td>
      <td>15.0</td>
      <td>293.0</td>
      <td>1527</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.0</td>
      <td>260.0</td>
      <td>6574.0</td>
      <td>953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69.0</td>
      <td>65.0</td>
      <td>1170.0</td>
      <td>1858</td>
    </tr>
  </tbody>
</table>
</div>



**Snake plots to understand and compare segments**
- Market research technique to compare different segments
- Visual representation of each segment's attributes
- Need to first normalize data (center & scale)
- Plot each cluster's average normalized values of each attribute




```python
rfm_normalized = pd.DataFrame(rfm_normalized,index=rfm_rfm.index,columns=rfm_rfm.columns)
rfm_normalized['K_Cluster'] = kc.labels_
rfm_normalized['General_Segment'] = rfm['General_Segment']
rfm_normalized.reset_index(inplace = True)

#Melt the data into a long format so RFM values and metric names are stored in 1 column each
rfm_melt = pd.melt(rfm_normalized,id_vars=['CustomerID','General_Segment','K_Cluster'],value_vars=['Recency', 'Frequency', 'MonetaryValue'],
var_name='Metric',value_name='Value')
rfm_melt.head()

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
      <th>CustomerID</th>
      <th>General_Segment</th>
      <th>K_Cluster</th>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12346.0</td>
      <td>Sliver</td>
      <td>2</td>
      <td>Recency</td>
      <td>1.409982</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12347.0</td>
      <td>Gold</td>
      <td>1</td>
      <td>Recency</td>
      <td>-2.146578</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12348.0</td>
      <td>Sliver</td>
      <td>2</td>
      <td>Recency</td>
      <td>0.383648</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12349.0</td>
      <td>Gold</td>
      <td>2</td>
      <td>Recency</td>
      <td>-0.574961</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12350.0</td>
      <td>Bronze</td>
      <td>0</td>
      <td>Recency</td>
      <td>1.375072</td>
    </tr>
  </tbody>
</table>
</div>



**Snake Plot and Heatmap**


```python
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(x = 'Metric', y = 'Value', hue = 'General_Segment', data = rfm_melt,ax=ax1)

# a snake plot with K-Means
sns.lineplot(x = 'Metric', y = 'Value', hue = 'K_Cluster', data = rfm_melt,ax=ax2)

plt.suptitle("Snake Plot of RFM",fontsize=24) #make title fontsize subtitle 
plt.show()

```

    /opt/conda/lib/python3.6/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](output_92_1.png)


**Relative importance of segment attributes **
- Useful technique to identify relative importance of each segment's attribute
- Calculate average values of each cluster
- Calculate average values of population
- Calculate importance score by dividing them and subtracting 1 (ensures 0 is returned when cluster average equals population average)

**Let’s try again with a heat map. Heat maps are a graphical representation of data where larger values were colored in darker scales and smaller values in lighter scales. We can compare the variance between the groups quite intuitively by colors. **




```python
# The further a ratio is from 0, the more important that attribute is for a segment relative to the total population
cluster_avg = rfm_rfm_k3.groupby(['K_Cluster']).mean()
population_avg = rfm_rfm.mean()
relative_imp = cluster_avg / population_avg - 1
relative_imp.round(2)


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
      <th>Recency</th>
      <th>Frequency</th>
      <th>MonetaryValue</th>
    </tr>
    <tr>
      <th>K_Cluster</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.85</td>
      <td>-0.84</td>
      <td>-0.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.86</td>
      <td>1.88</td>
      <td>2.21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.25</td>
      <td>-0.28</td>
      <td>-0.43</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the mean value in total 
total_avg = rfm.iloc[:, 0:3].mean()
# calculate the proportional gap with total mean
cluster_avg = rfm.groupby('General_Segment').mean().iloc[:, 0:3]
prop_rfm = cluster_avg/total_avg - 1
prop_rfm.round(2)

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
      <th>Recency</th>
      <th>Frequency</th>
      <th>MonetaryValue</th>
    </tr>
    <tr>
      <th>General_Segment</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bronze</th>
      <td>1.08</td>
      <td>-0.83</td>
      <td>-0.87</td>
    </tr>
    <tr>
      <th>Gold</th>
      <td>-0.78</td>
      <td>1.49</td>
      <td>1.56</td>
    </tr>
    <tr>
      <th>Sliver</th>
      <td>-0.22</td>
      <td>-0.45</td>
      <td>-0.48</td>
    </tr>
  </tbody>
</table>
</div>




```python
# heatmap with RFM
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='Blues',ax=ax1)
ax1.set(title = "Heatmap of K-Means")

# a snake plot with K-Means
sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True,ax=ax2)
ax2.set(title = "Heatmap of RFM quantile")

plt.suptitle("Heat Map of RFM",fontsize=20) #make title fontsize subtitle 

plt.show()


```


![png](output_96_0.png)


**You can Updated RFM data by adding Tenure variable : **
** -Tenure: time since the first transaction ، Defines how long the customer has been with the company**

**Conclusion:  We talked about how to get RFM values from customer purchase data, and we made two kinds of segmentation with RFM quantiles and K-Means clustering methods. **



