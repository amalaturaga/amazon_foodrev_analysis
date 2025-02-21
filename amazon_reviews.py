import pandas as pd
import sqlite3
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

### Load and Inspect the Dataset

# Load the dataset
df = pd.read_csv("Reviews.csv")
# Show first few rows
df.head()

# Check the dataset info
df.info()
df.describe()
df.isnull().sum() # Check for missing values


### Clean and Structure the Dataset

# Keep only the columns that are needed
df = df[['Text', 'Score', 'Time', 'ProductId']]

# Drop missing values
df.dropna(inplace=True)

# Convert Unix timestamp to a readable date
df['Time'] = pd.to_datetime(df['Time'], unit='s')

# Remove duplicate reviews
df = df.drop_duplicates()

# Display cleaned dataset
df.head()


### Store Data in SQL Database

# Connect to SQLite
conn = sqlite3.connect("reviews.db")
cursor = conn.cursor()

# Save DataFrame to SQL table
df.to_sql("reviews", conn, if_exists="replace", index=False)

# Check stored data
pd.read_sql("SELECT * FROM reviews LIMIT 5", conn)

### Perform Sentiment Analysis

# Function to get sentiment
def get_sentiment(text):
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['Sentiment'] = df['Text'].apply(get_sentiment)

# Check results
df[['Text', 'Score', 'Sentiment']].head()

# Bar Chart of Sentiment Distribution
# Count sentiment categories
sentiment_counts = df['Sentiment'].value_counts()

# Plot results
plt.figure(figsize=(8,5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
plt.title("Sentiment Analysis of Customer Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

#### Further Analysis: Score vs. Time and Its Effect on Sentiment ####

## Agreggate Data by Month/Year

# Convert Time column to datetime format (if not already done)
df['Time'] = pd.to_datetime(df['Time'])

# Extract Year-Month for analysis
df['YearMonth'] = df['Time'].dt.to_period('M')

# Group by time and compute average score
score_trend = df.groupby('YearMonth')['Score'].mean().reset_index()

# Display the first few rows
print(score_trend.head())

### Visualizing Score Trends Over Time
# Plot average review score over time
plt.figure(figsize=(12,6))
plt.plot(score_trend['YearMonth'].astype(str), score_trend['Score'], marker='o', linestyle='-', color='blue')

# Customize plot
plt.title("Average Review Score Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.grid()
plt.show()

### Comparing Sentiment Distribution Over Time
# Group data by time and sentiment count
sentiment_trend = df.groupby(['YearMonth', 'Sentiment']).size().unstack()

# Plot sentiment distribution over time
sentiment_trend.plot(kind='line', figsize=(12,6), marker='o')

plt.title("Sentiment Trends Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.grid()
plt.show()

#### Find Correlation Between Score & Negative Sentiment
# Calculate the percentage of negative reviews per time period
df['Negative'] = df['Sentiment'].apply(lambda x: 1 if x == "Negative" else 0)
neg_trend = df.groupby('YearMonth')['Negative'].mean().reset_index()

# Merge with score trend
merged = pd.merge(score_trend, neg_trend, on="YearMonth")

# Plot correlation between negative sentiment and score
plt.figure(figsize=(10,5))
sns.scatterplot(x=merged['Negative'], y=merged['Score'])
plt.title("Correlation Between Negative Sentiment and Score")
plt.xlabel("Percentage of Negative Reviews")
plt.ylabel("Average Score")
plt.show()