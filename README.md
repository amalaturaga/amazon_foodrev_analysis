# **Customer Reviews Data Cleaning & Sentiment Analysis**

## **Project Overview**
This project analyzes customer review data to identify sentiment trends over time, providing insights into how product perception evolves. The goal is to **clean unstructured data**, **store it efficiently**, **analyze sentiment**, and **visualize trends** to understand how sentiment correlates with product scores and time.

## **Research Question**
How do customer sentiments change over time, and how does sentiment distribution impact product ratings?

## **Dataset Chosen**
We used a publicly available customer reviews dataset from Kaggle:
- [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

## **Data Cleaning & Preprocessing**
1. **Filtered relevant columns** (`Text`, `Score`, `Time`, `ProductId`).
2. **Removed missing values** to ensure data integrity.
3. **Converted timestamps** to a readable format (`Year-Month`).
4. **Eliminated duplicate reviews** to avoid skewed analysis.
5. **Stored cleaned data in an SQLite database (`reviews.db`)** for structured storage and easy querying.

## **Analysis & Visualizations**

### **1. Sentiment Distribution in Reviews**
#### 📌 **Goal:** Categorize reviews as **Positive, Negative, or Neutral** using `TextBlob`.

#### **Results:**
- **Positive Reviews:** 500,946 (88.31%)
- **Negative Reviews:** 57,946 (10.22%)
- **Neutral Reviews:** 8,371 (1.48%)

#### **Visualization:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sentiment_counts = df['Sentiment'].value_counts()
plt.figure(figsize=(8,5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="coolwarm")
plt.title("Sentiment Analysis of Customer Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
```
#### **Insights:**
- Most reviews are **positive**, suggesting general customer satisfaction.
- Negative reviews are significantly fewer but crucial for improvement insights.

---

### **2. Score vs. Time Trend**
#### 📌 **Goal:** Analyze how **average review scores change over time**.

#### **Results:**
- **Peak Score:** 5.0 (October 1999)
- **Lowest Score:** 1.0 (August 2001)

#### **Visualization:**
```python
plt.figure(figsize=(12,6))
plt.plot(score_trend['YearMonth'].astype(str), score_trend['Score'], marker='o', linestyle='-', color='blue')
plt.title("Average Review Score Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.grid()
plt.show()
```
#### **Insights:**
- Identifies **periods of declining or improving customer satisfaction**.
- Correlate dips with potential product changes, seasonal trends, or external factors.

---

### **3. Sentiment Trends Over Time**
#### 📌 **Goal:** Show how the **number of positive, negative, and neutral reviews** fluctuates over time.

#### **Results:**
- **Month with Highest Negative Sentiment:** September 2012
- **Month with Highest Positive Sentiment:** September 2012

#### **Visualization:**
```python
sentiment_trend = df.groupby(['YearMonth', 'Sentiment']).size().unstack()
sentiment_trend.plot(kind='line', figsize=(12,6), marker='o')
plt.title("Sentiment Trends Over Time")
plt.xlabel("Year-Month")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.grid()
plt.show()
```
#### **Insights:**
- Spikes in **negative reviews** may indicate product defects or controversies.
- Observing positive trends can highlight successful product improvements or marketing strategies.

---

### **4. Correlation Between Score & Negative Sentiment**
#### 📌 **Goal:** Check if **negative sentiment percentage** affects the **average review score**.

#### **Results:**
- **Correlation Coefficient:** -0.288 (Moderate negative correlation)

#### **Visualization:**
```python
import seaborn as sns

plt.figure(figsize=(10,5))
sns.scatterplot(x=merged['Negative_Percentage'], y=merged['Score'])
plt.title("Correlation Between Negative Sentiment and Score")
plt.xlabel("Percentage of Negative Reviews")
plt.ylabel("Average Score")
plt.show()
```
#### **Insights:**
- A high percentage of negative reviews **correlates with a drop in average score**.
- Brands can use this insight to proactively **address customer concerns**.

---

## **Key Takeaways**
1. **Overall Sentiment:** Majority of customer reviews are positive (88.31%), with 10.22% being negative.
2. **Common Praise & Complaints:** Positive reviews highlight **product satisfaction**, while negative reviews reveal potential **improvement areas**.
3. **Time-Based Trends:** Review scores **fluctuate over time**, with the highest score recorded in **October 1999** and the lowest in **August 2001**.
4. **Impact of Negative Sentiment:** Higher percentages of negative reviews **directly lower average scores** (-0.288 correlation), impacting brand reputation.
5. **Peak Negative & Positive Sentiment Month:** **September 2012** had the highest number of both positive and negative reviews, possibly due to external market influences.

## **Deliverables**
- `amazon_reviews.ipynb` (Jupyter Notebook with full analysis)
- `reviews.db` (SQLite database storing structured data)
- `README.md` (Project summary)

## **Next Steps**
- **Deeper NLP Analysis:** Extract specific **customer complaints** using topic modeling.
- **Product-Specific Trends:** Analyze sentiment changes **per product**.
- **Predictive Analysis:** Use ML models to predict **future sentiment trends**.

---
