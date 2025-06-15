import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse

# Load datasets
legit = pd.read_csv("legit.csv", header=None, names=["url"])
legit["label"] = "legitimate"
legit["url"] = "http://" + legit["url"]

phis = pd.read_csv("phis.csv")
phis["label"] = "phishing"

# Combine
df = pd.concat([legit, phis], ignore_index=True)
df.reset_index(drop=True, inplace=True)

# Parse domains
df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)
df['url_length'] = df['url'].apply(len)

# 1. Basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass balance:\n", df['label'].value_counts())

# 2. Domain frequency
print("\nTop 10 domains overall:\n", df['domain'].value_counts().head(10))
top_domains = df['domain'].value_counts().head(10).index

# 3. Visualizations

## URL length distribution
plt.figure(figsize=(8,4))
sns.histplot(df['url_length'], bins=50, kde=True)
plt.title('URL Length Distribution')
plt.xlabel('URL Length')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("url_length_dist.png")
plt.close()

## URL length by label
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='url_length', hue='label', bins=50, kde=True, element='step')
plt.title('URL Length by Label')
plt.xlabel('URL Length')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("url_length_by_label.png")
plt.close()

## Top 10 domains by label
plt.figure(figsize=(10,5))
sns.countplot(data=df[df['domain'].isin(top_domains)], y='domain', hue='label')
plt.title('Top 10 Domains by Label')
plt.ylabel('Domain')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig("top_domains_by_label.png")
plt.close()

# 4. Show some URL samples by label
print("\nSample legitimate URLs:")
print(df[df['label'] == 'legitimate']['url'].sample(5, random_state=1).to_list())
print("\nSample phishing URLs:")
print(df[df['label'] == 'phishing']['url'].sample(5, random_state=1).to_list())

# 5. Correlation between URL length and label
mean_legit = df[df['label'] == 'legitimate']['url_length'].mean()
mean_phish = df[df['label'] == 'phishing']['url_length'].mean()
print(f"\nMean URL length (legitimate): {mean_legit:.2f}")
print(f"Mean URL length (phishing): {mean_phish:.2f}")

# 6. Save EDA summary
df.describe(include='all').to_csv("eda_summary.csv")
print("\nSummary statistics saved to eda_summary.csv")
