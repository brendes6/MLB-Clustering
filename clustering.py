import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Data/MasterNonPitching.csv")

data["ISO"] = data["SLG"] - data["BA"]

# Filter data to remove outliers, and only include players with significant experience
data = data[(data["OBP"] <= 0.5) & (data["SLG"] <= 0.8)]
data = data[data["AB"] >= 1000]
# data = data[data["Allstar"] == 1]

# Calculate ratio of power to contact
data["power_contact_ratio"] = data["ISO"] / data["OBP"]

# Scale data
scaler = StandardScaler()
data["scaled_ratio"] = scaler.fit_transform(data[["power_contact_ratio"]])

# Build KMeans model, predict cluster for every dataframe value
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
data["Cluster"] = kmeans.fit_predict(data[["scaled_ratio"]])

# Extract player decade based on debut date
data["debut"] = pd.to_datetime(data["debut"], errors="coerce")
data["debut_year"] = data["debut"].dt.year
data["decade"] = (data["debut_year"] // 10) * 10

# Check counts for decades and clusters
print(data["decade"].value_counts())
print(data["Cluster"].value_counts())

# Assign proper label based on cluster
cluster_labels = {
    0: "Power Hitter",
    1: "Contact Hitter"
}

data["HitterType"] = data["Cluster"].map(cluster_labels)

# Find hitter types by decade and proportion of hitter types
grouped = data.groupby(["decade", "HitterType"]).size().unstack()
proportions = grouped.div(grouped.sum(axis=1), axis=0)

# Plot the proportions
proportions.plot(kind="bar", stacked=True, colormap="Set3")
plt.title("How Hitter Archetypes Have Changed By Decade")
plt.ylabel("Proportion of Hitters")
plt.xlabel("Decade")
plt.legend(title="Hitter Type")
plt.show()
