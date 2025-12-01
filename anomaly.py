import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

df = pd.read_csv("logs.csv")

model = IsolationForest(contamination=0.15, random_state=42)
df["anomaly"] = model.fit_predict(df[["cpu","error_count"]])

print(df)

# visualize
plt.plot(df["cpu"], label="CPU")
plt.scatter(df.index, df["cpu"], c=df["anomaly"], cmap="coolwarm", label="Anomaly")
plt.title("CPU Anomaly Detection")
plt.legend()
plt.show()

