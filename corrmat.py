import pandas as ps
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

data=ps.read_csv("C:\Users\mathe\Desktop\New folder\Dataset-vf.csv")

x=data.iloc[:,1:-2]
y=data.iloc[:,-1]
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(30,30))
g=sbn.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
print data[top_corr_features].corr()
plt.show()