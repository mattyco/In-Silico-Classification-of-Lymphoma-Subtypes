import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# print accuracy
# print cm

# print y_test
# print svm_predictions

# # plt.scatter(X_test,y_test)
# plt.scatter(y_test,svm_predictions)
# plt.show()


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
rf_predictions = classifier.predict(X_test)

accuracy=r2_score(y_test,rf_predictions)
cm = confusion_matrix(y_test, rf_predictions)

print accuracy
print cm
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)

print y_test
print rf_predictions

# plt.scatter(X_test,y_test)
plt.scatter(y_test,rf_predictions)
plt.show()
