import pandas as pd
import matplotlib.pyplot as plt

checkValidationDataSet = pd.read_excel(r'../database/predictedDataSet.xlsx')

real = checkValidationDataSet['Disease'].values.tolist()
predicted = checkValidationDataSet['Prediction'].values.tolist()

XandYLen = []
for i in range(1, len(real) + 1):
    XandYLen.append(i)
axes = plt.axes()
plt.plot(XandYLen, real, color='#55E6C1', linewidth=6)
plt.plot(XandYLen, predicted, color='#182C61', linewidth=3)
axes.set_yticks([ 0, .5, 1, 1.5, 2, 2.5,3,3.5])
plt.grid()
plt.legend(['Real value', 'Predicted value'])
plt.savefig(" real vs prediction.png")
plt.show()

print('###Bar Graph###')