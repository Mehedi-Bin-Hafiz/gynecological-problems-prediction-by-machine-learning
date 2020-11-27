import matplotlib.pyplot as plt
import pandas as pd

########### vvi code for paper#########
plt.rcParams.update({'font.size': 8})
plt.rcParams["font.family"] = "Times New Roman"
########### vvi code for paper#########

mainDataset = pd.read_excel(r'../database/MainDataset.xlsx')


# real = mainDataset['Pneumonia (+ / -)'].values.tolist()
# predicted = mainDataset['Prediction'].values.tolist()
Miscarriage = mainDataset['Disease'].values.tolist().count(1)
Anemia = mainDataset['Disease'].values.tolist().count(2)


labels = 'Miscarriage', 'Anemia'

sizes = [Miscarriage, Anemia]
explode = (0,0,)
#autopact show percentage inside graph
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',)
plt.axis('equal')
plt.show()