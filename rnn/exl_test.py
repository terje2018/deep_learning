import pandas as pd
import matplotlib.pyplot as plt

exl = pd.read_excel("doc/COE.xls")
print(exl['DATE'])
print(exl.head())
print(exl.keys().values[1])
print(exl['DATE'].head(20))

print(exl[193:204])
exl.at[194,'DATE'] = pd.Timestamp('2004-02-15')
exl.at[198,'DATE'] = pd.Timestamp('2004-04-15')
exl.at[202,'DATE'] = pd.Timestamp('2004-06-15')
print(exl[193:204])
exl.to_csv("doc/COE.csv")

xary = exl['COE$'].to_xarray()
plt.plot(xary)
plt.show()