import pandas as pd
import matplotlib.pyplot as plt
sn = pd.read_csv("doc/SN_m_tot_V2.0.csv",header=None,sep=';')
print(sn)
yt = sn.iloc[0:3210,3]
sn1 = yt.shift(1)
print(sn1)
sn2 = yt.shift(2)
print(sn2)
sn3 = yt.shift(3)
print(sn3)
sn4 = yt.shift(4)
print(sn4)
sn5 = yt.shift(5)
print(sn5)

sn_cat=pd.concat([sn1,sn2,sn3,sn4,sn5],axis=1)
print(sn_cat)
sn_cat_na = sn_cat.dropna()
print(sn_cat_na)
sn_cat_na_index = sn_cat_na.reset_index(drop=True)
print(sn_cat_na_index)

sn_cat_na_index.to_csv("doc/time_series.csv")

plt.plot(yt)
plt.show()

