import pandas as pd
import matplotlib.pyplot as plt
xls = pd.ExcelFile('doc/tc.xlsx')
print(xls.sheet_names)
exl = pd.read_excel(xls,'A1. Headline series')
print(exl.iloc[201,0])
print(exl.iloc[361,0])

unemployment = exl.iloc[201:361,15]
inflation = exl.iloc[201:361,28]
bank_rate = exl.iloc[201:361,30]
debt = exl.iloc[201:361,57]
GDP_trend = exl.iloc[201:361,3]
x=pd.concat([GDP_trend,debt,bank_rate,inflation],axis=1)
x.columns = ["GDP_trend","debt","bank_rate" ,"inflation"]
print(x.dtypes)

x["GDP_trend"] = x["GDP_trend"].astype('float64')
x["debt"] = x["debt"].astype('float64')
x["bank_rate"] = x["bank_rate"].astype('float64')
x["inflation"] = x["inflation"].astype('float64')
x.reset_index(drop=True, inplace=True)
print(x.dtypes)

y = pd.to_numeric(unemployment)
y.to_csv("doc/economic_y.csv",index_label='index')
y.reset_index(drop=True, inplace=True)
x.to_csv("doc/economic_x.csv",index_label='index')






#plt.plot(unemployment)
#plt.show()