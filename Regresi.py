#Buat Sebuah data dumy
import numpy as np


#data kamar
kamar=np.array([1,1,2,2,3,4,4,5,5,5])

#data harga tiap kamar, asumsi adalah sebuah dolar
harga_rumah=np.array([15000,18000,27000,34000,50000,68000,65000,81000,85000,90000])

"""
Kamar=>data x(indepent)->fitur
harga_rumah=>data y(dependent)->label
"""

#train data set model
from sklearn.linear_model import LinearRegression
#latih dengan model linear regresesion.fit()
kamar=kamar.reshape(-1,1)
print(kamar)
#assigment model
linreg=LinearRegression()
#train dataset
linreg=linreg.fit(kamar,harga_rumah)
#testing data set
linreg=linreg.predict(kamar)#predict ini untuk testing, data yang di testing adalah data fiturnya
#predict mengembalikan sebuah nilai yang sudah di train untuk di testing
#visualisasi
import matplotlib.pylab as plt
fig,ax=plt.subplots()
ax.scatter(kamar,harga_rumah)
#garis lurus untuk regresi
ax.plot(kamar,linreg)
plt.show()