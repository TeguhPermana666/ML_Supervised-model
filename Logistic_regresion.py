#membuat sebuah data frame

#include data external
import pandas as pd
from pandas.core.tools.datetimes import Scalar
file=pd.read_csv("D:\Machine Learning\Coding\Supervised\Social_Network_Ads.csv")
print(file.head())
print("\n")
print(file.info())#melihat apakah ada data yang kosong->lebih terlihat seperti table query

#bersihkan id->primary keynya karena akan merusak sebuah model machnie learning
print("Hapus primary key data")
file=file.drop("User ID",axis=1)
print(file.head())
#membersihkan data
#one hot decoding yakni proses merubah data kategorik pada gender(object->dtype) menjadi numerik
file=pd.get_dummies(file)
#karena sebuah machine learning pada model kurng bisa menangkap jenis data kategorik secara langsung
print(file.head())

#pisahkan fitur dan label
X=file[["Age","EstimatedSalary","Gender_Female","Gender_Male"]]
Y=file["Purchased"]

#normalisasi data
"""
   Age  EstimatedSalary  Purchased  Gender_Female  Gender_Male
0   19            19000          0              0            1
1   35            20000          0              0            1
2   26            43000          0              1            0
3   27            57000          0              1            0
4   19            76000          0              0            1

data pada skala fitur age 19-35
estimatedsalary 19k-76k
purched->0
gender f&m=>0-1
jadinya skala di setiap fitur tidak sama atau mendekati di setiap limit nilai
"""
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler=scaler.fit(X)
scaler=scaler.transform(X)
print("Normalization\n")
print(scaler)
"""
[[0.02380952 0.02962963 0.         0.         1.        ]
 [0.4047619  0.03703704 0.         0.         1.        ]
 [0.19047619 0.20740741 0.         1.         0.        ]
 ...
 [0.76190476 0.03703704 1.         1.         0.        ]
 [0.42857143 0.13333333 0.         0.         1.        ]
 [0.73809524 0.15555556 1.         1.         0.        ]]
"""
scaler=pd.DataFrame(scaler, columns=X.columns)
print(scaler.head())
#membagi data set menjadi train dan test data
from sklearn.model_selection import train_test_split
#scaler=X (yang sudah mengalami normalisasi)
X_train,X_test,Y_train,Y_test=train_test_split(scaler,Y,test_size=0.3,random_state=42)
# print(X_train)
print(X_test)
# print(Y_train)
print(Y_test)

#buat model Machine learningnya
#Train data Pada data set train
from sklearn import linear_model
#latih model dengan fungsi fit()
model=linear_model.LogisticRegression()
model=model.fit(X_train,Y_train)
print(model)
"""
Setelah melatih sbeuah data train pada data set 70% data train
dapat dilakukan validasi dengan data testing dimana mengambil sudah di split tadi
pengujian data trainer (data testing) dpt dilakukan dengan fungsi score

"""
hasil=model.score(X_test,Y_test)
print(hasil)
"""
Haisl=> 0.8416666666666667
yang berarti ada 0.16% signifikasi kesalahan data
dimana hasil mendekati satu yang berarti 

dengan komponen :

age,gender(m&f),estimatedsalary (x)->variabel indepent/bebas

terhadap

y:purschased->variabel depent(terikat)
yg berarti komponen tersebut dalam pengiklanan berpengaruh terhadap pembelian barang
jadi seseorang yang berkaitan terhadap berbagai macam komponen yang bebas age,gender(m&f),estimatedsalary
setelah menonton iklan, kemungkinan seseorang tersebut membeli suatu produk sangat besar dengan tingkat pengaruh estimasi
adalah 0.842 yg menandakan mendekati satu 
"""