import pandas as pd
from sklearn.datasets import load_iris
#load dataset
iris=pd.read_csv("D:/Machine Learning/Coding/Supervised/Iris.csv")
print(iris.head())#melihat 5 baris pertama pada file iris
#menghilangkan data yang tidak penting->id
#axis=1->kolom,axis=0->baris
iris=iris.drop("Id",axis=1)
print(iris)

#pisahkan data set menjadi atribut(fitur) dan label
X=iris[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=iris["Species"]

#membuat model desicion tree
from sklearn.tree import DecisionTreeClassifier
desicion_tree=DecisionTreeClassifier()
#melakukan pelatihan model(data trainer)
model_tree=desicion_tree.fit(X,y)
#fit adalah sebuah methode yang digunakan untuk meng-trainer data, latihan tergantung dari class (jenis model) yang memanggilnya

#testing data
model_tree=model_tree.predict([[5.1,3.5,1.4,0.2]])
print(model_tree)

#visualisasi
from sklearn.tree import export_graphviz
export_graphviz(
    desicion_tree,
    out_file="iris_tree.dot",
    feature_names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],
    class_names=["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    rounded=True,
    filled=True
)