import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV

# imbalance data
from imblearn.pipeline import Pipeline as pline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn import naive_bayes




def mapping_nama(y, y_trans):
    coba_ya = np.unique(y_trans, return_index=True)

    labelnya, indexnya = coba_ya

    label_asli = {}
    for x in range(len(labelnya)):
        label_asli[indexnya[x]] = labelnya[x]

    sorted_list = sorted(label_asli)

    hasilnyaa = {}
    for k in sorted_list:
        hasilnyaa[k] = label_asli[k]

    y_trans_unique = [v for k, v in hasilnyaa.items()]

    mappingnya = {}
    for x in range(len(y_trans_unique)):
        mappingnya[y_trans_unique[x]] = y.unique()[x]

    return mappingnya


st.markdown(
    """
    # Aplikasi Klasifikasi Bidang Masalah Layanan Aspirasi dan Pengaduan Masyarakat DPR RI

    ### Tugas Akhir : Muhammad Bondan Vitto Ramadhan
    
    """
)
#df = pd.read_excel('Data 7 Kategori.xlsx',header=0)
df = pd.read_csv("Data-7-Kategori.csv")
doc_clean_df = pd.read_csv("Perihal_Pengaduan_bersih.csv")

x = df.perihal
y = df.bidang_masalah

le = preprocessing.LabelEncoder()
le.fit(y)
y_trans=le.transform(y)

x_bersih = doc_clean_df["0"]
x1_train, x1_test, y1_train, y1_test = train_test_split(x_bersih, y_trans, test_size=0.1, random_state=0, stratify=y_trans)

vectorizer = TfidfVectorizer(max_features=None, min_df=5, max_df=0.7)
vectorizer.fit(x1_train.values.ravel())
X_train=vectorizer.transform(x1_train.values.ravel())
X_train=X_train.toarray()

over = SMOTE(random_state = 10, k_neighbors=8)
under = RandomUnderSampler(random_state = 10)
balancing = pline(steps=[('over', over), ('under', under)])
x_bal, y_bal = balancing.fit_resample(X_train, y1_train)


param_grid = {'alpha': [0.1,0.3,0.5,0.7,1]}
grid_NB = GridSearchCV(naive_bayes.MultinomialNB(), param_grid=param_grid, cv=3, n_jobs=-1)

grid_NB.fit(x_bal, y_bal)

prediksi_ini = st.text_input("Masukkan kata-kata")
prediksi_ini = pd.Series(prediksi_ini)
prediksi_ini = vectorizer.transform(prediksi_ini.values.ravel())
prediksi_ini = prediksi_ini.toarray()

# temp = grid_NB.predict(prediksi_ini)
hasil = grid_NB.predict(prediksi_ini)

yokbisa = mapping_nama(y, y_trans)



st.write(yokbisa[hasil[0]])
st.write(hasil[0])