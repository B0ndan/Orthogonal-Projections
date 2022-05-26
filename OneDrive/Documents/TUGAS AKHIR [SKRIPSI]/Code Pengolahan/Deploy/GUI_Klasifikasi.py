from turtle import title
from requests import options
import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

#Preprocessing And TF-IDF
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from collections import OrderedDict

# imbalance data
from imblearn.pipeline import Pipeline as pline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn import naive_bayes

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,KFold,ShuffleSplit,StratifiedShuffleSplit

# Library Baru
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


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


# Untuk Bikin Navigation Bar
with st.sidebar:
    selected3 = option_menu("Main Menu", ["Home", "Projects", 'Contact'], 
        icons=['house', 'book', 'cloud-upload'], menu_icon="cast", default_index=1) #, orientation = "horizontal")
    selected3

if selected3 == "Home":
    st.header('Apa Itu Aplikasi Klasifikasi Bidang Masalah Dari Layanan Aspirasi dan Pengaduan Masyarakat DPR RI?')
    st.text('Graphic User Interface (GUI) menunjukkan tampilan penggunaan model dalam klasifikasi')
    st.text('kalimat pada perihal berdasarkan kategori bidang masalah yang ada pada Layanan')
    st.text('Aspirasi dan Pengaduan Masyarakat DPR RI. Desain perangkat lunak yang akan dibangun')
    st.text('pada penelitian tugas akhir ini menggunakan aplikasi web yang dibangun dengan')
    st.text('Python dan framework Streamlit untuk memproses data dan menampilkan')
    st.text('data hasil dari klasifikasi.')
    st.code('Kelas atau Kategori penelitian ini mewakilkan 7 kelas yaitu:')
    st.code('Hukum, Pertahanan dan Reforma Agraria, Tenaga Kerja, Pendidikan, Energi Sumber Daya dan Mineral, Kesehatan, dan Lain-lain.')
    with st.form(key='my_form'):
        username = st.text_input('Username/e-mail')
        password = st.text_input('Password')
        st.form_submit_button('Login')

if selected3 == "Projects":
    st.markdown(
        """
        ## Aplikasi Klasifikasi Bidang Masalah Dari Layanan Aspirasi dan Pengaduan Masyarakat DPR RI
        #### Tugas Akhir : Muhammad Bondan Vitto Ramadhan
        
        """
    )
    st.image('Foto_DPR.jpg')
    #st.image('FotoBersamaBuPuan.jpg')
    #st.video('DPR RI Internship Experience.mp4')

    df = pd.read_csv("Kategori_Bidang_Masalah.csv")
    doc_clean_df = pd.read_csv("Perihal_Pengaduan_Bersih.csv")

    #x = df.perihal
    y = df.bidang_masalah

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_trans=le.transform(y)

    x_bersih = (doc_clean_df)['0']

    vectorizer = TfidfVectorizer(max_features=None, min_df=0., max_df=1.0)
    vectorizer.fit(x_bersih.values.ravel())
    X_Clean=vectorizer.transform(x_bersih.values.ravel())
    X_Clean=X_Clean.toarray()

    #x1_train, x1_test, y1_train, y1_test = train_test_split(X_Clean, y_trans, test_size=0.1, random_state=0, stratify=y_trans)

    #vectorizer = TfidfVectorizer(max_features=None, min_df=0., max_df=1.0)
    #vectorizer.fit(x1_train.values.ravel())
    #X_train=vectorizer.transform(x1_train.values.ravel())
    #X_train=X_train.toarray()


    over = SMOTE(random_state = 10, k_neighbors=8)
    under = RandomUnderSampler(random_state = 10)
    balancing = pline(steps=[('over', over), ('under', under)])
    x_bal, y_bal = balancing.fit_resample(X_Clean, y_trans)

    #==================================================================================================
    # SVM Sigmoid
    modellin_sigmoid = SVC(kernel='sigmoid', probability=True, C=100, gamma=0.1)
    clf_lin = modellin_sigmoid.fit(x_bal,y_bal)
    #==================================================================================================
    # NAIVE BAYES LAMA
    #param_grid = {'alpha': [0.1,0.3,0.5,0.7,1]}
    #grid_NB = GridSearchCV(naive_bayes.MultinomialNB(), param_grid=param_grid, cv=3, n_jobs=-1)
    #grid_NB.fit(x_bal, y_bal)
    #==================================================================================================

    with st.form(key='my_form'):
        text_input = st.text_input(label='Masukkan Kata-kata atau Kalimat yang Ingin Diklasifikasikan')
        submit_button = st.form_submit_button(label='Submit')

    prediksi_ini = text_input

    documents = []
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    x1 = prediksi_ini
    def remove_all_extra_spaces(string):
        return " ".join(string.split())

    for sen in range(0, len(x1)):
        text = re.sub(r'#([^\s]+)', r'\1', x1)
        document = re.sub(r'\W', ' ', text)
        document1 = re.sub(r'\s+[a-zA-Z]\s+', '', document)
        document2 = re.sub(r'\^[a-zA-Z]\s+', '', document1) 
        document3 = re.sub(r"\d+", "", document2)
        document4 = re.sub(r'\s+', ' ', document3, flags=re.I)
        document5 = re.sub(r'^b\s+', '', document4)
        document5 = remove_all_extra_spaces(document5)
        document6 = document5.lower()
        document7 = document6.split()
        document8 = stemmer.stem(document6)
        document9 = ''.join(document8)
        documents.append(document9)
    Wtd = documents

    factory = StopWordRemoverFactory() 
    stopword1 = factory.get_stop_words()
    more_stopwords=['mohon', 'yang', 'berdasarkan', 'ada', 'kepala', 'timur', 'kantor', 'pt',' pt', 'pihak', 'bidang',
                    'ata','ii', 'iii', 'tertanggal', 'melalui', 'jo', 'menjadi', 'terletak', 'tidak', 'ganti', 'di', 'pn',
                    'sdr', 'res', 'tgl', 'mengenai', 'tahun', 'su', 'ri', 'ix', 'atas', 'melalui', 'tanggapan', 'tentang', 'diduga', 
                    'kec', 'adanya', 'ada', 'tengah', 'pernyataan', 'tembusan', 'sesuai', 'ii', 'iii', 'iiiin', 'iv', 'ix', 'vi', 'vii', 'viii', 'xii', 'xiii',
                    'selua', 'sh', 'bapak', '', 'hgu', 'ma', 'su', 'ham', 'perihal', 'milik', 'satu', 'tidak', ' narada', ' di', 'narada ',
                    'nomor','atas','pk', 'okt', 'agustus','juli','april','terhadap','kedua','jaya','untuk','bin','upaya','melalui','tentang','februari','dilakukan','pusat','selatan',
                    'atas','data','lp','dalam','juni','adanya','mengenai','jkt','atau','jawaban','tinggi','telah','maret','bapak','oktober', 'januari', 'juli', 'mei','september',
                    'xi','agung','ada','dengan','kedua','di','selatan','nama','ada','terkait', 'tentang','yang','nomor','tidak','dengan','terhadap','sept', 'november', 'nov',
                    'dalam','atau','bapak','nama','kami','ada','melalui', 'assalamualaikum', 'wr', 'wb', 'jp', 'lp', 'md','mh' , 'melakukuan', 'sbg', 'selasa'
                    'oleh','segera','tahun','melakukan','oleh','agustus','atau','dki','kab','belum','untuk','adanya','kecamatan', 'yang', 'yg', 
                    'memberikan','mengenai','ayat','tanggal','dan','bukan','dab', 'dan','ke','qq']
    sw = stopword1 + more_stopwords
    dictionary = ArrayDictionary(sw) 
    strw = StopWordRemover(dictionary) 
    removestop=[]
    for line in Wtd : 
        word_token = nltk.word_tokenize(line) 
        word_token = [word for word in word_token if not word in sw]
        removestop.append(" ".join(word_token)) 
    doc_clean = removestop

    kata1 = {
            "adatno":"adat","admnistrasi":"administrasi", "ahali":"ahli","agutus":"agustus", "asset":"aset", "bantenh":"banten", "baratq":"barat", "bareskim":"bareskrim", 
            "cengkaraeng":"cengkareng", "consultan":"consultant", "demostrasi":"demonstrasi", "dgn":"dengan", "dugan":"duga", "gubenur":"gubernur", "hipoteisnya":"hipotesis", 
            "hukumabd":"hukum", "hukumn":"hukum", "illegal":"ilegal", "indonesi":"indonesia", "jatinagor":"jatinangor", "jendral":"jenderal", "jenerbarang":"jeneberang", 
            "karen":"karena", "ketenagakerjaam":"ketenagakerjaan", "klarfikasi": "klarifikasi", "kod":"kpd", "kpd":"kepada", "kuh":"kuhp", "lh":"lhk", "manfaayt":"manfaat",
            "merugikjan":"rugi", "merugkan":"rugi", "negri":"negeri", "nurokhimhh":"nurokhim", "omibus":"omnibus", "omnimbus":"omnibus", "paraktik":"praktik", 
            "pelangaran":"langgar", "pelangaaran":"langgar", "pelanikan":"lantik", "pemprof":"pemprov", "pemrohonan":"mohon", "pengadialan":"adil", 'provinsiinsi':'provinsi',
            "perlingungan":"lindung", "permaslahan":"masalah", "permohoan":"mohon", "permohoanan":"mohon", "permohonann":"mohon", "pernytaan":"nyata", "petujunjuk":"tunjuk", 
            "pidan":"pidana", "praktek":"praktik", "pratik":"praktik", "pres":"presiden", "profosional":"profesional", "propinsi":"provinsi", "prov":"provinsi", 
            "rekayas":"rekayasa", "rill":"riil", "sangerengh":"sangereng", "sebagaimna":"bagai", "sebgai":"bagai", "sekrteariat":"sekretariat", "seripikat":"sertifikat",
            "sertipikat":"sertifikat", "sertikat":"sertifikat", "sesui":"sesuai", "swata":"swasta", "tanag":"tanah", "taruana":"taruna", "temapat":"tempat", "tidka":"tidak",
            "tsb":"sebut", "ttg":"tentang", "unsure":"unsur", "utk":"untuk", "warisa":"waris", "wenag":"wenang"
            }

    def replace_all(text, dic): 
        for i, j in dic.items(): 
            text = text.replace(i, j) 
        return text 
    dic = OrderedDict(kata1) 
    doc_clean_new = [] 
    for line in doc_clean: 
        result = replace_all(line, dic) 
        doc_clean_new.append(result)

    prediksi_ini = doc_clean_new
    prediksi_ini = pd.Series(prediksi_ini)
    prediksi_ini = vectorizer.transform(prediksi_ini.values.ravel())
    prediksi_ini = prediksi_ini.toarray()

    #hasil = grid_NB.predict(prediksi_ini)
    hasil = modellin_sigmoid.predict(prediksi_ini)
    yokbisa = mapping_nama(y, y_trans)

    st.text('Hasil Klasifikasi Kategori:')

    with st.spinner(text='In progress'):
        st.write(yokbisa[hasil[0]])
        st.write(hasil[0])
        st.success('Classification Success')


if selected3 == "Contact":
    st.header('About Me')
    st.subheader('Muhammad Bondan Vitto Ramadhan')
    st.subheader('NRP : 06211840000086')
    st.image('Foto_Di_Ketua_DPR_2.jpg')
    st.text('Contact Me Through : ')    
    st.text('Email      : bondanvitto1@gmail.com') 
    st.text('LinkedIn   : linkedin.com/in/bondanvitto')   
    st.text('Instagram  : bondanvitto') 


