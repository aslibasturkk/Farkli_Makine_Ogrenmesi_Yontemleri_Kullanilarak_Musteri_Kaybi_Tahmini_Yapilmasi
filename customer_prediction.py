"""
DİZİ-FİLM İZLEME PLATFORMUNDA 
MÜŞTERİNİN ABONELİĞİNİ İPTAL EDİP ETMEYECEĞİNİN MAKİNE ÖĞRENMESİ MODELLERİYLE
TAHMİN EDİLMESİ

Amaç: Bir dizi-film platformunda müşterilerinin üyeliğini iptal edip etmeyeceğini tahmin etmek.

    1)Veri Seti Açıklaması:
    Veri kaynağı: https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset
    Bu projede kullanılacak olan veri setinde toplam 5000 kayıt,14 adet feature değeri bulunmaktadır.
         Sütunlar:
        -Customer_id
        -Age
        -Gender
        -Subscription_type
        -watch_hours
        -last_login_days
        -region
        -device
        -monthly_fee
        -churned
        -payment_method
        -number_of_profiles
        -avg_watch_time_per_day
        -favorite_genre
        
    2)Kullanılacak ML Yöntemleri:
        -Logistic Regression
        -Random Forest

    3)İzlenilecek Adımlar:
        1)Veri seti yükleme
        2)Veri setini inceleyerek eksik ve tutarsız değerleri gözden geçirme
        3)feature-hedef değişken oluşturma
        4)Train_test_split için verileri oluşturma
        5)Eğitim gerçekleştirme ve test etme
        6)Doğruluk oranını bulma ve confusion matris oluşturma
        7)Hiperparametreleri belirleme
        8)Sonu.ları görselleştirme
"""
#1 Veri setini ve ilgili kütüphaneleri yükleme
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#1 dosyayı okuma
path = kagglehub.dataset_download("abdulwadood11220/netflix-customer-churn-dataset")
csv_file = glob.glob(f"{path}/*.csv")[0]
df = pd.read_csv(csv_file)
df.head(10)

#2 eksik/tutarsız değer kontrolü
if df.isna().any().any():
    df.dropna()(inplace=True)
    print("Eksik değerler silindi.")
else:
    print("Eksik değer yok.")

#3 öznitelik(x) ve Hedef değişken(y) oluşturma
#Metinsel verilere one-hot encoding gerçekleştirme
df=df.drop(columns=["Customer_id"],errors="ignore")
X=df.drop(columns=["churned"])
y=df["churned"]
#Kategorik verileri sayısal formata dönüştürme
X=pd.get_dummies (X, drop_first=True)


#4 Train_etst_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=78)
#6 Model Eğitimi
log_reg=LogisticRegression(penalty="l2",C=1,max_iter=1000)
log_reg.fit(X_train,y_train)
log_reg_pred=log_reg.predict(X_test)

#7 Test 
acc= log_reg.score(X_test,y_test)
print(f"Lojistik regresyonla elde edilen doğruluk:{acc}")


#KARAR AGACI İLE TAHMİN

tree_clf=DecisionTreeClassifier(criterion="gini",max_depth=14,random_state=78)
random_forest_clf=RandomForestClassifier(n_estimators=100,max_depth=2,random_state=78)
tree_clf.fit(X_train,y_train)
random_forest_clf.fit(X_train,y_train)

tree_y_pred=tree_clf.predict(X_test)
random_forest_pred=random_forest_clf.predict(X_test)

acc_decision_tree=tree_clf.score(X_test,y_test)
print(f"Karar ağacı için doğruluk: {acc_decision_tree}")

acc_random_forest=random_forest_clf.score(X_test,y_test)
print(f"Random Forest için doğruluk: {acc_random_forest}")

#SONUÇLARIN GÖRSELLEŞTİRİLMESİ
cm_random_forest = confusion_matrix(y_test, random_forest_pred)
cm_log=confusion_matrix(y_test, log_reg_pred)
cm_decision_tree=confusion_matrix(y_test, tree_y_pred)


#Logistic regression matrisini görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Abone (0)', 'Abonelik iptal (1)'], 
            yticklabels=['Abone (0)', 'Abonelik iptal(1)'])

plt.title('Logistic regression - Karmaşıklık Matrisi')
plt.xlabel('Tahmin edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.show()

#Karar ağacı ile confusion matris oluşturma
plt.figure(figsize=(6, 4))
sns.heatmap(cm_decision_tree, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Abone (0)', 'Abonelik iptal (1)'], 
            yticklabels=['Abone (0)', 'Abonelik iptal(1)'])

plt.title('karar ağacı - Karmaşıklık Matrisi')
plt.xlabel('Tahmin edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.show()

#Random Forest Matrisini Görselleştirme 
plt.figure(figsize=(6, 4))
sns.heatmap(cm_random_forest, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Abone (0)', 'Abonelik iptal (1)'], 
            yticklabels=['Abone (0)', 'Abonelik iptal(1)'])

plt.title('Random Forest - Karmaşıklık Matrisi')
plt.xlabel('Tahmin edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.show()












