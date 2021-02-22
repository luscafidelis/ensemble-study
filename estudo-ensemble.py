# -*- coding: utf-8 -*-
"""
Script para testar a performance de classificação dos métodos ensemble
"""

#Importação dos pacotes utilizados para o estudo
import pandas as pd
import os

#Importação dos métodos ensemble
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

#Importação dos metodos de classificação de modelo único
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

#Importação do pacote que divide a base em teste e treino
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

root = 'bases'
extension = '.csv'

for subdir, dirs, files in os.walk(root):
    for file in files:

        evaluation = {}        

        #Lê a base de dados para o dataframe
        data = pd.read_csv(os.path.join(subdir, file))
        label = data.iloc[:, -1:]
        attri = data.iloc[:,:-1]
        
        #divide a base em teste e treino
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        skf.get_n_splits(attri,label)
        
        #KNeighborsClassifier
        knn = KNeighborsClassifier()
        print('knn')


        #Support Vector Machine
        svm = SVC()
        print('svc')

        #Decision Tree
        dt = DecisionTreeClassifier()
        print('dt')

        #Neural Network
        nn = MLPClassifier()
        print('nn')

        #Naive Bayes
        nb = GaussianNB()
        print('nn')

        
        #Bagging
        bagging_knn = BaggingClassifier(knn)
        bagging_svm = BaggingClassifier(svm)
        bagging_dt = BaggingClassifier(dt)
        bagging_nn = BaggingClassifier(nn)
        bagging_nb = BaggingClassifier(nb,)


        #Boosting
        boosting_svm = AdaBoostClassifier(svm, algorithm='SAMME')
        boosting_dt = AdaBoostClassifier(dt)
        boosting_nb = AdaBoostClassifier(nb)

        #Stacking - Define os classificadores base
        estimators = [
            ('knn',knn),
            ('svm',svm),
            ('dt',dt),
            ('nn',nn),
            ('nb',nb)
            ]

        #define o meta-classificador
        stacking_knn = StackingClassifier(estimators=estimators, final_estimator=knn)
        stacking_svm = StackingClassifier(estimators=estimators, final_estimator=svm)
        stacking_dt = StackingClassifier(estimators=estimators, final_estimator=dt)
        stacking_nn = StackingClassifier(estimators=estimators, final_estimator=nn)
        stacking_nb = StackingClassifier(estimators=estimators, final_estimator=nb)
        
        for train_index,test_index in skf.split(attri,label):
            attri_train = attri.iloc[train_index]
            attri_test = attri.iloc[test_index] 
            
            label_train = label.iloc[train_index].values.ravel()
            label_test = label.iloc[test_index].values.ravel()

            #Treina os classificadores de único aprendizado
            knn.fit(attri_train,label_train)
            svm.fit(attri_train,label_train)
            dt.fit(attri_train,label_train)
            nn.fit(attri_train,label_train)
            nb.fit(attri_train,label_train)            

            #Treina os ensemble
            bagging_knn.fit(attri_train,label_train)
            bagging_svm.fit(attri_train,label_train)
            bagging_dt.fit(attri_train,label_train)
            bagging_nn.fit(attri_train,label_train)
            bagging_nb.fit(attri_train,label_train)

            boosting_svm.fit(attri_train,label_train)
            boosting_dt.fit(attri_train,label_train)
            boosting_nb.fit(attri_train,label_train)

            stacking_knn.fit(attri_train,label_train)
            stacking_svm.fit(attri_train,label_train)
            stacking_dt.fit(attri_train,label_train)
            stacking_nn.fit(attri_train,label_train)
            stacking_nb.fit(attri_train,label_train)
            

            #Avalia os classificadores            
            predictions = knn.predict(attri_test)            
            evaluation['kNN Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['kNN Macro_F1'] = f1_score(label_test, predictions, average='macro')

            predictions = svm.predict(attri_test)            
            evaluation['SVM Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['SVM Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = dt.predict(attri_test)            
            evaluation['DT Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['DT Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = nn.predict(attri_test)            
            evaluation['NN Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['NN Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = nb.predict(attri_test)            
            evaluation['NB Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['NB Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = bagging_knn.predict(attri_test)            
            evaluation['Bagging_kNN Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Bagging_kNN Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = bagging_svm.predict(attri_test)            
            evaluation['Bagging_SVM Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Bagging_SVM Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = bagging_dt.predict(attri_test)            
            evaluation['Bagging_DT Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Bagging_DT Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = bagging_nn.predict(attri_test)            
            evaluation['Bagging_NN Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Bagging_NN Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = bagging_nb.predict(attri_test)            
            evaluation['Bagging_NB Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Bagging_NB Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = boosting_svm.predict(attri_test)            
            evaluation['Boosting_SVM Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Boosting_SVM Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = boosting_dt.predict(attri_test)            
            evaluation['Boosting_DT Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Boosting_DT Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = boosting_nb.predict(attri_test)            
            evaluation['Boosting_NB Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Boosting_NB Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = stacking_knn.predict(attri_test)            
            evaluation['Stacking_kNN Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Stacking_kNN Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = stacking_svm.predict(attri_test)            
            evaluation['Stacking_SVM Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Stacking_SVM Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = stacking_dt.predict(attri_test)            
            evaluation['Stacking_DT Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Stacking_DT Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = stacking_nn.predict(attri_test)            
            evaluation['Stacking_NN Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Stacking_NN Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            predictions = stacking_nb.predict(attri_test)            
            evaluation['Stacking_NB Micro_F1'] = f1_score(label_test, predictions, average='micro')
            evaluation['Stacking_NB Macro_F1'] = f1_score(label_test, predictions, average='macro')
            
            results = pd.DataFrame([evaluation])
            results.to_csv("results_"+file)