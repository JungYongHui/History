import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE, ADASYN
import webbrowser

def main():
    page = st.sidebar.selectbox('Choose a page', ['Homepage', 'Machine Learning'])
    
    if page == 'Homepage':
        
        image = Image.open('Welcome.jpg')
        st.image(image, caption='@Adobe', use_column_width=True)
        st.title('Exoplanet Hunting in Deep Space')
        image = Image.open('Deep Space.jpg')
        st.image(image, caption='@iStockphoto', use_column_width=True)
        st.info('To optimize web application, We used Pre-processed Data.')
        st.warning('Original Data is 250MB(train), 28MB(test) / Preprocessed Data is 7MB(train), 1MB(test).')
        st.success('This Web Application is to predict the existense of exoplanets based on the FLUX(to be observed in the future)')
        st.success('It is also to inform you how to treat imbalanced data.')
        st.success('We hope it helps you when you treat imbalanced data. Good Luck!')
        st.markdown('## Take a look! Just check the box below.')
        
        if st.button('Original Data and Description'):
            webbrowser.open_new_tab('https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data')
        
        if st.checkbox('View Part of the Data'):
            st.markdown('### Outliers Removed, Dimension Reducted, Data Scaled.')
            st.text('Removed Outliers : 51\nOriginal Dimension : 3197\nReducted Dimension : 73\nScaling : StandardScaler')
            st.markdown('#### Train Data')
            st.write(exo_train.head(15))
            st.markdown('#### Test Data')
            st.write(exo_test.head(15))
    
    elif page == 'Machine Learning':
        
        navigate = st.sidebar.selectbox('Navigate',
                                        ['1. Data Pre-processing',
                                         '2. Classification',
                                         '3. Classification with SMOTE',
                                         '4. Classification with ADASYN',
                                         '5. Cost-Sensitive Learning',
                                         '6. Conclusion'])
        
        if navigate == '1. Data Pre-processing':
            # 1
            st.markdown('# Data Pre-processing')
            image = Image.open('Data_preprocessing.png')
            st.image(image, caption='@Towards Data Science', use_column_width=True)
            st.markdown('## Loading Data')
            st.markdown('#### Summary')
            temp = pd.DataFrame({'LABEL':[5050, 37]}, index=[1, 2])
            st.write(temp)
            st.text('Exoplanet stars are {:.2f}% of total.'.format(100*37/5087))
            
            # 2
            st.markdown('## Checking for Missing Values')
            st.text('Total Missing values in train data : {}'.format(exo_train.isnull().sum().sum()))
            col_miss = exo_train.isnull().sum()
            row_miss = exo_train.isnull().sum(1)
    
            occupation = st.selectbox('Choose Column/Row', ['None', 'Column', 'Row'])
            if occupation=='Column':
                st.write(col_miss)
            elif occupation=='Row':
                st.write(row_miss)
                
            # 3    
            st.markdown('## Detecting Outliers')
            st.markdown('### with IsolationForest, contamination_rate = 1%')
            st.markdown('#### Code')
            
            code = '''from sklearn.ensemble import IsolationForest
            
clf = IsolationForest(n_estimators=100, max_samples='auto', 
                      contamination=float(0.01), 
                      max_features=1.0, bootstrap=True, 
                      n_jobs=-1, random_state=0, verbose=0)
clf.fit(exo_train.iloc[:, 1:])
exo_train['anomaly'] = clf.predict(exo_train.iloc[:, 1:])'''
            st.code(code, language='python')
            temp = pd.DataFrame({'ANOMALY':[5000, 51]}, index=[1, -1])
            st.write(temp)
            st.text('Outliers in Exoplanet stars : 1')

            code = '''count = 0
for i in range(len(exo_train)):
    if (exo_train['LABEL'][i]==2)&(exo_train['anomaly'][i]==-1):
        count+=1
print('Outliers in Exoplanet stars : ', count)'''
            st.code(code, language='python')
            
            # 4
            st.markdown('## Removing Outliers')
            st.markdown('#### Code')
            
            code = '''exo_train.drop(exo_train.loc[exo_train['anomaly']==-1].index, inplace=True)
exo_train.drop(['anomaly'], axis='columns', inplace=True)'''
            st.code(code, language='python')
            temp = pd.DataFrame({'LABEL':[5000, 36]}, index=[1, 2])
            st.write(temp)
            
            # 5
            st.markdown('## Data Scaling')
            st.markdown('### with StandardScaler')
            st.markdown('#### Code')
            
            code = '''from sklearn.preprocessing import StandardScaler
            
X_train = exo_train.iloc[:, 1:]
X_test = exo_test.iloc[:, 1:]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
y_train = exo_train[['LABEL']]
y_test = exo_test[['LABEL']]'''
            st.code(code, language='python')
                
            # 6
            st.markdown('## Dimension Reduction')
            st.markdown('### with Principal Component Analysis')
            st.markdown('#### Code')
            
            code = '''from sklearn.decomposition import PCA
            
pca = PCA()
pca.fit(X_train_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dimension = np.argmax(cumsum>=0.95)
pca = PCA(n_components=dimension)
pca.fit(X_train_scaled)
pca_X_train = pca.transform(X_train_scaled)
pca_X_test = pca.transform(X_test_scaled)'''
            st.code(code, language='python')
                
            st.text('Number of dimensions with 95% variance : 73')
            st.text('Size of the train data : (5036, 73)')
            
            # 7
            st.markdown('## Label Binarization')
            st.markdown('#### Code')
            
            code = '''y_train.loc[y_train['LABEL'] == 1, 'BINARY'] = 0
y_train.loc[y_train['LABEL'] > 1, 'BINARY'] = 1
y_test.loc[y_test['LABEL'] == 1, 'BINARY'] = 0
y_test.loc[y_test['LABEL'] > 1, 'BINARY'] = 1'''
            st.code(code, language='python')
            
            st.write(y_train.head(10))

            # 8
            st.markdown('## Save Pre-processed data to CSV File')
            st.markdown('#### Code')
            
            code = '''import pandas as pd
            
df_y_train = pd.DataFrame(y_train)
df_X_train = pd.DataFrame(pca_X_train)
df_concat_train = pd.concat([df_y_train.reset_index(), df_X_train], axis=1)
df_concat_train.drop(['index'], axis='columns', inplace=True)
df_y_test = pd.DataFrame(y_test)
df_X_test = pd.DataFrame(pca_X_test)
df_concat_test = pd.concat([df_y_test, df_X_test], axis=1)
df_concat_train.to_csv('exoTrain_preprocessed.csv', header=True, index=False)
df_concat_test.to_csv('exoTest_preprocessed.csv', header=True, index=False)'''
            st.code(code, language='python')
            
        elif navigate == '2. Classification':
            # 1
            st.markdown('# Classification')
            image = 'classification.jpg'
            st.image(image, caption='@Analytics Vidhya', use_column_width=True)
            st.warning('As can be seen below, Models that are build without handling imbalance of data are not worth as classifiers.')
            st.markdown('## Support Vector Machine')
            if st.checkbox('Polynomial Kernel'):
                st.markdown('### SVM with Polynomial Kernel')
                model = joblib.load('Poly_SVM.model')
                result_label(model)
            
            # 2
            if st.checkbox('RBF Kernel'):
                st.markdown('### SVM with RBF Kernel')
                model = joblib.load('RBF_SVM.model')
                result_label(model)
            
            # 3
            if st.checkbox('Sigmoid Kernel'):
                st.markdown('### SVM with Sigmoid Kernel')
                model = joblib.load('Sigmoid_SVM.model')
                result_label(model)
            
            # 4
            st.markdown('## Logistic Regression')
            if st.checkbox('Result'):
                model = joblib.load('logit.model')
                result_binary(model)
            
                if st.checkbox('ROC Curve'):
                    fpr, tpr, thresholds = roc_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                    logit_auc = auc(fpr, tpr)
                    plt.figure()
                    plt.plot(fpr, tpr, label='Logistic Regression (area = {:.4f})'.format(logit_auc))
                    plt.plot([0, 1], [0, 1],'r--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver operating characteristic')
                    plt.legend(loc="best")
                    st.pyplot(plt)
                    st.error('Hmm... It seems like overestimating a model.')
                    st.info('How about using PR Curve?')
                    
                    if st.checkbox('PR Curve'):
                        PR_CURVE(model, 'Logistic Regression')
                        st.success('''It's fine now.''')
                        st.info('For better evaluation, we will use PR Curve instead of ROC Curve in this web application.')
                    
        elif navigate == '3. Classification with SMOTE':
            # 1
            st.markdown('# Classification with SMOTE')
            image = 'SMOTE.jpg'
            st.image(image, caption='@Bhavesh Bhatt', use_column_width=True)
            st.markdown('## SMOTE Algorithm')
            st.markdown('### Synthetic Minority Oversampling Technique')
            
            if st.button('For more info of SMOTE'):
                webbrowser.open_new_tab('https://arxiv.org/pdf/1106.1813.pdf')
                
            smote = SMOTE(random_state=0)
            smote_X_train, smote_y_train = smote.fit_sample(X_train, y_train['BINARY'])
            smote_y_train = smote_y_train.astype('int')
            st.markdown('#### Summary')
            st.write(smote_y_train.value_counts())
            neg, pos = np.bincount(smote_y_train)
            total = neg + pos
            st.text('Exoplanet stars are {:.2f}% of total'.format(100*pos/total))
            smote_y_train = smote_y_train.values.tolist()
            st.success('Cool! Now the data is balanced.')

            code = '''from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
smote_X_train, smote_y_train = smote.fit_sample(X_train, y_train['BINARY'])
smote_y_train = smote_y_train.astype('int')
smote_y_train = smote_y_train.values.tolist()'''
            st.markdown('#### Code')
            st.code(code, language='python')
            
            # 2
            st.markdown('## Support Vector Machine')
            if st.checkbox('Polynomial SVM'):
                model = joblib.load('Poly_SVM_SMOTE.model')
                result_binary(model)
                poly_precision, poly_recall, poly_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                poly_pr_auc = auc(poly_recall, poly_precision)
    
            # 3
            if st.checkbox('RBF SVM'):
                model = joblib.load('RBF_SVM_SMOTE.model')
                result_binary(model)
                rbf_precision, rbf_recall, rbf_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                rbf_pr_auc = auc(rbf_recall, rbf_precision)
                
            # 4
            if st.checkbox('Sigmoid SVM'):
                model = joblib.load('Sigmoid_SVM_SMOTE.model')
                result_binary(model)
                sigmoid_precision, sigmoid_recall, sigmoid_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                sigmoid_pr_auc = auc(sigmoid_recall, sigmoid_precision)
                
            # 5
            if st.checkbox('PR Curve of SVM with SMOTE'):
                plt.figure()
                plt.plot(poly_recall, poly_precision, label='Polynomial SVM with SMOTE(area = {:.4f})'.format(poly_pr_auc))
                plt.plot(rbf_recall, rbf_precision, label='RBF SVM with SMOTE (area = {:.4f})'.format(rbf_pr_auc))
                plt.plot(sigmoid_recall, sigmoid_precision, label='Sigmoid SVM with SMOTE (area = {:.4f})'.format(sigmoid_pr_auc))
                plt.plot([1, 0], [0, 0],'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([-0.1, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall')
                plt.legend(loc='best')
                st.pyplot(plt)
                st.info('Seems like RBF SVM has the highest performance.')
            st.warning('Did you checked all the Kernel? Then check above for PR Curve.')
   
            # 6
            st.markdown('## Logistic Regression')
            if st.checkbox('Logistic Regression Result'):
                model = joblib.load('logit_SMOTE.model')
                result_binary(model)
                if st.checkbox('PR Curve of Logistic Regression with SMOTE'):
                    PR_CURVE(model, 'Logistic Regression with SMOTE')
                    st.warning('Umm...Not that Good.')
                    
            # 7
            st.markdown('## Decision Tree')
            if st.checkbox('Decision Tree Result'):
                model = joblib.load('DT_SMOTE.model')
                result_binary(model)
                if st.checkbox('PR Curve of Decision Tree with SMOTE'):
                    PR_CURVE(model, 'Decision Tree with SMOTE')
                    st.success('Oh, It looks nice! Much Better than RBF SVM.')
            
            # 8
            st.markdown('## Random Forest')
            if st.checkbox('Random Forest Result'):
                model = joblib.load('Random_Forest_SMOTE.model')
                result_binary(model)
                st.error('Result is very, very Bad. Not worth drawing PR Curve.')
            
            # 9
            st.markdown('## Adaptive Boosting')
            if st.checkbox('AdaBoost Result'):
                model = joblib.load('AdaBoost_SMOTE.model')
                result_binary(model)
                if st.checkbox('PR Curve of AdaBoost with SMOTE'):
                    PR_CURVE(model, 'PR Curve of AdaBoost with SMOTE')
                    st.warning('Not that Bad, but Decision Tree is Better.')
                    
            # 10
            if st.button('RANK : Model with SMOTE'):
                st.markdown('#### PR-AUC')
                st.success('Decision Tree (0.2535) > Adaptive Boosting (0.0965) > RBF SVM (0.0806)')
                st.markdown('#### Confusion Matrix')
                st.info('Adaptive Boosting > RBF SVM > Decision Tree')
        
        elif navigate == '4. Classification with ADASYN':
            # 1
            st.markdown('# Classification with ADASYN')
            image = 'ADASYN.jpeg'
            st.image(image, caption='@Towards Data Science', use_column_width=True)
            st.markdown('## ADASYN Algorithm')
            st.markdown('### Adaptive Synthetic Sampling Approach')
            
            if st.button('For more info of ADASYN'):
                webbrowser.open_new_tab('https://ieeexplore.ieee.org/document/4633969')
            
            adasyn = ADASYN(random_state=0)
            adasyn_X_train, adasyn_y_train = adasyn.fit_resample(X_train, y_train['BINARY'])
            adasyn_y_train = adasyn_y_train.astype('int')
            st.markdown('#### Summary')
            st.write(adasyn_y_train.value_counts())
            neg, pos = np.bincount(adasyn_y_train)
            total = neg + pos
            st.text('\nExoplanet stars are {:.2f}% of total'.format(100*pos/total))
            adasyn_y_train = adasyn_y_train.values.tolist()
            st.success('Nice. ADASYN also works well!')

            code = '''from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=0)
adasyn_X_train, adasyn_y_train = adasyn.fit_resample(X_train, y_train['BINARY'])
adasyn_y_train = adasyn_y_train.astype('int')
adasyn_y_train = adasyn_y_train.values.tolist()'''
            st.markdown('#### Code')
            st.code(code, language='python')
            
            # 2
            st.markdown('## Support Vector Machine')
            if st.checkbox('Polynomial SVM'):
                model = joblib.load('Poly_SVM_ADASYN.model')
                result_binary(model)
                poly_precision, poly_recall, poly_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                poly_pr_auc = auc(poly_recall, poly_precision)
    
            # 3
            if st.checkbox('RBF SVM'):
                model = joblib.load('RBF_SVM_ADASYN.model')
                result_binary(model)
                rbf_precision, rbf_recall, rbf_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                rbf_pr_auc = auc(rbf_recall, rbf_precision)
                
            # 4
            if st.checkbox('Sigmoid SVM'):
                model = joblib.load('Sigmoid_SVM_ADASYN.model')
                result_binary(model)
                sigmoid_precision, sigmoid_recall, sigmoid_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                sigmoid_pr_auc = auc(sigmoid_recall, sigmoid_precision)
                st.info("Wow, Although the accuracy is slightly lower, Sigmoid SVM classified 3 Exoplanet Stars correctly.")
                
            # 5
            if st.checkbox('PR Curve of SVM with ADASYN'):
                plt.figure()
                plt.plot(poly_recall, poly_precision, label='Polynomial SVM with ADASYN(area = {:.4f})'.format(poly_pr_auc))
                plt.plot(rbf_recall, rbf_precision, label='RBF SVM with ADASYN (area = {:.4f})'.format(rbf_pr_auc))
                plt.plot(sigmoid_recall, sigmoid_precision, label='Sigmoid SVM with ADASYN (area = {:.4f})'.format(sigmoid_pr_auc))
                plt.plot([1, 0], [0, 0],'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([-0.1, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall')
                plt.legend(loc='best')
                st.pyplot(plt)
                st.info('RBF SVM has the highest performance of 3, same as SMOTE.')
            st.warning('Did you checked all the Kernel? Then check above for PR Curve.')
                
            # 6
            st.markdown('## Logistic Regression')
            if st.checkbox('Logistic Regression Result'):
                model = joblib.load('logit_ADASYN.model')
                result_binary(model)
                if st.checkbox('PR Curve of Logistic Regression with ADASYN'):
                    PR_CURVE(model, 'Logistic Regression with ADASYN')
                    st.warning("There's no improvement compared to SMOTE.")
                    
            # 7
            st.markdown('## Decision Tree')
            if st.checkbox('Decision Tree Result'):
                model = joblib.load('DT_ADASYN.model')
                result_binary(model)
                if st.checkbox('PR Curve of Decision Tree with ADASYN'):
                    PR_CURVE(model, 'Decision Tree with ADASYN')
                    st.error("Same as No-Skill Classification...")
            
            # 8
            st.markdown('## Random Forest')
            if st.checkbox('Random Forest Result'):
                model = joblib.load('Random_Forest_ADASYN.model')
                result_binary(model)
                st.error('Result is awful. Same as SMOTE!')
            
            # 9
            st.markdown('## Adaptive Boosting')
            if st.checkbox('AdaBoost Result'):
                model = joblib.load('AdaBoost_ADASYN.model')
                result_binary(model)
                if st.checkbox('PR Curve of AdaBoost with ADASYN'):
                    PR_CURVE(model, 'PR Curve of AdaBoost with ADASYN')
                    st.error('Poor Accuracy, Bad PR-AUC.')
                    
            # 10
            if st.button('RANK : Model with ADASYN'):
                st.markdown('#### PR-AUC')
                st.success('RBF SVM (0.2278) > Logistic Regression (0.0361) > Adaptive Boosting (0.0260)')
                st.markdown('#### Confusion Matrix')
                st.info('Sigmoid SVM > RBF SVM > Decision Tree')
                st.error('ADASYN seems to bring bad classification result than SMOTE.')
                
        elif navigate == '5. Cost-Sensitive Learning':
            # 1
            st.markdown('# Cost-Sensitive Learning')
            image = 'Cost_Sensitive_Learning.jpg'
            st.image(image, caption='@Machine Learning Mastery', use_column_width=True)
            st.info('''Can you see ships in the picture? They are the same in terms of ships, but 
                    their sizes are quiet different. That's the key point. We should consider SIZE.
                    This concept lead us to Cost-Sensitive Learning.''')

            st.markdown('## Class Weight')
            st.warning('It can be tuned whenever you want. In this web application, I just took simple method. Check the code below.')
            neg, pos = np.bincount(y_train['LABEL']-1)
            total = neg + pos
            weight_0 = (1/neg)*(total)/5.0
            weight_1 = (1/pos)*(total)/5.0
            class_weight = {0:weight_0, 1:weight_1}
            st.text('Weight for class 0 : {:.2f}\nWeight for class 1 : {:.2f}'.format(weight_0, weight_1))
            st.markdown('#### Code')

            code = '''neg, pos = np.bincount(y_train['LABEL']-1)
total = neg + pos
weight_0 = (1/neg)*(total)/5.0
weight_1 = (1/pos)*(total)/5.0
class_weight = {0:weight_0, 1:weight_1}'''
            st.code(code, language='python')

            # 2
            st.markdown('## Support Vector Machine')
            if st.checkbox('Polynomial SVM'):
                model = joblib.load('Poly_SVM_Cost.model')
                result_binary(model)
                poly_precision, poly_recall, poly_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                poly_pr_auc = auc(poly_recall, poly_precision)
            # 3
            if st.checkbox('RBF SVM'):
                model = joblib.load('RBF_SVM_Cost.model')
                result_binary(model)
                rbf_precision, rbf_recall, rbf_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                rbf_pr_auc = auc(rbf_recall, rbf_precision)
                
            # 4
            if st.checkbox('Sigmoid SVM'):
                model = joblib.load('Sigmoid_SVM_Cost.model')
                result_binary(model)
                sigmoid_precision, sigmoid_recall, sigmoid_thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
                sigmoid_pr_auc = auc(sigmoid_recall, sigmoid_precision)
            # 5
            if st.checkbox('PR Curve of SVM with Cost-Sensitive Learning'):
                plt.figure()
                plt.plot(poly_recall, poly_precision, label='Polynomial SVM with Cost-Sensitive Learning (area = {:.4f})'.format(poly_pr_auc))
                plt.plot(rbf_recall, rbf_precision, label='RBF SVM with Cost-Sensitive Learning (area = {:.4f})'.format(rbf_pr_auc))
                plt.plot(sigmoid_recall, sigmoid_precision, label='Sigmoid SVM with Cost-Sensitive Learning (area = {:.4f})'.format(sigmoid_pr_auc))
                plt.plot([1, 0], [0, 0],'r--')
                plt.xlim([0.0, 1.0])
                plt.ylim([-0.1, 1.05])
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall')
                plt.legend(loc='best')
                st.pyplot(plt)
                st.warning('RBF SVM has the highest performance, but not to say great.')
            st.warning('Did you checked all the Kernel? Then check above for PR Curve.')

            # 6
            st.markdown('## Logistic Regression')
            if st.checkbox('Logistic Regression Result'):
                model = joblib.load('logit_Cost.model')
                result_binary(model)
                st.error('Not worth drawing PR Curve...')
                    
            # 7
            st.markdown('## Decision Tree')
            if st.checkbox('Decision Tree Result'):
                model = joblib.load('DT_Cost.model')
                result_binary(model)
                if st.checkbox('PR Curve of Decision Tree with Cost-Sensitive Learning'):
                    PR_CURVE(model, 'Decision Tree with Cost-Sensitive Learning')
                    st.info('Higher performance than RBF SVM, but not as Decision Tree with SMOTE.')
            
            # 8
            st.markdown('## Random Forest')
            if st.checkbox('Random Forest Result'):
                model = joblib.load('Random_Forest_Cost.model')
                result_binary(model)
                st.info('Only one, but finally succeeded in classifying Exoplanet stars correctly.')
                if st.checkbox('PR Curve of Random Forest with Cost-Sensitive Learning'):
                    PR_CURVE(model, 'Random Forest with Cost-Sensitive Learning')
                    st.warning('But PR Curve looks not that good...')

            # 9
            st.markdown('## Adaptive Boosting')
            st.error('Unfortunately, Adaptive Boosting does not support Cost-Sensitive Learning.')

            # 10
            if st.button('RANK : Model with Cost-Sensitive Learning'):
                st.markdown('#### PR-AUC')
                st.success('Decision Tree (0.1307) > RBF SVM (0.0580) > Random Forest (0.0477)')
                st.markdown('#### Confusion Matrix')
                st.info('Decision Tree > RBF SVM > Random Forest')
                st.error('Cost-Sensitive Learning seems to bring bad classification result than SMOTE.')
                st.error('Maybe class weight needs tuning.')

        elif navigate == '6. Conclusion':
            # 1
            st.markdown('# Conclusion')
            st.markdown("## PR Curve of the 3 Models with the highest PR-AUC")
            first = joblib.load('DT_SMOTE.model')
            second = joblib.load('RBF_SVM_ADASYN.model')
            third = joblib.load('DT_Cost.model')
            fisrt_pred = first.predict(X_test)
            second_pred = second.predict(X_test)
            third_pred = third.predict(X_test)

            # 2
            precision_1, recall_1, thresholds_1 = precision_recall_curve(y_test['BINARY'], first.predict_proba(X_test)[:,1])
            precision_2, recall_2, thresholds_2 = precision_recall_curve(y_test['BINARY'], second.predict_proba(X_test)[:,1])
            precision_3, recall_3, thresholds_3 = precision_recall_curve(y_test['BINARY'], third.predict_proba(X_test)[:,1])
            pr_auc_1 = auc(recall_1, precision_1)
            pr_auc_2 = auc(recall_2, precision_2)
            pr_auc_3 = auc(recall_3, precision_3)

            # 3
            plt.plot(recall_1, precision_1, label='Decision Tree with SMOTE (area = {:.4f})'.format(pr_auc_1))
            plt.plot(recall_2, precision_2, label='RBF SVM with ADASYN (area = {:.4f})'.format(pr_auc_2))
            plt.plot(recall_3, precision_3, label='Decisioni Tree with Cost-Sensitive Learning (area = {:.4f})'.format(pr_auc_3))
            plt.plot([1, 0], [0, 0],'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([-0.1, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall')
            plt.legend(bbox_to_anchor=(0.975, -0.15))
            st.pyplot(plt)

            # 4
            st.markdown('## Notable Point')
            st.warning('The most ideal PR-AUC is 1, same as AUC. Unfortunately, There was no model with PR-AUC close to 1.')
            st.warning('''But, We can see Decision Tree has consistently ranked First and Second. This suggests that the performance of the model could be
            improved if the imbalance in train data could be better treated.''')
            st.info('''### In other words, we can build better models through Hyperparameter tuning when we balance the data.''')
            st.success('### Through this web application, We hope you could know how to deal with imbalanced data.')
            st.markdown('## Thank you for reading this web application.')

def result_label(model):
    y_pred = model.predict(X_test)
    st.write(confusion_matrix(y_test['LABEL'], y_pred))
    st.text(classification_report(y_test['LABEL'], y_pred))
                
def result_binary(model):
    y_pred = model.predict(X_test)
    st.write(confusion_matrix(y_test['BINARY'], y_pred))
    st.text(classification_report(y_test['BINARY'], y_pred))

def PR_CURVE(model, name):
    precision, recall, thresholds = precision_recall_curve(y_test['BINARY'], model.predict_proba(X_test)[:,1])
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label='{} (area = {:.4f})'.format(name, pr_auc))
    plt.plot([1, 0], [0, 0],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="best")
    st.pyplot(plt)
    
if __name__ == '__main__':
    exo_train = pd.read_csv('exoTrain_preprocessed.csv')
    exo_test = pd.read_csv('exoTest_preprocessed.csv')
    
    X_train = exo_train.iloc[:, 2:]
    y_train = exo_train.iloc[:, :2]
    X_test = exo_test.iloc[:, 2:]
    y_test = exo_test.iloc[:, :2]
    
    main()