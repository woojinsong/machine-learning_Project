import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import warnings
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import category_encoders as ce
from lightgbm import plot_importance
warnings.filterwarnings('ignore')


#데이터 로드
df=pd.read_excel('./data.xlsx')

check_categori=[]
check_categori_2=[]
check_categori_over2=[]

#카테고리 변수 찾기(2개,3개이상)
for i in df:
    if 2<=len(df[i].value_counts())<=12:
        check_categori.append([i,len(df[i].value_counts())])
        if len(df[i].value_counts())==2:
            check_categori_2.append([i,len(df[i].value_counts())])
        else:
            check_categori_over2.append([i,len(df[i].value_counts())])
            
#성별 변수 라벨인코딩 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(df['gender'])
labels = encoder.transform(df['gender'])
df['gender'] = labels

#3개 피처 이상 : 타겟 인코딩
for i in check_categori_over2:
    encoder = ce.target_encoder.TargetEncoder(cols=[i[0]])
    encoder.fit(df[i[0]], df['voted'])
    df[i[0]]=encoder.transform(df[i[0]])




#lgbc 적합
y=df['voted']
df=df.drop(['voted'],axis=1)
train_x, test_x, train_y, test_y = train_test_split(df, y, test_size = 0.2, random_state = 42) 


#최적 파라미터
clf = LGBMClassifier(n_estimators=138, max_depth=7,random_state=0,n_jobs=-1)
clf.fit(train_x,train_y)
predict = clf.predict(test_x)

print(confusion_matrix(list(test_y),predict))
print(classification_report(list(test_y),predict))


#중요변수
fig,ax=plt.subplots(figsize=(20,20))
plot_importance(clf,ax=ax)
plt.show()




#lgbc 하이퍼 파라미터 튜닝 
# params = { 'n_estimators' : list(range(2,100)),
#            'max_depth' : list(range(1,20,2))
#             }

# y=df['voted']
# df=df.drop(['voted'],axis=1)
# train_x, test_x, train_y, test_y = train_test_split(df, y, test_size = 0.2, random_state = 42) 

# clf = LGBMClassifier(random_state = 0, n_jobs = -1)
# grid_cv = GridSearchCV(clf, param_grid = params, cv = 3, n_jobs = -1)
# grid_cv.fit(train_x, train_y)

# print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
# print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

