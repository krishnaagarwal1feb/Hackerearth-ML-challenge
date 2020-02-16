# Importing the libraries
import pandas as pd
import seaborn as sns
#%%
def plotcorr(df):
    ax = sns.heatmap(
            df, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
        )
    ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );

#%% Importing the dataset

dataset = pd.read_csv('/home/krish/Desktop/flight/train.csv')
X_train = dataset.iloc[:, 1:]
X_test = pd.read_csv('/home/krish/Desktop/flight/test.csv')
y_train = dataset.iloc[:, 0]
y_train = y_train.to_frame()

#%%

data = dataset.drop(['Severity'],axis=1)
df1 = data.corr()
plotcorr(df1)

#%%how to predict for two targets -- normally fit it to model 
X_train.drop(['Accident_ID'],axis=1,inplace = True)
X_test.drop(['Accident_ID'],axis=1,inplace = True)

#%% Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#%% implementing train test and apply other algos 
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

#%% trying diff models
from sklearn.ensemble import GradientBoostingClassifier
grad_boost = GradientBoostingClassifier(max_depth=7,n_estimators=1000,subsample=0.9,
                                        min_samples_split=10)
grad_boost.fit(X_train,y_train)
y_pred1_grad  = grad_boost.predict(X_test)
y_p1_grad = pd.DataFrame(y_pred1_grad,columns=['Severity'])

#%%evaluate
def evaluate(y_t1,y_p1):
    from sklearn.metrics import f1_score,accuracy_score
    acc = accuracy_score(y_t1,y_p1)
    f1 = f1_score(y_t1,y_p1,average='weighted')
    print(acc)
    print(f1)
#%%
evaluate(y_test1,y_p1_grad)
"""
opt rf gives 94.6 score 
opt grad boost 95.9

"""
#%% submission 
res = y_p1_grad
res['Accident_ID'] = (pd.read_csv('/home/krish/Desktop/flight/test.csv'))['Accident_ID']
res.set_index("Accident_ID", inplace = True)
#%%
res.to_csv('grad_boost.csv')
