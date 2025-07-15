import pickle
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#========================================you need to change!!!==============================================
tw_junction_data = pd.read_excel(r"xx/xx/Data/TrainingSet/TrainingData.xlsx")  ##!!!change your file path
#===========================================================================================================

tw_data = pd.DataFrame(tw_junction_data)
tw_data=tw_data.iloc[:,1:]

tw_data['Class'] = tw_data['Class'].replace({'Coaxial': 1, 'Non-coaxial': 0})
# print(tw_data.head())

x = tw_data.drop('Class',axis=1)
y = tw_data['Class']

def optimize_rf(n_estimators, max_depth, min_samples_split,min_samples_leaf,max_features,class_weight):
    model = RandomForestClassifier(n_estimators=int(n_estimators),
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf), 
                                   max_features=float(max_features),
                                   class_weight={0: 1, 1: class_weight}, 
                                   random_state=42)
    return cross_val_score(model, x,  y, cv=10, scoring='accuracy').mean()

pbounds_rf = {'n_estimators': (50, 800),
              'max_depth': (5, 50),
              'min_samples_split': (2, 20),
             'min_samples_leaf': (1,10),
             'max_features':(0.1,1),
             'class_weight': (1, 10)}

optimizer_rf = BayesianOptimization(f=optimize_rf, pbounds=pbounds_rf, random_state=1)

optimizer_rf.maximize(init_points=5, n_iter=10)

print("Random Forest:", optimizer_rf.max)

best_rf_model = RandomForestClassifier(max_depth=24,min_samples_split=18,n_estimators=562,max_features=0.1,min_samples_leaf=1,class_weight={0: 1, 1: 1.09})
best_rf_model.fit(x,y)


# save model to file
with open('gCoSRNA_model.pkl', 'wb') as f:
    pickle.dump(best_rf_model, f)