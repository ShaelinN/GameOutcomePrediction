import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")

data = pd.read_csv("teams_season.csv", sep=',')

def train():

    wins = (data.loc[:,"won"]*100) / (data.loc[:,"lost"] +data.loc[:,"won"])

    X_train, X_test,Y_train, Y_test = train_test_split(data.iloc[:,3:33], wins, test_size=0.3, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=28, svd_solver="full")
    pca.fit_transform(X_train)


    X_train = pca.transform(X_train)
    X_test =pca.transform(X_test)


    rf = RandomForestRegressor(n_estimators=300,random_state=0)
    lr = LinearRegression()

    svr = SVR(C=15, epsilon=2, kernel="poly")


    estimators = [('ridge', RidgeCV()),
                    ('lasso', LassoCV(random_state=42)),
                    ('knr', KNeighborsRegressor(n_neighbors=40,metric = 'euclidean'))]

    

    final_estimator = GradientBoostingRegressor(
    n_estimators = 40, subsample = 0.99, min_samples_leaf = 30, max_features =0.01,
    random_state = 42)


    ensemble = StackingRegressor(
    estimators = estimators,
    final_estimator = final_estimator)

    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=25), n_estimators=100, random_state=0)

    lr.fit(X_train,Y_train)
    rf.fit(X_train,Y_train)
    svr.fit(X_train,Y_train)
    ensemble.fit(X_train,Y_train)
    ada.fit(X_train,Y_train)
    print("-------------------------------------------------------")
    y_pred = lr.predict(X_test)
    print("Linear Regressor : ")
    print("     Score - ",lr.score(X_test,Y_test)*100)
    print("     Root Mean Squared Error - ",np.sqrt(mean_squared_error(Y_test,y_pred)))
    print("     Mean Absolute Error - ", mean_absolute_error(Y_test,y_pred))
    y_pred = rf.predict(X_test)
    print("Random Forest Regressor : ")
    print("     Score - ",rf.score(X_test,Y_test)*100)
    print("     Root Mean Squared Error - ",np.sqrt(mean_squared_error(Y_test,y_pred)))
    print("     Mean Absolute Error - ", mean_absolute_error(Y_test,y_pred))
    y_pred = svr.predict(X_test)
    print("Support Vector Regressor : ")
    print("     Score - ",svr.score(X_test,Y_test)*100)
    print("     Root Mean Squared Error - ",np.sqrt(mean_squared_error(Y_test,y_pred)))
    print("     Mean Absolute Error - ", mean_absolute_error(Y_test,y_pred))
    y_pred = ensemble.predict(X_test)
    print("Ensembled Stacking Regressor with Gradient Boosting")
    print("     Score - ",ensemble.score(X_test,Y_test)*100)
    print("     Root Mean Squared Error - ",np.sqrt(mean_squared_error(Y_test,y_pred)))
    print("     Mean Absolute Error - ", mean_absolute_error(Y_test,y_pred))
    y_pred = ada.predict(X_test)
    print("AdaBooster Regressor : ")
    print("     Score - ",ada.score(X_test,Y_test)*100)
    print("     Root Mean Squared Error - ",np.sqrt(mean_squared_error(Y_test,y_pred)))
    print("     Mean Absolute Error - ", mean_absolute_error(Y_test,y_pred))
    print("-------------------------------------------------------")

    return lr, pca


Year_2003 = data.iloc[1128:1157,:33]
Year_2004 = data.iloc[1157:, :33]
model , pca= train()


for i in range(1,len(Year_2003)):
    X = pca.transform(Year_2003.iloc[i-1:i, 3:33])
    TeamX = model.predict(X)
    TeamX_Name = Year_2003.iloc[i-1:i, 0:1]
    for j in range(i+1, len(Year_2003)):
        Y = pca.transform(Year_2003.iloc[j-1:j,3:33])
        TeamY = model.predict(Y)
        TeamY_Name = Year_2003.iloc[j-1:j, 0:1]
        if (TeamX > TeamY):
            print(TeamX_Name, " beats ", TeamY_Name)
        elif (TeamY > TeamX):
            print(TeamY_Name, " beats ", TeamX_Name)
        else:
            print(TeamX_Name," DRAW ", TeamY_Name)
    print()
    print()

for i in range(1,len(Year_2004)):
    X = pca.transform(Year_2004.iloc[i-1:i, 3:33])
    TeamX = model.predict(X)
    TeamX_Name = Year_2004.iloc[i-1:i, 0:1]
    for j in range(i+1, len(Year_2004)):
        Y = pca.transform(Year_2004.iloc[j-1:j,3:33])
        TeamY = model.predict(Y)
        TeamY_Name = Year_2004.iloc[j-1:j, 0:1]
        if (TeamX > TeamY):
            print(TeamX_Name, " beats ", TeamY_Name)
        elif (TeamY > TeamX):
            print(TeamY_Name, " beats ", TeamX_Name)
        else:
            print(TeamX_Name," DRAW ", TeamY_Name)
    print()
    print()