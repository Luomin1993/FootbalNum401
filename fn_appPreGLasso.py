import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib 
import fn_featuresGet as fnf

#--------------------------app Data-------------------------
TeamHome = 'Sunderland'
TeamAway = 'Man City'
NameHome = 'Sunderland'
NameAway = 'ManCity'

#       [  perGameGoal, Fortune, OddRate, HorA]
#       [perGameGoaled, Fortune, OddRate, HorA]
Home_G  = np.array([1.55,6,1.29/11,3])
Away_G  = np.array([0.89,1,11/1.29,1])

Home_Gd = np.array([1.55,6,1.29/11,3])
Away_Gd = np.array([1.85,1,11/1.29,1])

wayOfCSV = 'D:\FootballNumA\Data\E0.csv'

if __name__ == '__main__':
    #fnf.make_Team_Predictor_Lasso(TeamHome,wayOfCSV,NameHome)
    #fnf.make_Team_Predictor_Lasso(TeamAway,wayOfCSV,NameAway)

    preG_Home  = joblib.load(NameHome + '/Goal_pre.pkl')  
    preGd_Home = joblib.load(NameHome + '/Goaled_pre.pkl')

    preG_Away  = joblib.load(NameAway + '/Goal_pre.pkl')  
    preGd_Away = joblib.load(NameAway + '/Goaled_pre.pkl')

    G_h = preG_Home.predict(Home_G)[0]
    L_h = preGd_Home.predict(Home_G)[0]
    G_a = preG_Away.predict(Away_G)[0]
    L_a = preGd_Away.predict(Away_G)[0]


    print NameHome + ' Goals:' + str(G_h) + '\n' 
    print NameHome + ' Losts:' + str(L_h) + '\n'
    print NameAway + ' Goals:' + str(G_a) + '\n'        
    print NameAway + ' Losts:' + str(L_a) + '\n'

    print '------------------per res-------------------'+'\n'
    print str((G_h+L_a)/2) + ':' + str((G_a+L_h)/2)            