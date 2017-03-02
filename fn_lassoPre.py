import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn import linear_model
from sklearn.externals import joblib  

#public methods
def CleanDataT(AT):
	for i in range(len(AT)):
		AT[i][0] /= 10000
		AT[i][1] = 7 - AT[i][1]
		AT[i][2] = 20 - AT[i][2]
	return AT	


def CleanDataR(AR):
	for i in range(len(AR)):
		AR[i][1] /= 10
	return AR

def withoutTG(AT):
    AT = np.delete(AT , 3, axis = 1)
    return AT

def withoutTGDim1(AT):
	AT = np.delete(AT, 3)
	return AT  

def makeRes(HomeRes,AwayRes,HPerG,APerG):
	#[[  1.65950427   2.22327135  12.22333476   4.77497455]]
    #[[  0.62019478   8.67225777  12.33880961   3.1350144 ]]
    HomeRes[0] += HPerG
    AwayRes[0] += APerG

    Hcr,Acr = HomeRes[1],AwayRes[1]
    HomeRes[1] = (Hcr/(Hcr+Acr))*100
    AwayRes[1] = (Acr/(Hcr+Acr))*100

    HomeRes[0] = int(round(HomeRes[0]))
    AwayRes[0] = int(round(AwayRes[0]))
    HomeRes[2] = int(round(HomeRes[2]))
    HomeRes[3] = int(round(HomeRes[3]))
    AwayRes[2] = int(round(AwayRes[2]))
    AwayRes[3] = int(round(AwayRes[3]))

    return (HomeRes,AwayRes)

def makeResA(HomeRes,AwayRes):
	#[[  1.65950427   2.22327135  12.22333476   4.77497455]]
    #[[  0.62019478   8.67225777  12.33880961   3.1350144 ]]
    Hcr,Acr = HomeRes[1],AwayRes[1]
    HomeRes[1] = (Hcr/(Hcr+Acr))*100
    AwayRes[1] = (Acr/(Hcr+Acr))*100

    # HomeRes[0] = int(round(HomeRes[0]))
    # AwayRes[0] = int(round(AwayRes[0]))
    # HomeRes[2] = int(round(HomeRes[2]))
    # HomeRes[3] = int(round(HomeRes[3]))
    # AwayRes[2] = int(round(AwayRes[2]))
    # AwayRes[3] = int(round(AwayRes[3]))

    return (HomeRes,AwayRes)    

pd_IDData =pd.read_csv('data/pd_IDData.csv')
# #test examples
# #2016.12.20 **************Everton VS Liverpool*******************

Eve = np.array([2.4825,2,12,21,3])
#Liv = np.array([3.7970,2,18,40,1])
#HPerG = pd_IDData.loc['Eve','perG']
#APerG = pd_IDData['perG']['Liv']
# #2016-03-20 **************SouthApton VS Liverpool****************
#[perGoals,Fortune,Rank,Scores,HoA]

Sou = np.array([1.7975,2,13,38,3])
#Liv = np.array([3.6725,0,12,43,1])

# #2016-12-04 **************Burmouth VS Liverpool****************

Bur = np.array([1.2175,5,8,15,3])
#Liv = np.array([3.6725,1,18,32,1])

# #2017-01-16 **************Manchester United VS Liverpool****************

# ****************************** app *********************
Home_ = np.array([2.00,0,17,48,3])
Away_ = np.array([1.40,2,12,35,1])

if __name__ == '__main__':
	Acc_en  = joblib.load('Acc_Lasso.pkl')  
	Base_en = joblib.load('Base_Lasso.pkl')

	Home = Home_
	Away = Away_

	HomeRes = Acc_en.predict(withoutTGDim1(Home)) + Base_en.predict(Home - Away)
	AwayRes = Acc_en.predict(withoutTGDim1(Away)) + Base_en.predict(Away - Home)

	print makeResA(HomeRes[0],AwayRes[0])