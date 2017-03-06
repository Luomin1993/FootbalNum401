import pandas as pd
import numpy as np

"""
cashflowData = df[df.code == stockID]
"""

def get_df(wayOfCSV):
    return pd.read_csv(wayOfCSV)

def get_col(df,colName):
    pass

def handle_NAN(df):
    pass 

def get_TeamsNames(df):
    TeamsNames = []
    for line in range(20):
        if df['HomeTeam'][line] not in TeamsNames:
           TeamsNames.append(df['HomeTeam'][line])
        if df['AwayTeam'][line] not in TeamsNames:
           TeamsNames.append(df['AwayTeam'][line])
    return TeamsNames

def get_TeamMatches(df,TeamName):
    Team_Match_results = []
    for line in range(0,df.shape[0]):
        Match = []
        if df['HomeTeam'][line] == TeamName:
           Match.append(df['B365A'][line]/df['B365H'][line])          # odd Rate
           Match.append(3)                                            # HA signal
           Match.append(df['FTHG'][line])                             # Goal
           Match.append(df['FTAG'][line])                             # Goaled
           #if df['FTR'][line] == 'D':
           Team_Match_results.append(Match)      
        
        if df['AwayTeam'][line] == TeamName:
           Match.append(df['B365H'][line]/df['B365A'][line])
           Match.append(1)
           Match.append(df['FTAG'][line])
           Match.append(df['FTHG'][line])
           Team_Match_results.append(Match)
    return Team_Match_results          

def get_Fortune(Res_1,Res_2,Res_3):
    return 1 + get_basicFor(Res_1)*3 + get_basicFor(Res_2)*2 +get_basicFor(Res_3)

def get_basicFor(Res_i):
    if Res_i>0:
       return 1
    if Res_i == 0:
       return 0.5
    else:
       return 0      

def get_perGoal(pG,No):
    return float(sum(pG[0:No]))/No

def get_perGoaled(pGd,No):
    return float(sum(pGd[0:No]))/No

def get_PreRes_Series(Team_Match_results):
    PreRes_Series = []
    pG  = np.array(Team_Match_results).T[2]
    pGd = np.array(Team_Match_results).T[3]
    for No in range(3,len(Team_Match_results)):
        PreRes = []
        PreRes.append(get_perGoal(pG,No))
        PreRes.append(get_perGoaled(pGd,No))
        PreRes.append(get_Fortune(pGd[No-1]-pG[No-1],pGd[No-2]-pG[No-2],pGd[No-3]-pG[No-3]))
        PreRes.append(Team_Match_results[No][0])
        PreRes.append(Team_Match_results[No][1])
        PreRes.append(Team_Match_results[No][2])
        PreRes.append(Team_Match_results[No][3])
        PreRes_Series.append(PreRes)
    return PreRes_Series    

def get_PreG_and_resG(PreRes_Series):
    PreG = []
    ResG = np.array(PreRes_Series).T[5]
    PreG.append(np.array(PreRes_Series).T[0])
    PreG.append(np.array(PreRes_Series).T[2])
    PreG.append(np.array(PreRes_Series).T[3])
    PreG.append(np.array(PreRes_Series).T[4])
    return (np.array(PreG).T,ResG)

def get_PreGd_and_resGd(PreRes_Series):
    PreGd = []
    ResGd = np.array(PreRes_Series).T[6]
    PreGd.append(np.array(PreRes_Series).T[1])
    PreGd.append(np.array(PreRes_Series).T[2])
    PreGd.append(np.array(PreRes_Series).T[3])
    PreGd.append(np.array(PreRes_Series).T[4])
    return (np.array(PreGd).T,ResGd)    

def get_TeamPredicter(TeamMatches):           
    pass

def get_TeamData_forTrain(TeamName,wayOfCSV):
    df = get_df(wayOfCSV)
    Team_Match_results = get_TeamMatches(df,TeamName)
    PreRes_Series = get_PreRes_Series(Team_Match_results)
    (PreG,ResG)   = get_PreG_and_resG(PreRes_Series)
    (PreGd,ResGd) = get_PreGd_and_resGd(PreRes_Series)
    return (PreG,ResG,PreGd,ResGd)

def make_Team_Predictor_Lasso(TeamName,wayOfCSV,Name):
    (PreG,ResG,PreGd,ResGd) = get_TeamData_forTrain(TeamName,wayOfCSV)
    from sklearn import linear_model
    from sklearn.externals import joblib
    regG  = linear_model.Lasso(alpha = 0.1)
    regGd = linear_model.Lasso(alpha = 0.1)
    regG  = regG.fit(PreG,ResG)
    regGd = regGd.fit(PreGd,ResGd)
    import os
    #-----------------Notice here!!!!!!!!!!!!!!can't mkdir Repeatly!!!!!!!!!!!!-----------------------------
    #delete the previous One before reCal
    os.mkdir(r''+Name)          
    joblib.dump(regG,Name + '\Goal_pre.pkl')
    joblib.dump(regGd,Name + '\Goaled_pre.pkl')
    return True

def test_Lasso():
    TeamName = 'Bastia'
    wayOfCSV = 'C:\Users\Lenovo\Downloads\F1.csv'
    Name     = 'Bastia'
    return make_Team_Predictor_Lasso(TeamName,wayOfCSV,Name)


if __name__ == '__main__':
    print test_Lasso()   