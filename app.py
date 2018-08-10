import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler #x minus mean divided by standard deviation
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from collections import Counter
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State,Event
from plotly import tools
#######################################################
sc = StandardScaler()
#######################################################

listOfCountries=pd.read_csv('List_Of_Countries.csv')

listOfCountries.set_index('Economy',drop=False,inplace=True)#reset index to Economy
listOfCountries.iloc[47,1]='Ivory Coast'
listOfCountries.iloc[50,1]='Curacao'

economicFinalDf=pd.read_csv('ECOFINAL.CSV')#had to be imported as CSVs due to DILL issues
economicFinalDf.set_index('Country',drop=False,inplace=True)

environmentalFinalDf=pd.read_csv('Environment.csv')
environmentalFinalDf.set_index('Economy',drop=True,inplace=True)

socialFinalDf=pd.read_csv('SOCFINAL.CSV')
socialFinalDf.set_index('Country',drop=False,inplace=True)

politicalFinalDf=pd.read_csv('POLFINAL.CSV')
politicalFinalDf.set_index('Country',drop=False,inplace=True)

#############################################################

def replaceIndex(df):
    as_list = df.index.tolist()
    idx = as_list.index('Curaçao')#changed for version 3
    as_list[idx] = 'Curacao'
    df.index = as_list
    idx = as_list.index("Côte d'Ivoire")# changed for version 3
    as_list[idx] = 'Ivory Coast'
    df.index = as_list
    return df

###############################################################



def deletes(eco,env,pol,soc):#getting rid of unwated duplicated columns

    for i in ['Tariff Rate (%)', 'Income Tax Rate (%)', 'Corporate Tax Rate (%)', 
              'Tax Burden % of GDP', 'Public Debt (% of GDP)', 'Population (Millions)']:
        del eco[i]
    
    
    for i in ['Depth of food deficit - capped','Adult literacy rate - capped', 'Secondary school enrollment - capped',
     'Gender parity in secondary enrollment - difference from parity', 'Mobile telephone subscriptions - capped',
     'Greenhouse gas emissions - capped', 'Globally ranked universities - bucketed',
     'Percentage of tertiary students enrolled in globally ranked universities - bucketed',
     "Nutrition and Basic Medical Care","Water and Sanitation","Shelter","Personal Safety","Access to Basic Knowledge",
    "Access to Information and Communications","Health and Wellness","Environmental Quality",
    "Personal Rights","Personal Freedom and Choice","Tolerance and Inclusion","Access to Advanced Education"]:
        del soc[i]#found them to be useless columns by observation, they are repeated or aggregates
    
    #### Invert the scales of identified columns where Higher score is not associated with lower risk
    
    
    # IV. Create smaller buckets of features
    
    for i in [eco,env,soc,pol]:
        for k in ['Country','Income group']:
            if k in i.columns:
                del i[k]
    
    #### Collapsing Economic scores to one metric
    
    scaledEconomicDf = eco.iloc[:,:12]
    
    #### Collapsing Political into one
    
    for i in [u'E1: Economy', u'E2: Economic Inequality',u'P2: Public Services', u'P3: Human Rights',u'S1: Demographic Pressures']:
        del pol[i]#remove duplicated information from other dataframes
    scaledPoliticalDf = pol
    
    #### Collapsing Enivronmental Score
    
    #### move columns out of environmental and concat them with social
    
    scaledEnvironmentalDf = env
    
    #### Collapsing Social Score
    
    ##### Remove identified unwated columns
    
    '''for i in  ['Personal Rights', 'Personal Freedom and Choice', 'Tolerance and Inclusion', 'Mobile telephone subscriptions', 
               'Internet users', 'Political rights', 'Freedom of expression', 'Freedom of assembly', 'Private property rights', 
               'Freedom over life choices', 'Freedom of religion', 'Corruption', 'Tolerance for immigrants', 'Press Freedom Index',
               'Tolerance for homosexuals', 'Discrimination and violence against minorities', 'Religious tolerance']:
        del soc[i]'''
    scaledSocialDf = soc#seem to be repeated columns
    
    # V. Concatenate into 1 dataframe
    
    #for i in [scaledEconomicDf,scaledEnvironmentalDf,scaledPoliticalDf,scaledSocialDf]:
    #    print (i.shape)
    
    df_concated=pd.concat([scaledEconomicDf,scaledEnvironmentalDf,scaledPoliticalDf,scaledSocialDf],axis=1)
    df_concated.columns=['Property Rights', 'Judical Effectiveness', 'Government Integrity',
       'Tax Burden', 'Gov\'t Spending', 'Fiscal Health', 'Business Freedom',
       'Labor Freedom', 'Monetary Freedom', 'Trade Freedom',
       'Investment Freedom ', 'Financial Freedom', 'HazardValue',
       'Security Apparatus', 'Factionalized Elites',
       'Group Grievance', 'Human Flight and Brain Drain',
       'State Legitimacy', 'Refugees and IDPs',
       'External Intervention', 'Undernourishment',
       'Depth of food deficit', 'Maternal mortality rate',
       'Child mortality rate', 'Deaths from infectious diseases',
       'Access to piped water', 'Rural access to improved water source',
       'Access to improved sanitation facilities',
       'Availability of affordable housing', 'Access to electricity',
       'Quality of electricity supply',
       'Household air pollution attributable deaths', 'Homicide rate',
       'Level of violent crime', 'Perceived criminality', 'Political terror',
       'Traffic deaths', 'Adult literacy rate', 'Primary school enrollment',
       'Secondary school enrollment', 'Gender parity in secondary enrollment',
       'Mobile telephone subscriptions', 'Internet users',
       'Press Freedom Index', 'Life expectancy at 60',
       'Premature deaths from non-communicable diseases', 'Suicide rate',
       'Outdoor air pollution attributable deaths', 'Wastewater treatment',
       'Biodiversity and habitat', 'Greenhouse gas emissions',
       'Political rights', 'Freedom of expression', 'Freedom of assembly',
       'Private property rights', 'Freedom over life choices',
       'Freedom of religion', 'Early marriage',
       'Satisfied demand for contraception', 'Corruption',
       'Tolerance for immigrants', 'Tolerance for homosexuals',
       'Discrimination and violence against minorities', 'Religious tolerance',
       'Community safety net', 'Years of tertiary schooling',
       'Women\'s average years in school',
       'Inequality in the attainment of education',
       'Globally ranked universities',
       'Percentage of tertiary students enrolled in globally ranked universities']
   
    #return (scaledEconomicDf,scaledEnvironmentalDf,scaledPoliticalDf,scaledSocialDf)
    return df_concated


####################################################################
def removecorr(dff,corr=0.8):
    df=dff.copy()
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # find the list of columns to drop
    to_drop = [column for column in upper.columns if any(upper[column] >= corr)]
    #drop duplicated information columns columns
    df.drop(to_drop, axis=1,inplace=True)
    #create concated df with income group
    df_Ret=pd.DataFrame(index=df.index,columns=df.columns,data=StandardScaler().fit_transform(df))
    #add income group
    df_Ret=df_Ret.join(listOfCountries)

    for i in ['x','Economy','Code','Region']:
        del df_Ret[i]
    return df_Ret

def plotExplainedVariance(df_concated_income):
    pca_1 = PCA()
    df=df_concated_income.copy()
    #X = pca_1.fit(df_concated_income[df_concated_income['Income group']==group].iloc[:,:-1].values)
    #X = pca_1.fit(df_concated_income.iloc[:,:-1].values)
    pca_1.fit(df.values)
    explained_variance=np.cumsum(pca_1.explained_variance_ratio_)
    #return explained_variance
    return explained_variance
    #plt.figure(figsize=(6,4))
    #plt.plot(explained_variance)
    #plt.show()

def calcSVM(values,group='High income',nu=0.1,kernel="rbf",gamma=0.1):
    #run a PCA and Fit a OneClassSVM object
    pca_1 = PCA(n_components=2)
    transformed=pca_1.fit_transform(values)
    clf=svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(transformed)
    y_pred_train = clf.predict(transformed)
    description=[group,nu,kernel,gamma]
    return [transformed,y_pred_train,description,clf]

def calcIsoFor(values,group='High income',estimators=25,contamination=0.1):
    #run a PCA and Fit a IsoForest object
    pca_1 = PCA(n_components=2)
    transformed=pca_1.fit_transform(values)
    clf = IsolationForest(n_estimators=estimators,contamination=contamination)
    clf.fit(transformed)
    y_pred_train = clf.predict(transformed)
    description=[group,estimators]
    return [transformed,y_pred_train,description,clf]

def plotSVM(transformed,description,classifier):
    # get boundaries
    fig = tools.make_subplots()
    xmax=pd.DataFrame(transformed).describe()[0]['max']
    xmin=pd.DataFrame(transformed).describe()[0]['min']
    ymax=pd.DataFrame(transformed).describe()[1]['max']
    ymin=pd.DataFrame(transformed).describe()[1]['min']
    # plot the contour graph
    X=np.linspace(xmin-5, xmax+5, 500) 
    Y=np.linspace(ymin-5, ymax+5, 500)
    xx, yy = np.meshgrid(X,Y)
    clf=classifier
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    

    back = go.Contour(x=X,y=Y,z=Z,
                  autocontour=True,
                  #ncontours=7,
                  contours=dict(showlines=False),
                  showscale=False)
                  #colorscale = matplotlib_to_plotly(plt.cm.Blues, 10))
    #X_train=transformed.copy()
    b = go.Scatter(x=transformed[:, 0], 
               y=transformed[:, 1],
               showlegend=False,
               text=listOfCountries.index,
               mode='markers',
               marker=dict(color='white',line=dict(color='black', width=1))
               )

    fig.append_trace(back, 1, 1)
    fig.append_trace(b, 1, 1)
    #row+=1

    fig['layout'].update(height=900,hovermode='closest',hoverdistance=1)
    return fig


#x = 'xaxis' + '1'
#y = 'yaxis' + '1'
#fig['layout'][x].update(showticklabels=False, ticks='')
#fig['layout'][y].update(showticklabels=False, ticks='')

def plotIsoForest(transformed,description,classifier):
        # get boundaries
    fig = tools.make_subplots()
    xmax=pd.DataFrame(transformed).describe()[0]['max']
    xmin=pd.DataFrame(transformed).describe()[0]['min']
    ymax=pd.DataFrame(transformed).describe()[1]['max']
    ymin=pd.DataFrame(transformed).describe()[1]['min']
    # plot the contour graph
    X=np.linspace(xmin-5, xmax+5, 500) 
    Y=np.linspace(ymin-5, ymax+5, 500)
    xx, yy = np.meshgrid(X,Y)
    clf=classifier
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    

    back = go.Contour(x=X,y=Y,z=Z,
                  autocontour=True,
                  #ncontours=7,
                  contours=dict(showlines=False),
                  showscale=False)
                  #colorscale = matplotlib_to_plotly(plt.cm.Blues, 10))
    #X_train=transformed.copy()
    b = go.Scatter(x=transformed[:, 0], 
               y=transformed[:, 1],
               showlegend=False,
               text=listOfCountries.index,
               mode='markers',
               marker=dict(color='white',line=dict(color='black', width=1))
               )

    fig.append_trace(back, 1, 1)
    fig.append_trace(b, 1, 1)
    #row+=1

    fig['layout'].update(height=900,hovermode='closest',hoverdistance=1)
    return fig

def bothResults(df_concated_income,group='High income',nu=0.1,kernel="rbf",gamma=0.1,estimators=25,contamination=0.1,run='BOTH'):
    if(run=='SVM'):
        svmRet=calcSVM(df_concated_income.iloc[:,:-1].values,nu=nu,kernel=kernel,gamma=gamma)
        #isoRet=calcIsoFor(df_concated_income.iloc[:,:-1].values,estimators=estimators)
        svmDf=pd.DataFrame(data=svmRet[1],index=df_concated_income.index,columns=['SVM_Prediction'])
        #isoDf=pd.DataFrame(data=isoRet[1],index=df_concated_income.index,columns=['IsoFor_Prediction'])
        combinedDf=svmDf.copy()
        for i in combinedDf.index:
            combinedDf.at[i,'Income group']=df_concated_income.at[i,'Income group']
        combinedDf=combinedDf[combinedDf['SVM_Prediction']==-1].copy()
        #combinedDf=combinedDf[combinedDf['IsoFor_Prediction']==-1].copy()
        combinedDf=combinedDf[combinedDf['Income group']==group].copy()
        grouped=df_concated_income.groupby('Income group')
        incomeDf=grouped.get_group(group).copy()
        relevantDf=pd.DataFrame(data=0.0,index=combinedDf.index,columns=df_concated_income.columns)
        for i in combinedDf.index:
            relevantDf.loc[i,:]=df_concated_income.loc[i,:]
        return relevantDf.T.join(incomeDf.describe().T).drop(['count'],axis=1).drop(['Income group'],axis=0)
    elif (run=='ISOFOR'):
        #svmRet=calcSVM(df_concated_income.iloc[:,:-1].values,nu=0.1,kernel="rbf",gamma=0.1)
        isoRet=calcIsoFor(df_concated_income.iloc[:,:-1].values,estimators=estimators,contamination=contamination)
        #svmDf=pd.DataFrame(data=svmRet[1],index=df_concated_income.index,columns=['SVM_Prediction'])
        isoDf=pd.DataFrame(data=isoRet[1],index=df_concated_income.index,columns=['IsoFor_Prediction'])
        combinedDf=isoDf.copy()
        for i in combinedDf.index:
            combinedDf.at[i,'Income group']=df_concated_income.at[i,'Income group']
        #combinedDf=combinedDf[combinedDf['SVM_Prediction']==-1].copy()
        combinedDf=combinedDf[combinedDf['IsoFor_Prediction']==-1].copy()
        combinedDf=combinedDf[combinedDf['Income group']==group].copy()
        grouped=df_concated_income.groupby('Income group')
        incomeDf=grouped.get_group(group).copy()
        relevantDf=pd.DataFrame(data=0.0,index=combinedDf.index,columns=df_concated_income.columns)
        for i in combinedDf.index:
            relevantDf.loc[i,:]=df_concated_income.loc[i,:]
        return relevantDf.T.join(incomeDf.describe().T).drop(['count'],axis=1).drop(['Income group'],axis=0)
    elif (run=='BOTH'):
        svmRet=calcSVM(df_concated_income.iloc[:,:-1].values,nu=nu,kernel=kernel,gamma=gamma)
        isoRet=calcIsoFor(df_concated_income.iloc[:,:-1].values,estimators=estimators,contamination=contamination)
        svmDf=pd.DataFrame(data=svmRet[1],index=df_concated_income.index,columns=['SVM_Prediction'])
        isoDf=pd.DataFrame(data=isoRet[1],index=df_concated_income.index,columns=['IsoFor_Prediction'])
        combinedDf=svmDf.join(isoDf,how='inner')#orm intersection of calling frame’s index (or column if on is specified) with other frame’s index, preserving the order of the calling’s one
        for i in combinedDf.index:
            combinedDf.at[i,'Income group']=df_concated_income.at[i,'Income group']
        combinedDf=combinedDf[combinedDf['SVM_Prediction']==-1].copy()
        combinedDf=combinedDf[combinedDf['IsoFor_Prediction']==-1].copy()
        combinedDf=combinedDf[combinedDf['Income group']==group].copy()
        grouped=df_concated_income.groupby('Income group')
        incomeDf=grouped.get_group(group).copy()
        relevantDf=pd.DataFrame(data=0.0,index=combinedDf.index,columns=df_concated_income.columns)
        for i in combinedDf.index:
            relevantDf.loc[i,:]=df_concated_income.loc[i,:]
        return relevantDf.T.join(incomeDf.describe().T).drop(['count'],axis=1).drop(['Income group'],axis=0)

def analyseResult(country,descriptive):#provide the country df and descriptive statistics df
    resultDf=pd.concat([country,descriptive],axis=1)
    resultDf['1_SD_below']=resultDf['mean']-1.0*resultDf['std'] # for each parameter calculate 1SD and 2SD above and below
    resultDf['2_SD_below']=resultDf['mean']-2.0*resultDf['std']
    resultDf['1_SD_above']=resultDf['mean']+1.0*resultDf['std']
    resultDf['2_SD_above']=resultDf['mean']+2.0*resultDf['std']
    
    dictRet={k:0 for k in ['Country','Score','MeanScore','TwoSDBelow','OneSDBelow','TwoSDAbove','OneSDAbove','TypeOf',
                           'FarBehind','Behind','FarAhead','Ahead']}
    CountryList=[]
    ScoreList=[]
    MeanScoreList=[]
    TwoSDBelowList=[]
    OneSDBelowList=[]
    TwoSDAboveList=[]
    OneSDAboveList=[]
    TypeOfList=[]
    FarBehindList=[]
    BehindList=[]
    FarAheadList=[]
    AheadList=[]
    for col in resultDf.columns[:-11]: # for each country in the columns
        CountryList.append(col)
        # Total scores
        score=resultDf[col].sum() #this is the country total score
        meanscore=resultDf['mean'].sum()#this is a country with all mean scores
        ScoreList.append("%.2f"%(resultDf[col].sum())) #calculate a total country score for each country
        MeanScoreList.append("%.2f"%(resultDf['mean'].sum())) #calculate a mean country score which is a sum of mean scores across all parameters

        # Type of outlier
        if(score>meanscore): #check whether a country is a positive or negative outlier
            TypeOfList.append('positive outlier')
        else:
            TypeOfList.append('negative outlier')

        # comparisonWithMean
        OneSDBelowList.append(Counter(resultDf[col]<=resultDf['1_SD_below'])[1])#count the no.of params on which a country is below 1SD
        TwoSDBelowList.append(Counter(resultDf[col]<=resultDf['2_SD_below'])[1])#count the no.of params on which a country is below 2SD
        OneSDAboveList.append(Counter(resultDf[col]>=resultDf['1_SD_above'])[1])#count the no.of params on which a country is above 1SD
        TwoSDAboveList.append(Counter(resultDf[col]>=resultDf['2_SD_above'])[1])#count the no.of params on which a country is above 2SD
        
        lister1=[]
        lister2=[]
        lister3=[]
        lister4=[]
        #lister5=[]
        
        for row in resultDf.index:#for each parameter
            if resultDf.at[row,col]<=resultDf.at[row,'2_SD_below']:#if a country is below 2SD on a param
                lister1.append(row)#it is far behind than the global average
            elif (resultDf.at[row,col]<=resultDf.at[row,'1_SD_below'] and resultDf.at[row,col]>resultDf.at[row,'2_SD_below']):
                lister2.append(row)#if it is between 1 to 2 SD below, it is behind global average
        for row in resultDf.index:#for each parameter    
            if resultDf.at[row,col]>=resultDf.at[row,'2_SD_above'] :
                lister3.append(row)#if it is 2 SD above then it is far ahead than the global average
            elif (resultDf.at[row,col]>=resultDf.at[row,'1_SD_above'] and resultDf.at[row,col]<resultDf.at[row,'2_SD_above']) :
                lister4.append(row)#if it is 1 to 2 SD above then it is ahead than the global average
            
            
        FarBehindList.append(','.join(lister1))
        BehindList.append(','.join(lister2))
        FarAheadList.append(','.join(lister3))
        AheadList.append(','.join(lister4))
        
        #lister5.append((TwoSDBelow,OneSDBelow,TwoSDAbove,OneSDAbove,score,meanscore,typeOf,lister3,lister4,lister1,lister2))
        
        #dictRet[col]=lister5
    
    dictRet['Country']=CountryList
    dictRet['Score']=ScoreList
    dictRet['MeanScore']=MeanScoreList
    dictRet['TwoSDBelow']=TwoSDBelowList
    dictRet['OneSDBelow']=OneSDBelowList
    dictRet['TwoSDAbove']=TwoSDAboveList
    dictRet['OneSDAbove']=OneSDAboveList
    dictRet['TypeOf']=TypeOfList
    dictRet['FarBehind']=FarBehindList
    dictRet['Behind']=BehindList
    dictRet['FarAhead']=FarAheadList
    dictRet['Ahead']=AheadList
        
    return dictRet
##################################################################
    
###### Pre-Process for PCA
def runPCA():
    dfCorrRemoved=removecorr(df_concated,0.8)#return a dataframe after taking care of correlations given by the user
    X=plotExplainedVariance(dfCorrRemoved.iloc[:,:-1])#get the values of the components required for plotting the PCA
    return {
        'data':[
            go.Scatter(
                x=[i for i in range(1,len(X))], 
                y=X                    
            )
        ],
        'layout':{
                'height':650,
                #'width':500
            }

    }
############

def runPCASVM():
    dfret=removecorr(df_concated,0.8)
    R=calcSVM(dfret.iloc[:,:-1].values,nu=0.1,kernel="rbf",gamma=0.1)
    return plotSVM(R[0],R[2],R[3])
#############
def runISOFOR():
    dfret=removecorr(df_concated,0.8)
    R=calcIsoFor(dfret.iloc[:,:-1].values)
    return plotIsoForest(R[0],R[2],R[3])
#############
#df_concated_income=removecorr(df_concated,0.8)
#RT=bothResults(df_concated_income,estimators=25,contamination=0.1,nu=0.1,kernel='rbf',gamma=0.1,group='High income',run='BOTH')    
#rets=analyseResult(country=RT.iloc[:,:-7],descriptive=RT.iloc[:,-7:])
def generate_table(dataframe):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))]
    )
        
         
############################################################################
for i in [economicFinalDf,socialFinalDf,politicalFinalDf]:
    i=replaceIndex(i)

for i in [economicFinalDf,environmentalFinalDf,socialFinalDf,politicalFinalDf]:
    i.sort_index(inplace=True)
###############################################################
df_concated=deletes(economicFinalDf.copy(),
                  environmentalFinalDf.copy(),
                  politicalFinalDf.copy(),
                  socialFinalDf.copy())
df=df_concated.copy()

# inverting the scales for parameters where higher value is not associated with low risk
for i in [u'HazardValue',u'Security Apparatus', u'Factionalized Elites',u'Group Grievance', 
u'Human Flight and Brain Drain', u'State Legitimacy', u'Refugees and IDPs',
u'External Intervention','Undernourishment', 'Depth of food deficit', 
'Maternal mortality rate', 'Child mortality rate', 
'Deaths from infectious diseases','Household air pollution attributable deaths', 
'Homicide rate','Level of violent crime', 'Perceived criminality', 'Political terror', 'Traffic deaths',
'Premature deaths from non-communicable diseases', 'Suicide rate', 
'Outdoor air pollution attributable deaths','Greenhouse gas emissions', 'Early marriage',
'Inequality in the attainment of education','Press Freedom Index',
'Discrimination and violence against minorities']:
    df[i]=-1.0*df[i]

describer=df.describe()

for col in df.columns:
    for row in df.index:
        if(df.at[row,col]<=describer.at['25%',col]):
            df.at[row,col]=25
        elif(df.at[row,col]<=describer.at['50%',col]):
            df.at[row,col]=50
        elif(df.at[row,col]<=describer.at['75%',col]):
            df.at[row,col]=75
        else:
            df.at[row,col]=100
df['Total']=df.sum(axis=1)
countries=listOfCountries[['Code','Economy']].copy()
df=df.join(countries)
available_indicators=df.columns[:-2]#for choropleth

#app = dash.Dash(__name__, static_folder='resources')
app = dash.Dash(__name__)
server = app.server


colors = {
    'background': '#111111',#black
    'text': '#7FDBFF',#light blue
    'heading':'#FF0000',#red
    'heading2':'#F4CE42',#yellow
    'heading3':'#5342F4',#blue
}

box1='Estimating country risk is and has always been a complex and cumbersome exercise. Many industries, such as banking, insurance, hedge funds, transportation and logistics have been traditional users of tools which estimate financial and political country risks. Then there are tools such as Munich RE group’s popular Nathan system (https://www.munichre.com/en/reinsurance/business/non-life/nathan/index.html) which estimates a country’s natural hazard risk. A third category of tools such as non-profit Social Progress Imperative’s SP Index that estimates social risk associated with countries. \
For someone such as a country analyst in a development bank like World Bank or a university researcher or someone like you and me, who may have to move from one country to another in pursuit of our globalized careers, it does not make sense to refer to so many different tools (time effective, cost effective) to gather information. \
Through my capstone I have tried to bridge this information gap by creating a tool that presents a country risk paradigm encompassing Economic, Political, Social and Environmental Risk factors. I then go on to use ‘Income classification’ of countries and hunt for outlier countries. I go one step deeper to investigate the reasons why some countries are outliers.\
I give the user the flexibility to choose from a support vector machine based model or a Isolation Forest based model to generate the outlier results.'
box2='Environmental Risk - Web Scraping, source - https://www.preventionweb.net/english/   |   \
Economic Risk - Static CSV file read , source - Heritage Foundation, https://www.heritage.org/index/ranking   |   \
Political Risk - Static CSV file read , source - Fund for Peace , http://fundforpeace.org/global/   |   \
Social Risk - Static CSV file read , source - Social Productive Index, https://www.socialprogressindex.com/. References - United Nations University World Risk Report https://ehs.unu.edu/blog/articles/world-risk-report-2016-the-importance-of-infrastructure.html  '

box3='1) The country names in none of the datasets matched with one another. 2) There were countries that were missing in the dataset entirely.  3) There were countries that had some missing data. \
To tackle these challenges, I used World Bank’s list of countries as a standard list. I rectified the country names in each of the 4 datasets using text matching, and on some instances,  I had to do this job manually. \
For missing data, I devised an imputation strategy based on income group classification of a country. The mean of all High-Income countries on a parameter X, would be applied to any High-Income country missing  data on parameter X.  \
Data Munging occupied a large chunk of my effort towards this capstone.'
app.layout = html.Div( children=[
    html.H1(
        children='Country Risk Analysis and Outlier Detection using OneClassSVM and Isolation Forests',
        style={
            'textAlign': 'Left',
            'color': colors['heading']
        }
    ),

    html.H3(children='Capstone Project submitted by Anuj Patni, Fellow, The Data Incubator', style={
        'textAlign': 'Left',
        'color': colors['heading']
    }),
    html.H3(children='Business Objective', style={
        'textAlign': 'Left',
        'color': colors['heading3']
    }),
    html.H6(id='Param-text-out-BO',children=box1,style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    html.H3(children='Data Ingestion', style={
        'textAlign': 'Left',
        'color': colors['heading3']
    }),
    
    html.H6(id='Param-text-out-DI',children=box2,style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    html.H3(children='Challenges with Data Munging', style={
        'textAlign': 'Left',
        'color': colors['heading3']
    }),
    html.H6(id='Param-text-out-DM',children=box3,style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    html.H6(id='Param-text-out',children='Choose a parameter and see what happens on the World Map\n',style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    html.H3(id='Param-text-out-Choro',children='Visualization 1 - Choropleth',style={
        'textAlign': 'left',
        'color': colors['heading3']
    }),
    dcc.Dropdown(
                id='paramtersDropdown',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Total',
                searchable=True
            ),
    dcc.Graph(
        id='choropleth',
        figure={
            'data': [
                go.Choropleth(
                    locations=df['Code'],
                    z = df['Total'],
                    text = df['Economy'],
                    colorscale = [[0,"rgb(75, 83, 32)"],[1.0/3,"rgb(164, 198, 57)"],\
                    [2.0/3,"rgb(209, 226, 49)"],[1,"rgb(223, 255, 0)"]],
                    autocolorscale = False,
                    reversescale = False,
                    marker = dict(
                        line = dict (
                            color = 'rgb(180,180,180)',
                            width = 0.5
                        ) 
                    ),
                    colorbar = dict(
                        #autotick = False,
                        tickprefix = '= ',
                        title = ''
                    ),
              )
            ],
            'layout':{
                'height' : 650,
                #'width' : 800,
                #'plot_bgcolor':colors['background'],
                #'paper_bgcolor':colors['background'],
                'font': {
                    'color':colors['background']
                },
                'title' : 'A higher score means that the country fares better compared to others',
                'geo' : dict(
                    showframe = False,
                    showcoastlines = True,
                    showocean=True,
                    showlakes=True,
                    showcountries=True,
                    projection = dict(
                        type = 'natural earth'
                    )
                )
            }
        }
    ),
    
    ################################################ For PCA explained variance
    html.H3(id='Param-text-out-PCA',children='Visualization 2 - PCA',style={
        'textAlign': 'left',
        'color': colors['heading3']
    }),
    
    html.H6(id='PCA-text-out',children='This is the PCA explained variance graph. Choose a correlation factor to remove correlated features and click submit. Default correlations is set at 0.8.',style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    
    dcc.Input(id='corr-text-id', value=0.8, type='number',max=1.0,min=0.0,
              step=0.05,
              ),
    html.Button(id='corr-submit-button', n_clicks=0,children='Submit',style={
                'textAlign': 'center',
                'color': colors['background']
            }),    
    dcc.Graph(
        id='PCA',
        #style={'display':'none','vertical-align': 'middle'},
        figure=runPCA()
    ),
      
    
    ############################################################### For PCA and SVM
    html.H3(id='Param-text-out-PCASVM',children='Visualization 3 - SVM using 2 components',style={
        'textAlign': 'left',
        'color': colors['heading3']
    }),
    html.H6(id='PCASVM-static-input',children='Result of OneClassSVM',style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    html.Div(id='PCASVM-nu-static-out',children="NU",style={
        'textAlign': 'left',
        'color': colors['background']
    }),  
    dcc.Input(id='PCASVM-nu', value=0.1, type='number'),
    html.Div(id='PCASVM-kernel-static-out',children="Kernel",style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    dcc.Dropdown(
                id='PCASVM-kernel',
                options=[{'label': i, 'value': i} for i in ['rbf','sigmoid', 'linear','polynomial']],
                value='rbf',
                searchable=True
            ),
    #dcc.Input(id='PCASVM-kernel', value='rbf', type='text'),
    html.Div(id='PCASVM-gamma-static-out',children="Gamma",style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    dcc.Input(id='PCASVM-gamma', value=0.1, type='number'),
    html.Button(id='PCASVM-submit-button', n_clicks=0, children='Submit',style={
        'textAlign': 'center',
        'color': colors['background']
    }), 
    
    dcc.Graph(
        id='Contour1',
        #style={'display':'none'},
        figure=runPCASVM()
    ),
       
    
    ################################################################################ For PCA and ISOFOR
    #estimators=25,contamination=0.1
    html.H3(id='Param-text-out-PCAISOFOR',children='Visualization 4 - ISOFOR using 2 components',style={
        'textAlign': 'left',
        'color': colors['heading3']
    }),
    html.H6(id='PCAISOFOR-static-input',children='Result of Isolation Forest',style={
        'textAlign': 'left',
        'color': colors['background']
    }),    
    
    html.Div(id='PCAISOFOR-estimators-static-out',children="Estimators",style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    
    dcc.Input(id='PCAISOFOR-estimators', value=25, type='number'),
    
    html.Div(id='PCAISOFOR-contamination-static-out',children="Contamination",style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    
    dcc.Input(id='PCAISOFOR-contamination', value=0.1, type='number'),    
    
    html.Button(id='PCAISOFOR-submit-button', n_clicks=0, children='Submit',style={
        'textAlign': 'center',
        'color': colors['background']
    }),

    dcc.Graph(
        id='Contour2',
        #style={'display':'none'},
        figure=runISOFOR()
    ),
    ################################################################################### For Results
    #def bothResults(df_concated_income,group='High income',nu=0.1,kernel="rbf",gamma=0.1,estimators=25,contamination=0.1,run='BOTH'):
    
    html.H4(id='Results-static-outputTable',children='Results Analysis',style={
        'textAlign': 'left',
        'color': colors['heading3']
    }),
    
    html.H6(id='Results-static-outputBelowTable',children='To generate results select the income group and the type of run',style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    
    html.Div(id='Results-group-static-out',children="Group",style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    
    #dcc.Input(id='Results-group-dd', value='High income', type='text'),
    
    dcc.Dropdown(
                id='Results-group',
                options=[{'label': i, 'value': i} for i in ['High income','Low income', 'Upper middle income','Lower middle income']],
                value='High income',
                searchable=True
            ),
    
    html.Div(id='Results-run-static-out',children="Run",style={
        'textAlign': 'left',
        'color': colors['background']
    }),
    
    #dcc.Input(id='Results-run', value='BOTH', type='text'),    
    
    dcc.Dropdown(
                id='Results-run',
                options=[{'label': i, 'value': i} for i in ['BOTH','SVM', 'ISOFOR']],
                value='BOTH',
                searchable=True
            ),
    
    html.Button(id='Results-submit-button', n_clicks=0, children='Submit',style={
        'textAlign': 'center',
        'color': colors['background']
    }),   
       
    html.Table(id='Table1'),
    ################################################################################### For DataDef
    
    html.H4(id='Results-static-outputDefinitions',children='Data Definitions',style={
        'textAlign': 'left',
        'color': colors['heading3']
    }),
    html.Table(id='Table2',children=generate_table(pd.read_csv('DataDefinitions.csv')))
    #

])

########################################################Generate Results on click of submit button
@app.callback(
    Output('Table1', 'children'),
    [Input('Results-submit-button', 'n_clicks')],
    [State('corr-text-id', 'value'),
     State('PCAISOFOR-estimators', 'value'),
     State('PCAISOFOR-contamination', 'value'),
     State('PCASVM-nu','value'),
     State('PCASVM-kernel','value'),
     State('PCASVM-gamma','value'),
     State('Results-group','value'),
     State('Results-run','value')
    ]
)
def update_ResultsOfOutput(n_clicksR,corr,estimators,contamination,nu,kernel,gamma,group,run): 
    #if(n_clicksR>0):
    df_concated_income1=removecorr(df_concated,corr)
    RT1=bothResults(df_concated_income1,estimators=estimators,contamination=contamination,nu=nu,kernel=kernel,gamma=gamma,group=group,run=run)    
    rets1=analyseResult(country=RT1.iloc[:,:-7],descriptive=RT1.iloc[:,-7:])
    #global holderR
    #holderR=holderR+1
    return (generate_table(pd.DataFrame(rets1)))
    #return u'''{}'''.format(rets1) 
    '''df_concated_income1=removecorr(df_concated,corr)
        RT1=bothResults(df_concated_income1,estimators=estimators,contamination=contamination,nu=nu,kernel=kernel,gamma=gamma,group=group,run=run)    
        rets1=analyseResult(country=RT1.iloc[:,:-7],descriptive=RT1.iloc[:,-7:])
        return generate_table(pd.DataFrame(rets1))
    else:
        return generate_table(pd.DataFrame(rets))'''
        
##################################################Performing PCA and ISOFOR on click of submit button
@app.callback(
    Output('Contour2', 'figure'),
    [Input('PCAISOFOR-submit-button', 'n_clicks')],
    [State('corr-text-id', 'value'),
     State('PCAISOFOR-estimators', 'value'),
     State('PCAISOFOR-contamination', 'value')
    ]
)
def update_PCAISOFORoutput(n_clicksI,corr,estimators,contamination):
    #if(n_clicksI>0): 
        
    dfret=removecorr(df_concated,corr)
    R=calcIsoFor(dfret.iloc[:,:-1].values,estimators=estimators,contamination=contamination)
    return plotIsoForest(R[0],R[2],R[3])
           
########################################################Performing PCA and SVM on click of submit button
@app.callback(
    Output('Contour1', 'figure'),
    [Input('PCASVM-submit-button', 'n_clicks')],
    [State('corr-text-id', 'value'),
     State('PCASVM-nu','value'),
     State('PCASVM-kernel','value'),
     State('PCASVM-gamma','value')])
def update_PCASVMoutput(n_clicksS,corr,nu,kernel,gamma):
    #if(n_clicksS>0): 
        
    dfret=removecorr(df_concated,corr)
    R=calcSVM(dfret.iloc[:,:-1].values,nu=nu,kernel=kernel,gamma=gamma)
    return plotSVM(R[0],R[2],R[3])
    
#############################################################################Performing PCA explained variance
@app.callback(
    Output('PCA', 'figure'),
    [Input('corr-submit-button', 'n_clicks')],
    [State('corr-text-id', 'value')])
def update_output(n_clicksP, input1):
    #if(n_clicksP>0):
    dfCorrRemoved=removecorr(df_concated,input1)#return a dataframe after taking care of correlations given by the user
    X=plotExplainedVariance(dfCorrRemoved.iloc[:,:-1])#get the values of the components required for plotting the PCA
    return {
        'data':[
            go.Scatter(
                x=[i for i in range(1,len(X))], 
                y=X                    
            )
        ]

    }
        
######################################################################Parameter dropdown for Choropleth
@app.callback(
    dash.dependencies.Output('choropleth', 'figure'),
   [dash.dependencies.Input('paramtersDropdown', 'value')]
)
def update_Choropleth(parameter_value):
    return{
        'data': [
                go.Choropleth(
                    locations=df['Code'],
                    z = df[parameter_value],
                    text = df['Economy'],
                    colorscale = [[0,"rgb(75, 83, 32)"],[1.0/3,"rgb(164, 198, 57)"],\
                    [2.0/3,"rgb(209, 226, 49)"],[1,"rgb(223, 255, 0)"]],
                    autocolorscale = False,
                    reversescale = False,
                    marker = dict(
                        line = dict (
                            color = 'rgb(180,180,180)',
                            width = 0.5
                        ) 
                    ),
                    colorbar = dict(
                        #autotick = False,
                        tickprefix = '= ',
                        title = ''
                    ),
              )
            ],
            'layout':{
                'height' : 650,
                #'width':800,
                #'plot_bgcolor':colors['background'],
                #'paper_bgcolor':colors['background'],
                'font': {
                    'color':colors['background']
                },
                'title' : 'A higher score means that the country fares better compared to others',
                'geo' : dict(
                    showframe = False,
                    showcoastlines = True,
                    showocean=True,
                    showlakes=True,
                    showcountries=True,
                    projection = dict(
                        type = 'natural earth'
                    )
                )
            }
    }
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    #############
    
    app.run_server()
 