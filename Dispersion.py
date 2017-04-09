"""

@author: Jayesh
"""
import warnings 

import datetime
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#import pandas.io.data as web
#import zipline as zp
import statsmodels.tsa.stattools as ts
import sys
import math
import scipy.stats as sp
warnings.filterwarnings('ignore') 
start_time = time.time()
def ceil(x, s):
    return int(s * math.ceil(float(x)/s))

def floor(x, s):
    return int(s * math.floor(float(x)/s))
vfloor=np.vectorize(floor)
vceil=np.vectorize(ceil)
#Strike Difference and No. of Strikes being considered

no_of_strikes=6 #No.of Securities

Strike_Diff=(5,5,20,10,10,100)
Lot_Size=[3000,3500,500,800,1200,40]
Start_Time=01/04/2016
Corel_Weight=[0.2049, 0.1141, 0.2702,0.0936,0.131]
Lots_Bought=[10,10,10,10,10,50] #Quantity

Expiry=datetime.datetime(2016,04,25,0,0,0)
Interest=0
Dividend=0
#Black Scholes formula calculations
#Function to calculate dOne

#(sym[0]['Time']-Expiry)/np.timedelta64(1,'s')

def dOne (Underlying, Excercise, Time, Interest, Volatility, Dividend):
    dOne =(np.log(Underlying/float(Excercise))+((float(Interest)-Dividend+(0.5 *np.square(Volatility)))*Time))/(Volatility*np.sqrt(Time))    
    return dOne
#Function to calculate NdOne

def NdOne (Underlying, Excercise, Time, Interest, Volatility, Dividend):
    NdOne= np.exp(-(np.square(dOne(Underlying, Excercise, Time, Interest, Volatility, Dividend)) / 2)) / (np.sqrt(2 * 3.14159265358979))
    return NdOne
#Function to calculate dTwo    

def dTwo (Underlying, Excercise, Time, Interest, Volatility, Dividend):
    dTwo = dOne(Underlying, Excercise, Time, Interest, Volatility, Dividend) -1*Volatility*np.sqrt(Time)
    return dTwo
#Function to calculate NdTwo    

def NdTwo (Underlying, Excercise, Time, Interest, Volatility, Dividend):
    NdTwo=sp.norm.cdf(dTwo(Underlying, Excercise, Time, Interest, Volatility, Dividend))
    return NdTwo

#Function to calculate Premium of a Call Option

def CallOption(Underlying, Exercise, Time, Interest, Volatility, Dividend):
    #CallOption = np.exp(-Dividend * Time) * Underlying * sp.norm.cdf(dOne(Underlying, Exercise, Time, Interest, Volatility, Dividend)) - (Exercise * np.exp(-Interest * Time) * (sp.norm.cdf(dOne(Underlying, Exercise, Time, Interest, Volatility, Dividend)  - (Volatility*np.square(Time)) )))
    CallOption =  np.exp(-Dividend*Time)*Underlying * sp.norm.cdf(dOne(Underlying, Exercise, Time, Interest, Volatility, Dividend)) - (Exercise * np.exp(-Interest * Time) * (sp.norm.cdf(dTwo(Underlying, Exercise, Time, Interest, Volatility, Dividend) ) ))
    return CallOption

#Function to calculate Premium of a Put Option

def PutOption(Underlying, Exercise, Time, Interest, Volatility, Dividend):
    PutOption = np.exp(-Interest * Time) * Exercise *  sp.norm.cdf(-dTwo(Underlying, Exercise, Time, Interest, Volatility, Dividend)) - (Underlying * np.exp(-Dividend * Time)   * (sp.norm.cdf(-dOne(Underlying, Exercise, Time, Interest, Volatility, Dividend))))
    return PutOption

#Function to calculate Delta of a Call Option


def CallDelta(Underlying, Exercise, Time, Interest, Volatility, Dividend):
    CallDelta = sp.norm.cdf(dOne(Underlying, Exercise, Time, Interest, Volatility, Dividend))
    return CallDelta

#Function to calculate Delta of a Put  Option

def PutDelta(Underlying, Exercise, Time, Interest, Volatility, Dividend):
    PutDelta = sp.norm.cdf(dOne(Underlying, Exercise, Time, Interest, Volatility, Dividend)) - 1
    return PutDelta

#Function to calculate Implied Volatility of  a Call Option

def ImpliedCallVolatility(Underlying, Exercise, Time, Interest, Target, Dividend):
    high=5.0
    low=0.0
    
    while (high - low) > 0.001:
        if CallOption(Underlying, Exercise, Time, Interest, (high + low) / 2, Dividend) > Target :
            high = (high + low) / 2
        else: 
            low = (high + low) / 2
    ImpliedCallVolatility = (high + low) / 2    
    return float(ImpliedCallVolatility)

#Function to calculate Implied Volatility of  a Put Option

def ImpliedPutVolatility(Underlying, Exercise, Time, Interest, Target, Dividend):
    high = 5.0
    low = 0.0
    while (high - low) > 0.001:
        if PutOption(Underlying, Exercise, Time, Interest, (high + low) / 2, Dividend) > Target :
            high = (high + low) / 2
        else: 
            low = (high + low) / 2
    ImpliedPutVolatility = (high + low) / 2
    return float(ImpliedPutVolatility)

#Function to calculate Implied Volatility of  a Call/Put Option

def ImpliedVolatility(CEPE, Underlying, Exercise, Time, Interest, Target, Dividend):
    high = 5.0
    low = 0.0

    if CEPE == "CE":
        while (high - low) > 0.001:
              if CallOption(Underlying, Exercise, Time, Interest, (high + low) / 2, Dividend) > Target :
                  high = (high + low) / 2
              else: 
                  low = (high + low) / 2

    else:

        while (high - low) > 0.001:
              if PutOption(Underlying, Exercise, Time, Interest, (high + low) / 2, Dividend) > Target :
                  high = (high + low) / 2
              else: 
                  low = (high + low) / 2
    ImpliedVolatility = (high + low) / 2
    return float(ImpliedVolatility)


def GatherPremium(datadir, symbols):
    sym = {}
    counter=0
   # Open the individual CSV files and read into pandas DataFrames
    for i in symbols:
        
        sym[counter] =pd.DataFrame(pd.read_csv((os.path.join(datadir, '%s.csv' % symbols[counter]))))
        sym[counter]['Time']=pd.to_datetime(sym[counter]['Time'],format='%d/%m/%Y %H:%M')
        sym[counter]['TimeToExp']=(Expiry-sym[counter]['Time'])/np.timedelta64(1,'D')/365
       
            
        counter=counter+1
    return sym




def CalculateDeltas(symbols):
    
    
    for i in symbols:
        symbols[i]['CallDelta']=0
        symbols[i]['PutDelta']=0      
               

    return symbols  
            
    
datadir="H:\Quantexcercises\Practice"

if __name__ == "__main__":    
    

    datadir = "E:\Quantexcercises\Dispersion_data"  # Change this to reflect your data path!
    
    symbols = ('ICICI', 'SBI', 'HDFC','KMB','AXIS','BANKNIFTY')
       
    returns = []

    
def CalculateIV(dataframe,symbols):
    
    df=dataframe
    i=0
    for counter in symbols:
            
            df[i]['IVPut']=0
            df[i]['IVCall']=0
            df[i]['WtAvg']=0
        
            
            try:
              df[i]['IVPut']=np.vectorize(ImpliedPutVolatility)(df[i]['FUT'],df[i]['PutStrike'],df[i]['TimeToExp'],Interest,df[i]['PutPrem'],Dividend)
            except:
              pass
            try:
              df[i]['IVCall']=np.vectorize(ImpliedCallVolatility)(df[i]['FUT'],df[i]['CallStrike'],df[i]['TimeToExp'],Interest,df[i]['CallPrem'],Dividend)
            except:
              pass
            
            
            df[i]['WtAvg']=((df[i]['FUT']-df[i]['PutStrike'])/Strike_Diff[i]*df[i]['IVCall'])+((df[i]['CallStrike'] - df[i]['FUT'])/Strike_Diff[i]*df[i]['IVPut'])
            i=i+1
    return df

         
def CalculateNearestStrike(dataframe,symbols):
    PutStrike1=[]
    PutStrike2=[]
    PutStrike3=[]
    CallStrike1=[]
    CallStrike2=[]
    CallStrike3=[]
    
    i=0
    for counter in symbols:
    
        PutStrike1.append((floor(df[i]['FUT'][0],Strike_Diff[i])-(0*Strike_Diff[i])))
        PutStrike2.append((floor(df[i]['FUT'][0],Strike_Diff[i])-(1*Strike_Diff[i])))
        PutStrike3.append((floor(df[i]['FUT'][0],Strike_Diff[i])-(2*Strike_Diff[i])))
        CallStrike1.append((ceil(df[i]['FUT'][0],Strike_Diff[i])+(0*Strike_Diff[i])))
        CallStrike2.append((ceil(df[i]['FUT'][0],Strike_Diff[i])+(1*Strike_Diff[i])))
        CallStrike3.append((ceil(df[i]['FUT'][0],Strike_Diff[i])+(2*Strike_Diff[i])))
        
        df[i]['PutValue1']=0
        df[i]['PutValue2']=0
        df[i]['PutValue3']=0
        df[i]['CallValue1']=0
        df[i]['CallValue2']=0
        df[i]['CallValue3']=0
        
        try:
    
            df[i]['PutValue1']=df[i][str(PutStrike1[i])+"PE"]
        except:
            pass
        try:
            df[i]['PutValue2']=df[i][str(PutStrike2[i])+"PE"]
        except:
            pass
        try:
            df[i]['PutValue3']=df[i][str(PutStrike3[i])+"PE"]
        except:
            pass
        try:
            df[i]['CallValue1']=df[i][str(CallStrike1[i])+"CE"]
        except:
            pass
        try:
            df[i]['CallValue2']=df[i][str(CallStrike2[i])+"CE"]
        except:
            pass
        try:
            df[i]['CallValue3']=df[i][str(CallStrike3[i])+"CE"]
        except:
            pass
        
        df[i]['PutValue1'].fillna(0,inplace=True)
        df[i]['PutValue2'].fillna(0,inplace=True)
        df[i]['PutValue3'].fillna(0,inplace=True)
        df[i]['CallValue1'].fillna(0,inplace=True)
        df[i]['CallValue2'].fillna(0,inplace=True)
        df[i]['CallValue3'].fillna(0,inplace=True)
        i=i+1
    return df,PutStrike1,PutStrike2,PutStrike3,CallStrike1,CallStrike2,CallStrike3

def CalculateNearestIV(dataframe,PutStrike1,PutStrike2,PutStrike3,CallStrike1,CallStrike2,CallStrike3):
    
    i=0
    for counter in symbols:
      df[i]['ImpVolPut1']=np.vectorize(ImpliedPutVolatility)(df[i]['FUT'],PutStrike1[i],df[i]['TimeToExp'],Interest,df[i]['PutValue1'],Dividend)
      df[i]['ImpVolPut2']=np.vectorize(ImpliedPutVolatility)(df[i]['FUT'],PutStrike2[i],df[i]['TimeToExp'],Interest,df[i]['PutValue2'],Dividend)
      df[i]['ImpVolPut3']=np.vectorize(ImpliedPutVolatility)(df[i]['FUT'],PutStrike3[i],df[i]['TimeToExp'],Interest,df[i]['PutValue3'],Dividend)
      
      df[i]['ImpVolCall1']=np.vectorize(ImpliedCallVolatility)(df[i]['FUT'],CallStrike1[i],df[i]['TimeToExp'],Interest,df[i]['CallValue1'],Dividend)
      df[i]['ImpVolCall2']=np.vectorize(ImpliedCallVolatility)(df[i]['FUT'],CallStrike2[i],df[i]['TimeToExp'],Interest,df[i]['CallValue2'],Dividend)
      df[i]['ImpVolCall3']=np.vectorize(ImpliedCallVolatility)(df[i]['FUT'],CallStrike3[i],df[i]['TimeToExp'],Interest,df[i]['CallValue3'],Dividend)
      i=i+1
    
    return df,PutStrike1,PutStrike2,PutStrike3,CallStrike1,CallStrike2,CallStrike3

def CalculateDeltas(dataframe,symbols,PutStrike1,PutStrike2,PutStrike3,CallStrike1,CallStrike2,CallStrike3):
    
    i=0
    for counter in symbols:
        df[i]['PutDelta1'] =np.vectorize(PutDelta)(df[i]['FUT'],PutStrike1[i],df[i]['TimeToExp'],Interest,df[i]['ImpVolPut1'],Dividend)
        df[i]['PutDelta2'] =np.vectorize(PutDelta)(df[i]['FUT'],PutStrike2[i],df[i]['TimeToExp'],Interest,df[i]['ImpVolPut2'],Dividend)
        df[i]['PutDelta3'] =np.vectorize(PutDelta)(df[i]['FUT'],PutStrike3[i],df[i]['TimeToExp'],Interest,df[i]['ImpVolPut3'],Dividend)
    
        df[i]['CallDelta1'] =np.vectorize(CallDelta)(df[i]['FUT'],CallStrike1[i],df[i]['TimeToExp'],Interest,df[i]['ImpVolCall1'],Dividend)
        df[i]['CallDelta2'] =np.vectorize(CallDelta)(df[i]['FUT'],CallStrike2[i],df[i]['TimeToExp'],Interest,df[i]['ImpVolCall2'],Dividend)
        df[i]['CallDelta3'] =np.vectorize(CallDelta)(df[i]['FUT'],CallStrike3[i],df[i]['TimeToExp'],Interest,df[i]['ImpVolCall3'],Dividend)
        
        
        df[i]['SumDelta']=df[i]['PutDelta1']+df[i]['PutDelta2']+df[i]['PutDelta3']+df[i]['CallDelta1']+df[i]['CallDelta2']+df[i]['CallDelta3']
        df[i]['SumDelta'].fillna(0,inplace=True)
        
        i=i+1
    return df 
#Corel_Weight

def CalculateDirtyCoRel(dataframe,symbols):
    i=0;Ratio=0;Vol_IndSecurities=0
    df=dataframe
    for i in range(0,len(symbols)-1):
        Vol_IndSecurities=Vol_IndSecurities+(df[i]['WtAvg']*Corel_Weight[i])
        i=i+1
    Ratio=np.divide(df[len(symbols)-1]['WtAvg'],(Vol_IndSecurities))
    CoRel=np.square(Ratio)
    return Vol_IndSecurities,CoRel

def Generate_Signals(df,symbols,CoRel,entry_t1=0.2,entry_t2=0.8,exit_t=0.5):
        i=0
        
        if CoRel[0]<entry_t1 :
            position=-1 #Short Seucities
        if CoRel[0]>entry_t2 :
            position=1#Short Index
        j=16    
        for i in range(0,len(symbols)-1):
            df[i]['position']=0
            df[i]['position'][j]=1
            
        i=i+1
        df[i]['position']=0
        df[i]['position'][j]=-df[i-1]['position'][j]
        
        return df  

def Positions(df,symbols):
    i=0
    
    for i in range(0,len(symbols)):
     j=16      #First trade, static value passed
     df[i]['initial_delta']=df[i]['position'][j]*df[i]['SumDelta']*Lots_Bought[i]*Lot_Size[i]
        

     
     k=1914
     df[i]['total_fut_inpos']=0
     df[i]['future_buy/sell']=0
     df[i]['extraInvestment']=0  
     df[i]['final_delta']=0
     #for b in df[i].iterrows():
     for b in range (j,len(df[i])-1):
          df[i].loc[j,'total_fut_inpos']=df[i].loc[j-1,'total_fut_inpos']+ df[i].loc[j-1,'future_buy/sell']
          df[i].loc[j,'final_delta']=df[i].loc[j,'initial_delta']+df[i].loc[j,'total_fut_inpos']
          df[i].loc[j,'future_buy/sell']=np.where(df[i].loc[j,'final_delta']> Lot_Size[i],-floor(df[i].loc[j,'final_delta'],Lot_Size[i]),np.where(df[i].loc[j,'final_delta']< -Lot_Size[i],floor(-df[i].loc[j,'final_delta'],Lot_Size[i]),0))
          #df[i].loc[j,'future_buy/sell']=np.where(df[i].loc[j,'final_delta']< -Lot_Size[i],floor(-df[i].loc[j,'final_delta'],Lot_Size[i]),0)
          
          j=j+1    
     df[i]['extraInvestment']=df[i]['future_buy/sell']*df[i]['FUT']
     
     
    return df
        
def PNL(df,symbols):
    Profit=0
    j=16
    k=1914
    FutSet=[]
    Hedging=[]
    pro=[]
    for i in range(0,len(symbols)):
        df[i]['initPut']=df[i]['initCall']=df[i]['inittrade']=df[i]['sqofPut']=df[i]['sqofCall']=df[i]['sqoftrade']=0
        df[i]['initPut'][j]=np.sum([df[i]['PutValue1'][j],df[i]['PutValue2'][j],df[i]['PutValue3'][j]])
        df[i]['initCall'][j]=np.sum([df[i]['CallValue1'][j],df[i]['CallValue2'][j],df[i]['CallValue3'][j]])
        df[i]['inittrade'][j]=np.sum([df[i]['initPut'][j],df[i]['initCall'][j]])*Lot_Size[i]*-1*df[i]['position'][j]*Lots_Bought[i]
        df[i]['sqofPut'][k]=np.sum([df[i]['PutValue1'][k],df[i]['PutValue2'][k],df[i]['PutValue3'][k]])
        df[i]['sqofCall'][k]=np.sum([df[i]['CallValue1'][k],df[i]['CallValue2'][k],df[i]['CallValue3'][k]])
        df[i]['sqoftrade'][k]= np.sum([df[i]['sqofPut'][k],df[i]['sqofCall'][k]])*Lots_Bought[i]*Lot_Size[i]*df[i]['position'][j]
        FutSet.append(df[i]['future_buy/sell'].sum()*df[i]['FUT'][k])
        Hedging.append(-1*df[i]['extraInvestment'].sum())
        pro.append(df[i]['inittrade'][j]+df[i]['sqoftrade'][k]+Hedging[i]+FutSet[i])
        Profit=Profit+df[i]['inittrade'][j]+df[i]['sqoftrade'][k]+Hedging[i]+FutSet[i]
        
    return FutSet,Hedging,Profit,pro




if __name__ == "__main__":    
    
        
    returns = []

    print "Gathering Premiums..."
    df = GatherPremium(datadir, symbols)
    print "Calculating Volatility of Nearest Strikes..."
    df = CalculateIV(df, symbols)
    print "Calculating Dirty Corelation"
    Vol_IndSecurities,Corel=CalculateDirtyCoRel(df,symbols)
    print "Identifying Nearest Strikes"
    df,ps1,ps2,ps3,cs1,cs2,cs3=CalculateNearestStrike(df,symbols)
    print "Calculating Nearest IVs"
    df,ps1,ps2,ps3,cs1,cs2,cs3=CalculateNearestIV(df,ps1,ps2,ps3,cs1,cs2,cs3)
    print "Calculating Deltas of Securities"
    df=CalculateDeltas(df,symbols,ps1,ps2,ps3,cs1,cs2,cs3)
    print "Generating Signals"
    df=Generate_Signals(df,symbols,Corel)
    print "Calculating Positions"
    df=Positions(df,symbols)
    print "Calculating PNL"
    futset,hed,profit,pro=PNL(df,symbols)
    print "Net Profit Resulted is " + str(profit)
    totaltime=time.time() - start_time
    minutes,seconds=divmod(totaltime,60)
    print "Execution time is " + str(minutes) + "   minutes and " + str(int(seconds)) + " seconds"
    Corel.plot(grid=True)   