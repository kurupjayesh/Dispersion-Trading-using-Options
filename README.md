#Dispersion Trading using Options

#Introduction

The Dispersion Trading is a strategy used to exploit the difference between implied volatility and its subsequent realized volatility. The dispersion trading uses the fact that difference between implied and realized volatility is greater between index options than between individual stock options. A trader could therefore sell options on index and buy individual stocks options or vice versa based on this volatility difference. Dispersion trading is a sort of correlation trading as trades are usually profitable in a time when the individual stocks are not strongly correlated and loses money during stress periods when correlation rises.  
The correlation among the securities are used as a factor to determine the entry of a trade. Depending on the value of correlation between individual stocks, the dispersion can be traded by selling the index options and buying options on index components or by buying index options and selling options on index components. The most well-received theory for the profitability of this strategy is market inefficiency which states that supply and demand in the options market drive the premiums which deviate from their theoretical value. 

#Strategy

To distinguish dispersion trading, it is simply a hedged strategy which takes advantage of relative value differences in implied volatilities between an index and index component stocks. It involves a short options positions on securities of index and a long options positions on the components of the index or vice versa. Effectively we will be longing/shorting straddle’s based on our entry signals.
We have to note that this trade would be successful only when the delta exposure is close to zero. Thus the dispersion strategy is hedged against large market movements.

#Below is the progression of actions to be taken for a successful Dispersion Trade

•	Calculate the Dirty Correlation (ρ):
•	Generate Signals when Dirty Correlation crosses threshold/reaches its extreme 
•	Buy/Sell Index and individual securities as per the logic.
•	Compute Delta at regular intervals and offset them by buying/selling futures to make the trade delta neutral
•	Exit when Dirty Correlation crosses the mean (ρ=0.5)
To understand how the difference in volatility is captured, we need to understand 
The variance of index with a basket of Stocks with weight wi is given as:

 

where,
σI2  is the index variance 
wi is the weight for stock in the index. 
σi2 is the individual stock variance
ρij is the correlation of stock i with stock j.

The profit from this strategy comes from the fact that correlation tends to mean revert. Thus if one takes positions during the extremes of ratio, we can be assured that it would mean revert at a certain point.

#Implementing the strategy

To implement the strategy, we would be needing to calculate the below:
Calculating Implied volatility of nearest Strikes
Since we would be having the Premiums, time to expiry, Interest Rate, Dividend and the nearest Strike, we can compute the Implied Volatility of the nearest strikes using the Black Scholes model. The weighted average Implied Volatility among the nearest strikes needs to be added for the individual securities and Index in order to calculate the Correlation
Dirty Correlation:
This is the square of ratio of the Implied Volatility of Index and Weighted average of Stocks. Thus the formula would be
 

#Defining Thresholds:

This is an important step to generate entry/exit signals based on the risk appetite.
In this project, the thresholds are z1=0.2, z2=0.8, z3=0.5
where 
z1 gives the signal to buy Index and short the individual securities
z2 gives the signal to short the Index and long individual securities
z3 is the exit threshold where all positions are to be squared off

#Choosing option to be traded

As soon as we get the signal to buy/sell, we would be using a combination of straddle and strangle of both puts and calls. The nearest 3 OTM strikes are considered in this project. The investment amount needs to be equally split amongst the Index and Individual securities. While taking an entry the lot size, the quantity bought needs to be noted down so that the deltas at each stage is handy.

#Hedging

This is further hedged using future contracts to keep the whole process delta neutral. Delta of this strategy should be adjusted every fifteen minutes when the delta went above 1, one future contract was sold and when the delta dropped to -1 the delta was neutralized by buying one future contract. It is important to keep the delta close to zero for the duration of the trade.
Profit and loss calculation
To calculate the PNL the following four things needs to be considered:
•	Initial trade, cost at which options were bought/sold when entry was taken
•	Hedging cost, the total amount of futures invested to make the trade delta neutral
•	Future Settlement, the amount resulted in settling the Future at the exit signal
•	Square of Options, the amount resulted when all positions were squared off at exit signal

#Conclusion

Dispersion trading is complex strategy however this is rewarded with the strategy being a profitable one which offers high rewards in response to a low risk.
To make this strategy even better to use it would be necessary if the strategy is automated and that the hedging should be dynamic as per the price movements. 
Trading at times where volatility is high (viz. quarterly results, individual stock news etc) when the correlation would not be strong may result in more profit.
To maximize the accuracy of the strategy, we can decrease the time interval to capture the volatility and accordingly compute deltas.

