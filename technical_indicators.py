"""
Table of Contents of TA_package:

1. Simple Moving Average(df, n) (SMA)
2. Exponential Moving Average(df, n) (EMA)
3. Momentum(df, n) (MOM)
4. Rate of Change(df, n) (ROC)
5. Average True Range(df, n) (ATR)
6. Bollinger Bands(df, n, num_of_std) (BBANDS)
7. Pivot Points, Support & Resistance(df) (PPSR)
8. Stochastic oscillator & K (df) (STOK)
9. Stochastic oscillator (df, n) (STO)
10. TRIX (df, n) (TRIX)
11. Average Directional Movement Index(df, n, n_ADX) (ADX)
12. MACD, MACD Signal and MACD difference(df, n_fast, n_slow) (MACD)
13. Mass Index(df) (MassI)
14. Vortex Indicator(df, n) (Vortex)
15. KST Oscillator(df, r1, r2, r3, r4, n1, n2, n3, n4) (KST)
16. Relative Strength Index(df, n) (RSI)
17. True Strength Index(df, r, s) (TSI)
18. Accumulation/Distribution (df, n) (ACCDIST)
19. Chaikin Oscillator (df) (Chaikin)
20. Money Flow Index and Ratio(df, n) (MFI)
21. On-balance Volume(df, n) (OBV)
22. Force Index (df, n) (FORCE)
23. Ease of Movement(df, n) (EOM)
24. Commodity Channel Index(df, n) (CCI)
25. Coppock Curve(df, n) (COPP)
26. Keltner Channel(df, n) (KELCH)
27. Ultimate Oscillator(df) (ULTOSC)
28. Donchian Channel(df, n) (DONCH)
29. Standard Deviation(df, n) (STDDEV)

"""


"""
For now major strategies 

"""

# Done
def MA(df, n): # n = 5
    """
    Moving Average
    rationale CHECKED, code CHECKED. updated

    params:
        df: pd dataframe 
        n: number of days = 5
    """
    MA = df['Close'].rolling(window=n, center=False).mean().rename('MA_' + str(n))
    return MA


# Done
def EMA(df, n): # n = 5
    """
    Exponential Moving Average
    rationale CHECKED, code CHECKED, updated.

    params:
        df: pd dataframe
        n: number of days = 5
    """
    EMA = df['Close'].ewm(span=n, min_periods=n - 1).mean().rename('EMA_' + str(n))
    return EMA

# Done
def MOM(df, n): # n = 5
    """
    Momentum, Diff of prices between the current close and the past n day(s) of close price
    rationale CHECKED, code CHECKED, it involves n+1 numbers.

    params:
        df: pd dataframe
        n: number of days = 5
    """
    MOM = pd.Series(df['Close'].diff(n), name='Momentum_' + str(n))
    return MOM

# Done
def ROC(df, n): # n = 5
    """
    The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum, 
    is a pure momentum oscillator that measures the percent change in price from one period to the next. 
    The ROC calculation compares the current price with the price _ün_ü periods ago. 
    The plot forms an oscillator that fluctuates above and below the zero line as the Rate-of-Change 
    moves from positive to negative. As a momentum oscillator, ROC signals include centerline crossovers, 
    divergences and overbought-oversold readings. Divergences fail to foreshadow reversals more often 
    than not so this article will forgo a discussion on divergences. Even though centerline crossovers 
    are prone to whipsaw, especially short-term, these crossovers can be used to identify the overall trend. 
    Identifying overbought or oversold extremes comes naturally to the Rate-of-Change oscillator.
    
    ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:rate_of_change_roc_and_momentum
    """
    
    """
    Rate of Change, rationale CHECKED, code CHECKED
    Category: meansuring momentum
    How this works high-level:
    """
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    return ROC

# Done
def ATR(df, n): # n = 14
    
    """
    Typically, the Average True Range (ATR) is based on 14 periods and can be calculated on an intraday, 
    daily, weekly or monthly basis. For this example, the ATR will be based on daily data. 
    Because there must be a beginning, the first TR value is simply the High minus the Low, 
    and the first 14-day ATR is the average of the daily TR values for the last 14 days. 
    After that, Wilder sought to smooth the data by incorporating the previous period's ATR value.
    
    Current ATR = [(Prior ATR x 13) + Current TR] / 14

      - Multiply the previous 14-day ATR by 13.
      - Add the most recent day's TR value.
      - Divide the total by 14
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr
    """
    
    
    """
    Average True Range, rationale CHECKED, code CHECKED, updated
    Category: measuring volatility
    How this works high-level: 
        Calculate the most recent day's largest range considering yesterday's close, today's
        high and today's low
    """
    df = df.reset_index()
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) \
            - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = TR_s.ewm(span = n, min_periods = n).mean().rename('ATR_' + str(n))  
    df = df.join(ATR).set_index("index")
    return df["ATR_" + str(n)]


# Done
def BBANDS(df, n, num_of_std): # n = 20
    """
    Bollinger Bands, rationale CHECKED, code CHECKED
    Create the common bbands which I understand instead of using the one from Quantopian forum.
    """
    rlng_mean = df['Close'].ewm(span = n).mean().rename('Rolling_{}_Day_Mean'.format(n))
    rlng_std  = df['Close'].rolling(window = n).std()
    u_band = (rlng_mean + rlng_std*num_of_std).rename('U_{}d_{}_std_Band'.format(n, num_of_std))
    l_band = (rlng_mean - rlng_std*num_of_std).rename('L_{}d_{}_std_Band'.format(n, num_of_std))
    return rlng_mean, u_band, l_band

# Done
# Rationale checked
def MACD(df, n_fast, n_slow): # n_fast = 12, n_slow = 26
    """
    
    http://stockcharts.com/docs/doku.php?id=scans:indicators

    """
    
    
    """
    MACD, MACD Signal and MACD difference, rationale CHECKED, code CHECKED, updated
    # Conventional look-back window for calculating MACDsign is 9
    """
    EMAfast = df['Close'].ewm(span = n_fast, min_periods = n_fast - 1).mean()
    EMAslow = df['Close'].ewm(span = n_slow, min_periods = n_slow - 1).mean()
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = MACD.ewm(span = 9, min_periods = 8).mean().rename('MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    return MACD, MACDsign, MACDdiff

# No parameters,
# Still need to understand more
def PPSR(df):
    """
        Pivot Points, Supports and Resistances
        
        http://stockcharts.com/docs/doku.php?id=scans:indicators
    """
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3, name="PP")
    R1 = pd.Series(2 * PP - df['Low'], name="R1")
    S1 = pd.Series(2 * PP - df['High'], name="S1")
    R2 = pd.Series(PP + df['High'] - df['Low'], name="R2")
    S2 = pd.Series(PP - df['High'] + df['Low'], name="S2")
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']), name="R3")
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP), name="S3")
    return PP, R1, S1, R2, S2, R3, S3

# Done
# Rationale checked
def RSI(df, n): # n = 14
    """
    Relative Strength Index, updated.
    Conventional parameters: n = 14, 0.3 and 0.7 are two conventional thresholds
    """
    df = df.reset_index()
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.at[i + 1, 'High'] - df.at[i, 'High']
        DoMove = df.at[i, 'Low'] - df.at[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI   = pd.Series(UpI)
    DoI   = pd.Series(DoI)
    PosDI = UpI.ewm(span = n, min_periods = n - 1).mean()
    NegDI = DoI.ewm(span = n, min_periods = n - 1).mean()
    RSI   = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
    df    = df.join(RSI).set_index("index")
    return df["RSI_" + str(n)]


# Done
# Rationale checked
def MFI(df, n): # n = 14
    """Money Flow Index and Ratio, updated.
    http://stockcharts.com/docs/doku.php?id=scans:indicators#money_flow_index_mfi
    
    """
    df = df.reset_index()
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i  = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.at[i + 1, 'Volume'])
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR   = pd.Series(PosMF / TotMF)
    MFI   = pd.Series(MFR.rolling(window = n, center = False).mean(), name = 'MFI_' + str(n))
    df    = df.join(MFI).set_index("index")
    return df["MFI_" + str(n)]

# Done, no parameters
def STOK(df):
    """Stochastic oscillator %K (Fast Stoch%K)
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full
    """
    STOK = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    return STOK

# Done
def STO(df, n): # n =14
    """Stochastic oscillator %D, updated
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full
    """
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    STO = SOk.ewm(span=n, min_periods=n - 1).mean().rename('SO%d_' + str(n)) 
    return STO

# Done
def TRIX(df, n): # n = 15
    """Trix, updated
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix
    """
    df = df.reset_index()
    EX1 = df['Close'].ewm(span = n, min_periods = n - 1).mean()
    EX2 = EX1.ewm(span = n, min_periods = n - 1).mean()
    EX3 = EX2.ewm(span = n, min_periods = n - 1).mean()
    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))
    df = df.join(Trix).set_index("index")
    return df["Trix_" + str(n)]

# Done
def ADX(df, n, n_ADX): # n = 14 (for buildinhg ATR in this function), n_ADX = 14 (for building final ADX in this func)
    """
    The Average Directional Index (ADX), Minus Directional Indicator (-DI) and Plus Directional Indicator (+DI) 
    represent a group of directional movement indicators that form a trading system developed by Welles Wilder. 
    Although Wilder designed his Directional Movement System with commodities and daily prices in mind, these 
    indicators can also be applied to stocks.
    
    Positive and negative directional movement form the backbone of the Directional Movement System. Wilder 
    determined directional movement by comparing the difference between two consecutive lows with the difference 
    between their respective highs.
    
    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI) are derived from smoothed averages 
    of these differences, and measure trend direction over time. These two indicators are often referred to 
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed averages of the difference between 
    +DI and -DI, and measures the strength of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the direction and strength of the trend.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    """
    
    df = df.reset_index()
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.at[i + 1, 'High'] - df.at[i, 'High']
        DoMove = df.at[i, 'Low'] - df.at[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpI.append(UpMove)
        else:
            UpI.append(0)
        
        if DoMove > UpMove and DoMove > 0:
            DoI.append(DoMove)
        else:
            DoI.append(0)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) \
            - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])
        TR_l.append(TR)
        i = i + 1
    TR_s  = pd.Series(TR_l)
    ATR   = TR_s.ewm(span=n, min_periods=n).mean()
    UpI   = pd.Series(UpI)
    DoI   = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n - 1).mean() / ATR)
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n - 1).mean() / ATR)
    ADX   = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)).ewm(span=n_ADX, min_periods=n_ADX - 1).mean(), 
                    name = 'ADX_' + str(n) + '_' + str(n_ADX))
    df = df.join(ADX).set_index("index")
    return df['ADX_' + str(n) + '_' + str(n_ADX)]

# Done, default parameter setup = 9 and 25 
def MassI(df):
    """
    Developed by Donald Dorsey, the Mass Index uses the high-low range to identify trend reversals based on 
    range expansions. In this sense, the Mass Index is a volatility indicator that does not have a directional 
    bias. Instead, the Mass Index identifies range bulges that can foreshadow a reversal of the current trend.
    
    Single EMA = 9-period exponential moving average (EMA) of the high-low differential  
    Double EMA = 9-period EMA of the 9-period EMA of the high-low differential 
    EMA Ratio = Single EMA divided by Double EMA 
    Mass Index = 25-period sum of the EMA Ratio 
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index
    """
    
    """Mass Index, updated"""
    Range = df['High'] - df['Low']
    EX1 = Range.ewm(span=9, min_periods=8).mean()
    EX2 = EX1.ewm(span=9, min_periods=8).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(Mass.rolling(window=25, center=False).sum(), name='Mass Index')
    return MassI

# Done
def Vortex(df, n): # n = 14
    """
    The Vortex Indicator (VTX) can be used to identify the start of a trend and subsequently affirm trend 
    direction. First, a simple cross of the two oscillators can be used to signal the start of a trend. 
    After this crossover, the trend is up when +VI is above -VI and down when -VI is greater than +VI. 
    Second, a cross above or below a particular level can signal the start of a trend and these levels can 
    be used to affirm trend direction.
    
    Positive and negative trend movement:

    +VM = Current High less Prior Low (absolute value)
    -VM = Current Low less Prior High (absolute value)

    +VM14 = 14-period Sum of +VM
    -VM14 = 14-period Sum of -VM


    True Range (TR) is the greatest of:

      * Current High less current Low
      * Current High less previous Close (absolute value)
      * Current Low less previous Close (absolute value)

    TR14 = 14-period Sum of TR


    Normalize the positive and negative trend movements:

    +VI14 = +VM14/TR14
    -VI14 = -VM14/TR14
    
    
    Vortex Indicator 
    http://www.vortexindicator.com/VFX_VORTEX.PDF
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator
    """
    df = df.reset_index()
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.at[i + 1, 'High'], df.at[i, 'Close']) \
                - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.at[i + 1, 'High'] - df.at[i, 'Low']) \
                - abs(df.at[i + 1, 'Low'] - df.at[i, 'High'])
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.Series(VM).rolling(window=n).sum() / pd.Series(VM).rolling(window=n).sum(), 
                   name='Vortex_' + str(n))
    df = df.join(VI).set_index("index")
    return df["Vortex_" + str(n)]


# Done
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4): # 10, 15, 20, 30, 10, 10, 10, 15
    """
    KST Oscillator, updated
    Even though the formula for KST looks complicated, it is a rather straightforward indicator. 
    It is simply a weighted average of four different rate-of-change values that have been smoothed.
    
    The default parameters are as follows: KST(10,15,20,30,10,10,10,15,9). 
    The first four numbers represent the rate-of-change settings, the second four represent the 
    moving averages for these rate-of-change indicators and the last number is the signal line moving average.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst
    
    """
    M    = df['Close'].diff(r1 - 1)
    N    = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M    = df['Close'].diff(r2 - 1)
    N    = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M    = df['Close'].diff(r3 - 1)
    N    = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M    = df['Close'].diff(r4 - 1)
    N    = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST  = pd.Series(ROC1.rolling(window=n1).sum() + 
                    ROC2.rolling(window=n2).sum() * 2 + 
                    ROC3.rolling(window=n3).sum() * 3 + 
                    ROC4.rolling(window=n4).sum() * 4, 
                name = 'KST_{}_{}_{}_{}_{}_{}_{}_{}'.format(r1, r2, r3, r4, n1, n2, n3, n4))
    return KST


# Done
def TSI(df, r, s): # r = 25, s = 13
    """True Strength Index, updated
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:true_strength_index
    """
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1  = pd.Series(M.ewm(span=r, min_periods=r - 1).mean())
    aEMA1 = pd.Series(aM.ewm(span=r, min_periods=r - 1).mean())
    EMA2  = pd.Series(EMA1.ewm(span=s, min_periods=s - 1).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span=s, min_periods=s - 1).mean())
    TSI   = pd.Series(EMA2 / aEMA2, name = 'TSI_{}_{}'.format(r, s))
    return TSI


# Discretionarily Done
def ACCDIST(df, n): # n = 5
    """Accumulation/Distribution
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:accumulation_distribution_line
    """
    ad  = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M   = ad.diff(n - 1)
    N   = ad.shift(n - 1)
    ROC = M / N
    AD  = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))
    return AD


# Done, default parameter setup = 3, 10
def Chaikin(df):
    """Chaikin Oscillator
    
    Developed by Marc Chaikin, the Chaikin Oscillator measures the momentum of the Accumulation Distribution 
    Line using the MACD formula. This makes it an indicator of an indicator. The Chaikin Oscillator is the 
    difference between the 3-day EMA of the Accumulation Distribution Line and the 10-day EMA of the 
    Accumulation Distribution Line. Like other momentum indicators, this indicator is designed to anticipate 
    directional changes in the Accumulation Distribution Line by measuring the momentum behind the movements. 
    A momentum change is the first step to a trend change. Anticipating trend changes in the Accumulation 
    Distribution Line can help chartists anticipate trend changes in the underlying security. The Chaikin 
    Oscillator generates signals with crosses above/below the zero line or with bullish/bearish divergences.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_oscillator
    """
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series(ad.ewm(span=3, min_periods=2).mean() \
                        - ad.ewm(span=10, min_periods=9).mean(), name = 'Chaikin')
    return Chaikin

# Discretionarily Done
def OBV(df, n): # n = 5
    """On-balance Volume
    
    On Balance Volume (OBV) measures buying and selling pressure as a cumulative indicator that adds 
    volume on up days and subtracts volume on down days. OBV was developed by Joe Granville and introduced 
    in his 1963 book, Granville's New Key to Stock Market Profits. It was one of the first indicators to 
    measure positive and negative volume flow. Chartists can look for divergences between OBV and price 
    to predict price movements or use OBV to confirm price trends.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:on_balance_volume_obv
    """
    df = df.reset_index()
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] > 0:
            OBV.append(df.at[i + 1, 'Volume'])
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] == 0:
            OBV.append(0)
        if df.at[i + 1, 'Close'] - df.at[i, 'Close'] < 0:
            OBV.append(-df.at[i + 1, 'Volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(window=n).mean(), name = 'OBV_' + str(n))
    return OBV_ma

# Done
def FORCE(df, n): # n = 13
    """Force Index
    
    The Force Index combines all three as an oscillator that fluctuates in positive and negative 
    territory as the balance of power shifts. The Force Index can be used to reinforce the overall 
    trend, identify playable corrections or foreshadow reversals with divergences.
    
    Force Index(1) = {Close (current period)  -  Close (prior period)} x Volume
    Force Index(13) = 13-period EMA of Force Index(1)
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index
    """
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = 'Force_' + str(n))
    return F


# Done
# Rationale checked: helps identify strength of the trend
def EOM(df, n): # n = 14
    """Ease of Movement
    
    Developed by Richard Arms, Ease of Movement (EMV) is a volume-based oscillator that fluctuates above 
    and below the zero line. As its name implies, it is designed to measure the _üease_ü of price movement. 
    Arms created Equivolume charts to visually display price ranges and volume. Ease of Movement takes 
    Equivolume to the next level by quantifying the price/volume relationship and showing the results 
    as an oscillator. In general, prices are advancing with relative ease when the oscillator is in 
    positive territory. Conversely, prices are declining with relative ease when the oscillator is in 
    negative territory.
    
    Distance Moved = ((H + L)/2 - (Prior H + Prior L)/2) 
    Box Ratio = ((V/100,000,000)/(H - L))
    1-Period EMV = ((H + L)/2 - (Prior H + Prior L)/2) / ((V/100,000,000)/(H - L))
    14-Period Ease of Movement = 14-Period simple moving average of 1-period EMV
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ease_of_movement_emv
    """
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = EoM.rolling(window=n).mean().rename('EoM_' + str(n))
    return Eom_ma

# Done
# Rationale checked. 
def CCI(df, n): # n = 20
    """
    Commodity Channel Index
    Developed by Donald Lambert and featured in Commodities magazine in 1980, the Commodity Channel Index (CCI) 
    is a versatile indicator that can be used to identify a new trend or warn of extreme conditions. 
    Lambert originally developed CCI to identify cyclical turns in commodities, but the indicator can 
    be successfully applied to indices, ETFs, stocks, and other securities. In general, CCI measures the 
    current price level relative to an average price level over a given period of time. CCI is relatively 
    high when prices are far above their average. CCI is relatively low when prices are far below their average. 
    In this manner, CCI can be used to identify overbought and oversold levels.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    
    """
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    # This is rolling z-score
    CCI = pd.Series((PP - PP.rolling(window = n).mean()) / PP.rolling(window = n).std(), name = 'CCI_' + str(n))    
    return CCI


# Done
# Rationale checked
def COPP(df, n): # n = 10
    """Coppock Curve
    
    The Coppock Curve is a momentum indicator developed by Edwin _üSedge_ü Coppock, who was an economist 
    by training. Coppock introduced the indicator in Barron's in October 1965. The goal of this indicator 
    is to identify long-term buying opportunities in the S&P 500 and Dow Industrials. The signal is very simple. 
    Coppock used monthly data to identify buying opportunities when the indicator moved from negative territory 
    to positive territory. Although Coppock did not use it for sell signals, many technical analysts consider 
    a cross from positive to negative territory as a sell signal.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve
    
    """
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = (ROC1 + ROC2).ewm(span=n, min_periods=n).mean().rename("Copp_" + str(n))
    return Copp


# Done
# Rationale checked
def KELCH(df, n): # n = 20
    """Keltner Channel
    Keltner Channels are volatility-based envelopes set above and below an exponential moving average. 
    This indicator is similar to Bollinger Bands, which use the standard deviation to set the bands. 
    Instead of using the standard deviation, Keltner Channels use the Average True Range (ATR) to 
    set channel distance. The channels are typically set two Average True Range values above and below 
    the 20-day EMA. The exponential moving average dictates direction and the Average True Range sets 
    channel width. Keltner Channels are a trend following indicator used to identify reversals with channel 
    breakouts and channel direction. Channels can also be used to identify overbought and oversold levels 
    when the trend is flat.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:keltner_channels
    """
    KelChM = ((df['High'] + df['Low'] + df['Close']) / 3).rolling(window=n).mean().rename('KelChM_' + str(n))
    KelChU = ((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3).rolling(window=n).mean().rename('KelChU_' + str(n))
    KelChD = ((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3).rolling(window=n).mean().rename('KelChD_' + str(n))
    return KelChM, KelChU, KelChD


# Done, default parameter setup = 7, 14, 28 
def ULTOSC(df):
    """Ultimate Oscillator
    
    Developed by Larry Williams in 1976 and featured in Stocks & Commodities Magazine in 1985, 
    the Ultimate Oscillator is a momentum oscillator designed to capture momentum across three 
    different timeframes. The multiple timeframe objective seeks to avoid the pitfalls of other 
    oscillators. Many momentum oscillators surge at the beginning of a strong advance and then 
    form a bearish divergence as the advance continues. This is because they are stuck with one 
    timeframe. The Ultimate Oscillator attempts to correct this fault by incorporating longer 
    timeframes into the basic formula. Williams identified a buy signal a based on a bullish 
    divergence and a sell signal based on a bearish divergence.
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator
    
    """
    df = df.reset_index()
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.at[i + 1, 'High'], df.at[i, 'Close']) - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])
        TR_l.append(TR)
        BP = df.at[i + 1, 'Close'] - min(df.at[i + 1, 'Low'], df.at[i, 'Close'])
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.Series(BP_l).rolling(window=7).sum() / pd.Series(TR_l).rolling(window=7).sum()) \
                     + (2 * pd.Series(BP_l).rolling(window=14).sum() / pd.Series(TR_l).rolling(window=14).sum()) \
                     + (pd.Series(BP_l).rolling(window=28).sum() / pd.Series(TR_l).rolling(window=28).sum()),
                     name = 'Ultimate_Osc')
    df = df.join(UltO).set_index("index")
    return df["Ultimate_Osc"]


# TBD
def DONCH(df, n): # n = 5
    """Donchian Channel"""
    df = df.reset_index()
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < df.index[-1]:
        DC = max(df['High'].iloc[i:i + n - 1]) - min(df['Low'].iloc[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))
    DonCh = DonCh.shift(n - 1)
    df = df.join(DonCh).set_index("index")
    return df["Donchian_" + str(n)]

# Done, rationale checked
def STDDEV(df, n): # n = 10
    """Standard Deviation, i.e. historical volatility
    
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:standard_deviation_volatility
    """
    STDDEV = df['Close'].rolling(window=n).std().rename('STD_' + str(n))
    return STDDEV  


#===========================================
# Brad added: BIAS indicator:
# 计算方法：BIAS指标=(当日收盘价-当日对应的N日移动平均线)/当日对应的N日移动平均线×100%


def BIAS(df, n):
    BIAS = df['Close']-df['Close'].rolling(window=n).mean().rename('BIAS_'+str(n))
    return BIAS







