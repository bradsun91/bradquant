def pcm_etfs():

    pcm_etf_list = {

    # Asset Class 1: Stock Equity ETFs:

        # U.S. ETFs, key starting with 'US'
        'USlarge': ['SPY', 'IVV', 'VOO', 'QQQ'], 
        'USmid': ['IJH', 'MDY', 'VO'], 
        'USsmall': ['IWM', 'IJR', 'VB'], 
        'USvalue': ['VLUE', 'DVP', 'SYLD'], 
        'USmomentum':['MTUM', 'PDP', 'FV'],
        'USlowVol':['SPLV', 'USMV', 'USLB'], 
        'UShighQuality':['QUAL', 'SPHQ', 'OUSA'],
        # Other Developed country ETFs, keys starting with 'D'
        'Dlargecap': ['EFA', 'IEFA', 'VEA'], 
        'Dmidcap': ['DIM'], 
        'Dsmallcap': ['SCZ', 'GWX', 'VSS'], 
        'Dvalue': ['PFX', 'EFV', 'IVLU'], 
        'Dmomentum': ['PIZ', 'GMOM', 'IMTM'],
        'DlowVol': ['EFVA', 'IDLV', 'ACWV'],  
        'Dquality': ['IDHQ', 'IQLT', 'IQDF'], 

    # Asset Class 2: Real Estate ETFs

        # U.S.  
        'USre': ['VNQ', 'RWR', 'SCHH'], 
        # Developed Countries
        'Dre': ['VNQI', 'IFGL', 'RWX'], 


    # Asset Class 3: Bond ETFs: 

        # U.S.

            # Treasury bond ETFs:
            'UStbill': ['SHY', 'IEI', 'IEF'],
            # Inflation-Protection bond ETFs:
            'USinfl': ['TIP'],
            #Corporate bond ETFs
            'UScorp': ['LQD', 'VCIT', 'HYG', 'HYS'],

        # Developed Countries:

            # Treasury ETFs:
            'Dtbill': ['ISHG', 'BWX'],
            # Corporate bond ETFs
            'Dcorp': ['PICB', 'CORP', 'IBND', 'IHY'],

    # Asset Class 4: Commodity ETFs:
            'Com': ['USCI', 'PDBC'],

    # Asset Class 5: Alternative ETFs:
            'Alt': ['WDTI', 'QAI'],
        }

    return pcm_etf_list