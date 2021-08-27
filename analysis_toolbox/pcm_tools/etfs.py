def pcm_etfs_dict():

    pcm_etf_list_dict = {

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

    return pcm_etf_list_dict





def pcm_etfs():

    pcm_etf_list = [

    # Asset Class 1: Stock Equity ETFs:

        # U.S. ETFs, key starting with 'US'
        'SPY', 'IVV', 'VOO', 'QQQ', 
        'IJH', 'MDY', 'VO', 
        'IWM', 'IJR', 'VB', 
        'VLUE', 'DVP', 'SYLD', 
        'MTUM', 'PDP', 'FV',
        'SPLV', 'USMV', 'USLB', 
        'QUAL', 'SPHQ', 'OUSA',
        # Other Developed country ETFs, keys starting with 'D'
        'EFA', 'IEFA', 'VEA', 
        'DIM', 
        'SCZ', 'GWX', 'VSS', 
        'PFX', 'EFV', 'IVLU', 
        'PIZ', 'GMOM', 'IMTM',
        'EFVA', 'IDLV', 'ACWV',  
        'IDHQ', 'IQLT', 'IQDF', 

    # Asset Class 2: Real Estate ETFs

        # U.S.  
        'VNQ', 'RWR', 'SCHH', 
        # Developed Countries
        'VNQI', 'IFGL', 'RWX', 


    # Asset Class 3: Bond ETFs: 

        # U.S.

            # Treasury bond ETFs:
            'SHY', 'IEI', 'IEF',
            # Inflation-Protection bond ETFs:
            'TIP',
            #Corporate bond ETFs
            'LQD', 'VCIT', 'HYG', 'HYS',

        # Developed Countries:

            # Treasury ETFs:
            'ISHG', 'BWX',
            # Corporate bond ETFs
            'PICB', 'CORP', 'IBND', 'IHY',

    # Asset Class 4: Commodity ETFs:
            'USCI', 'PDBC',

    # Asset Class 5: Alternative ETFs:
            'WDTI', 'QAI',
        ]

    return pcm_etf_list