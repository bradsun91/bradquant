def setup():
    setup = {
            "T9000":{"commissions_type" : "yuan_per_contract",
                     "commissions":3, #元一手
                     "slippage": 0, #元一手
                     "margin": 0.02,
                     "unit": 10000,
                     "bpv": 1},

            "j9000":{"commissions_type" : "%%",
                     "commissions":0.00006, # 0.6%%
                     "slippage": 0, #元一手
                     "margin": 0.05,
                     "unit": 100,
                     "bpv": 1},

            "au000":{"commissions_type" : "yuan_per_contract",
                     "commissions":10, #元一手
                     "slippage": 0, #元一手
                     "margin": 0.07,
                     "unit": 1000,
                     "bpv": 1},    

            "IC000":{"commissions_type" : "%%",
                     "commissions":0.000023, #0.23%%
                     "slippage": 0, #元一手
                     "margin": 0.08,
                     "unit": 1,
                     "bpv": 200},

            "l9000":{"commissions_type" : "yuan_per_contract",
                     "commissions":2, #元一手
                     "slippage": 0, #元一手
                     "margin": 0.05,
                     "unit": 5,
                     "bpv": 1},

            "m9000":{"commissions_type" : "%%",
                     "commissions":0.00006, # 0.6%%
                     "slippage": 0, #元一手
                     "margin": 0.05,
                     "unit": 10,
                     "bpv": 1},

            "OI000":{"commissions_type" : "yuan_per_contract",
                     "commissions":2, #元一手
                     "slippage": 0, #元一手
                     "margin": 0.05,
                     "unit": 10,
                     "bpv": 1},
                        }