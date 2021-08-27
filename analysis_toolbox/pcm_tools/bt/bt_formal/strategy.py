# Part 3/8. strategy.py



import datetime
import numpy as np
import pandas as pd
import queue ############updated here 10-13-2017.

from abc import ABCMeta, abstractmethod

from pcm_tools.bt.bt_formal.event import SignalEvent





class Strategy(object):


    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_signals(self):

        raise NotImplementedError("Should implement calculate_signals()")




class BuyAndHoldStrategy(Strategy):


    def __init__(self, bars, events):

        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        # Once buy & hold signal is given, these are set to True
        self.bought = self._calculate_initial_bought()



	def _calculate_initial_bought(self):

        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought



	def calculate_signals(self, event):

        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars(s, N=1)
                if bars is not None and bars != []:
                    if self.bought[s] == False:
                        # (Symbol, Datetime, Type = LONG, SHORT or EXIT)
                        signal = SignalEvent(bars[0][0], bars[0][1], 'LONG')
                        self.events.put(signal)
                        self.bought[s] = True


