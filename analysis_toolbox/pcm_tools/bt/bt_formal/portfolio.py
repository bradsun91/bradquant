# Part 4/8. portfolio

"""
In the previous article on event-driven backtesting we considered how to construct a Strategy class hierarchy. 
Strategies, as defined here, are used to generate signals, which are used by a portfolio object in order 
to make decisions on whether to send orders. As before it is natural to create a Portfolio abstract base class 
(ABC) that all subsequent subclasses inherit from.

This article describes a NaivePortfolio object that keeps track of the positions within a portfolio and 
generates orders of a fixed quantity of stock based on signals. Later portfolio objects will include more 
sophisticated risk management tools and will be the subject of later articles.
"""

# Position Tracking and Order Management #

"""
The portfolio order management system is possibly the most complex component of an event-driven backtester. 
Its role is to keep track of all current market positions as well as the market value of the positions 
(known as the "holdings"). This is simply an estimate of the liquidation value of the position and is derived in 
part from the data handling facility of the backtester.

In addition to the positions and holdings management the portfolio must also be aware of risk factors and 
position sizing techniques in order to optimise orders that are sent to a brokerage or other form of market access.

Continuing in the vein of the Event class hierarchy a Portfolio object must be able to handle SignalEvent objects, 
generate OrderEvent objects and interpret FillEvent objects to update positions. Thus it is no surprise that the 
Portfolio objects are often the largest component of event-driven systems, in terms of lines of code (LOC).

"""

# Implementation #

"""
We create a new file portfolio.py and import the necessary libraries. These are the same as most of the other abstract 
base class implementations. We need to import the floor function from the math library in order to generate integer-valued 
order sizes. We also need the FillEvent and OrderEvent objects since the Portfolio handles both.
"""

import datetime
import numpy as np
import pandas as pd
import queue

from abc import ABCMeta, abstractmethod
from math import floor

from pcm_tools.bt.bt_formal.event import FillEvent, OrderEvent
from pcm_tools.bt.bt_formal.performance import create_sharpe_ratio, create_drawdowns


class Portfolio(object):


    __metaclass__ = ABCMeta

    @abstractmethod
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders 
        based on the portfolio logic.
        """
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):

        raise NotImplementedError("Should implement update_fill()")




class NaivePortfolio(Portfolio):

    
    def __init__(self, bars, events, start_date, initial_capital=100000.0):

        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        
        self.all_positions = self.construct_all_positions()
        self.current_positions = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()



    def construct_all_positions(self):

        d = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        d['datetime'] = self.start_date
        return [d]


    def construct_all_holdings(self):

        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]


    def construct_current_holdings(self):

        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.symbol_list] )
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d


    def update_timeindex(self, event):

        bars = {}
        for sym in self.symbol_list:
            bars[sym] = self.bars.get_latest_bars(sym, N=1)

        # Update positions
        dp = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dp['datetime'] = bars[self.symbol_list[0]][0][1]

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        dh = dict( (k,v) for k, v in [(s, 0) for s in self.symbol_list] )
        dh['datetime'] = bars[self.symbol_list[0]][0][1]
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for s in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[s] * bars[s][0][5]
            dh[s] = market_value
            dh['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(dh)



    def update_positions_from_fill(self, fill):

        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update positions list with new quantities
        self.current_positions[fill.symbol] += fill_dir*fill.quantity



    def update_holdings_from_fill(self, fill):

        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.bars.get_latest_bars(fill.symbol)[0][5]  # Close price
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= (cost + fill.commission)


    def update_fill(self, event):

        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)


    def generate_naive_order(self, signal):

        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength

        mkt_quantity = floor(100 * strength)
        cur_quantity = self.current_positions[symbol]
        order_type = 'MKT'

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')   
    
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
        return order




    def update_signal(self, event):

        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)



    def create_equity_curve_dataframe(self):

        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        self.equity_curve = curve



class NaivePortfolio(object):

    def output_summary_stats(self):

        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        max_dd, dd_duration = create_drawdowns(pnl)

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]
        return stats