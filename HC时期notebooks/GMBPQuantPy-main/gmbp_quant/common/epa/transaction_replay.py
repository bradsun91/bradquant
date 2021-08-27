import gmbp_quant.dal.mkt_data as dalmd
import gmbp_quant.common.utils.datetime_utils as dtu
import pandas as pd
import time
import datetime
import dateutil


def to_datetime(some_time):
    """
    change any format of time to a datetime.datetime object
    :param some_time: any type
    :return: datetime.datetime
    """
    if type(some_time) == int and (str(some_time)) == 8:
        return dtu.dateid_to_datetime(int(some_time))
    if type(some_time) == int or type(some_time) == float:  # epoch time
        current_epoch_time = time.time()
        if some_time <= current_epoch_time:
            # some_time is epoch time in seconds
            return datetime.datetime.fromtimestamp(some_time)
        elif some_time > current_epoch_time >= some_time / 1000.0:
            # epoch time in milliseconds
            return datetime.datetime.fromtimestamp(some_time / 1000.0)
        elif some_time / 1000.0 > current_epoch_time > some_time / 1000000.0:
            # epoch time in microseconds
            return datetime.datetime.fromtimestamp(some_time / 1000000.0)
    elif type(some_time) == str:
        try:
            return dateutil.parser.parse(str(some_time))
        except Exception as e:
            pass
    raise Exception("Format of input time can not be recognized: {}".format(some_time))


def transaction_replay(transaction_df, dt_col="Datetime", sec_id_col="DisplayCode", initial_portfolio={}):
    """
    "*CASH*" ticker is used for deposits/withdraws. SELL "*CASH*" means Deposit.

    :param transaction_df: pandas DataFrame with cols of [dt_col, sec_id_col, "Side", "Quantity", "FillPrice",
                            "CommissionFees"]
    :param dt_col: the name of the DateTime column in transaction_df
    :param sec_id_col: the name of the security column in transaction_df
    :param initial_portfolio: {sec_id: quantity}, the initial portfolio before the first transaction in transaction_df
    :return: two DataFrame's, one is daily snapshot showing ["Date", "DisplayCode", "ClosePrice", "QuantityChange",
             "Quantity", "QuantityPreviousDay", "CashChange", "TradingPnL"]), and another one is
             daily snapshot summary with columns of ["Date", "TradingPnL", "PositionPnL", "PnL", "CashChange",
             "CashAmount", "NMV", "AccountBalance"]
    """
    # Get all adj close prices that will be possibly at once to save query time for database server
    dates = [int(to_datetime(_dt).strftime("%Y%m%d")) for _dt in transaction_df[dt_col]]
    _first_transaction_date, _last_transaction_date = min(dates), max(dates)
    tickers_set = set(initial_portfolio.keys()) | set(transaction_df[sec_id_col])
    # TODO(Weimin): remove the following if clause when dalmd.query_security_day_price() supports ticker "*CASH*"
    if "*CASH*" in tickers_set:
        tickers_set.remove("*CASH*")
    tickers_str = ",".join(sorted(list(tickers_set)))
    close_prices = dalmd.query_security_day_price(tickers=tickers_str,
                                                  start_dateid=dtu.prev_biz_dateid(_first_transaction_date),
                                                  end_dateid=_last_transaction_date, cols='ADJ_CLOSE')
    _data = {"DisplayCode": [x.split(".")[0] for x in close_prices["TICKER"]],
             "Date": [int(x.to_pydatetime().strftime("%Y%m%d")) for x in close_prices["time_x"]],
             "AdjClose": list(close_prices["ADJ_CLOSE"])}
    # TODO(Weimin): remove the following if clause when dalmd.query_security_day_price() supports "*CASH*"
    for _date in dtu.infer_biz_dateids(dtu.prev_biz_dateid(_first_transaction_date), _last_transaction_date):
        _data["DisplayCode"].append("*CASH*")
        _data["Date"].append(_date)
        _data["AdjClose"].append(1)
    prices = pd.DataFrame(_data, columns=["DisplayCode", "Date", "AdjClose"])
    prices = prices.set_index(["DisplayCode", "Date"]).sort_index()

    # Start from transaction_df, get "Date", "DisplayCode", "QuantityChange", "CashChange", "TradingPnL"
    # and save them to to processed_transaction_df
    _data = []
    for i, row in transaction_df.iterrows():
        _date = dates[i]
        _display_code = row[sec_id_col]
        _quantity = abs(row["Quantity"])
        _close_price = prices.loc[(_display_code, _date), "AdjClose"]

        if row["Side"].upper() == "BUY":
            # row["CommissionFees"] is supposed to be a negative number in transactions_df if buyer pays money
            _trading_PnL = - _quantity * (row["FillPrice"] - _close_price) + row["CommissionFees"]
            _quantity_change = _quantity
            _cash_change = - _quantity * row["FillPrice"] + row["CommissionFees"]
        else:
            _trading_PnL = _quantity * (row["FillPrice"] - _close_price) + row["CommissionFees"]
            _quantity_change = - _quantity
            _cash_change = _quantity * row["FillPrice"] + row["CommissionFees"]

        _data.append([_date, _display_code, _close_price, _quantity_change, _cash_change, _trading_PnL])

    processed_transaction_df = pd.DataFrame(
        _data, columns=["Date", "DisplayCode", "ClosePrice", "QuantityChange", "CashChange", "TradingPnL"])

    # Group processed_transaction_df by ("Date", "DisplayCode"), and sum the "Quantity" and "TradingPnL"
    # and get a new DataFrame grouped_transaction_df
    grouped_transaction_df = processed_transaction_df.groupby(["Date", "DisplayCode"], as_index=False) \
        .agg({"ClosePrice": "first", "QuantityChange": "sum", "CashChange": "sum", "TradingPnL": "sum"}) \
        .sort_values(by=["Date", "DisplayCode"])

    # Get quantity, "ClosePrice" and "TradingPnL" and  for each security for each trading day including trading days
    # without transactions. For example, if a person bought TSLA on 2021-03-10(Wed) and sold TSLA on 2021-03-12(Fri),
    # data on 2021-03-11 will also be interpolated.
    cur_portfolio = {key: val for key, val in initial_portfolio.items()}
    cur_portfolio["*CASH*"] = cur_portfolio.get("*CASH*", 0)
    i, j = 0, 0
    _dates = dtu.infer_biz_dateids(_first_transaction_date, _last_transaction_date)
    _data = []
    while i < len(_dates):
        while i < len(_dates) and _dates[i] < grouped_transaction_df.iloc[j]["Date"]:
            # for days without any transaction
            for _ticker, _quantity in cur_portfolio.items():
                _data.append([_dates[i], _ticker, prices.loc[(_ticker, _dates[i])]["AdjClose"],
                              0, _quantity, _quantity, 0, 0])
            i += 1
        _seen_tickers = set()
        while j < grouped_transaction_df.shape[0] and _dates[i] == grouped_transaction_df.iloc[j]["Date"]:
            # For days with transactions
            _ticker = grouped_transaction_df.iloc[j]["DisplayCode"]
            _seen_tickers.add(_ticker)
            _quantity = grouped_transaction_df.iloc[j]["QuantityChange"] + cur_portfolio.get(_ticker, 0)
            _data.append([_dates[i], _ticker,
                          prices.loc[(grouped_transaction_df.iloc[j]["DisplayCode"], _dates[i])]["AdjClose"],
                          grouped_transaction_df.iloc[j]["QuantityChange"],
                          _quantity,
                          cur_portfolio.get(_ticker, 0),
                          grouped_transaction_df.iloc[j]["CashChange"],
                          grouped_transaction_df.iloc[j]["TradingPnL"]
                          ])
            # Update the portfolio due to transactions changing quantity
            if _quantity == 0 and _ticker in cur_portfolio:
                del cur_portfolio[_ticker]
            else:
                cur_portfolio[_ticker] = _quantity
            j += 1
        for _ticker, _quantity in cur_portfolio.items():
            if _ticker in _seen_tickers: continue
            _data.append([_dates[i], _ticker, prices.loc[(_ticker, _dates[i])]["AdjClose"],
                          0, _quantity, _quantity, 0, 0])
        i += 1

    # Get daily snapshot of the portfolio
    # Compute "NMV", "PositionPnL", and "PnL". "NMV" only includes the market value of the securities, excluding cash
    portfolio_daily_snapshot_df = pd.DataFrame(
        _data, columns=["Date", "DisplayCode", "ClosePrice", "QuantityChange", "Quantity", "QuantityPreviousDay",
                        "CashChange", "TradingPnL"])
    for i, row in portfolio_daily_snapshot_df.iterrows():
        if row["DisplayCode"] == "*CASH*":  # deposits and withdraws
            portfolio_daily_snapshot_df.at[i, "NMV"] = 0
            portfolio_daily_snapshot_df.at[i, "PositionPnL"] = 0
            portfolio_daily_snapshot_df.at[i, "PnL"] = 0
        else:
            portfolio_daily_snapshot_df.at[i, "NMV"] = portfolio_daily_snapshot_df.iloc[i]["Quantity"] \
                                                       * portfolio_daily_snapshot_df.iloc[i]["ClosePrice"]
            portfolio_daily_snapshot_df.at[i, "PositionPnL"] = \
                portfolio_daily_snapshot_df.iloc[i]["QuantityPreviousDay"] \
                * (portfolio_daily_snapshot_df.iloc[i]["ClosePrice"] \
                   - prices.loc[(row["DisplayCode"], dtu.prev_biz_dateid(row["Date"]))]["AdjClose"])
            portfolio_daily_snapshot_df.at[i, "PnL"] = portfolio_daily_snapshot_df.at[i, "PositionPnL"] \
                                                       + portfolio_daily_snapshot_df.at[i, "TradingPnL"]
    portfolio_daily_snapshot_df = portfolio_daily_snapshot_df.sort_values(by=["Date", "DisplayCode"]).reset_index(
        drop=True)

    # Get daily summary of portfolio
    # Group by portfolio_daily_snapshot_df by "Date" to find the total "PnL" for the whole portfolio
    portfolio_daily_summary_df = portfolio_daily_snapshot_df.groupby(["Date"], as_index=False) \
        .agg({"NMV": "sum", "CashChange": "sum", "TradingPnL": "sum", "PositionPnL": "sum", "PnL": "sum"}) \
        .sort_values(by=["Date"])
    # Get CashAmount and AccountBalance. AccountBalance = CashAmount + NMV
    _cash_mount = initial_portfolio.get("*CASH*", 0)
    for i, row in portfolio_daily_summary_df.iterrows():
        _cash_mount += portfolio_daily_summary_df.iloc[i]["CashChange"]
        portfolio_daily_summary_df.at[i, "CashAmount"] = _cash_mount
        portfolio_daily_summary_df.at[i, "AccountBalance"] = portfolio_daily_summary_df.iloc[i]["NMV"] + _cash_mount
    # Rearrange columns for portfolio_daily_summary_df
    cols = ["Date", "TradingPnL", "PositionPnL", "PnL", "CashChange", "CashAmount", "NMV", "AccountBalance"]
    portfolio_daily_summary_df = portfolio_daily_summary_df[cols]

    return portfolio_daily_snapshot_df, portfolio_daily_summary_df
