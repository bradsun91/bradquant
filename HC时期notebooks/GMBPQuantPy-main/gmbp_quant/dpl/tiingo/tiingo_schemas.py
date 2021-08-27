from sqlalchemy import Column, Date, DateTime, BigInteger, Float, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class TiingoDailyPriceSchema(Base):
    """FMP form 13F schema
    """
    __tablename__ = 'tiingo_daily_price_stg'
    symbol = Column(String(100), primary_key=True, nullable=False)
    report_date = Column(DateTime, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    volume = Column(Float)
    adjClose = Column(Float)
    adjHigh = Column(Float)
    adjLow = Column(Float)
    adjOpen = Column(Float)
    adjVolume = Column(Float)
    divCash = Column(Float)
    splitFactor = Column(Float)
