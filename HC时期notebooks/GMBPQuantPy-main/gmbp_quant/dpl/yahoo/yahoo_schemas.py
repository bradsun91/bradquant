from sqlalchemy import Column, Date, DateTime, BigInteger, Float, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class YahooIpoSchema(Base):
    """Yahoo IPO schema
    """
    __tablename__ = 'yahoo_ipo_stg'
    Symbol = Column(String(100), primary_key=True, nullable=True, default='')
    Company = Column(String(100), primary_key=True, nullable=True, default='')
    Exchange = Column(String(100))
    row_num = Column(BigInteger, primary_key=True, nullable=False)

    IpoDate = Column(DateTime)
    Currency = Column(String(100))
    Price = Column(String(100))
    Shares = Column(String(100))
    Actions = Column(String(100))

    PriceRange = Column(String(100))
