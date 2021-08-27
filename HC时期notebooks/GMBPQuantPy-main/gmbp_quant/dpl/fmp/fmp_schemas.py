from sqlalchemy import Column, Date, BigInteger, Float, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Fmp13FSchema(Base):
    """FMP form 13F schema
    """
    __tablename__ = 'fmp_13f_stg'
    cik = Column(String(100), primary_key=True, nullable=False)
    cusip = Column(String(100), primary_key=True, nullable=False)
    report_date = Column(Date, primary_key=True, nullable=False)
    fillingDate = Column(Date, primary_key=True, nullable=False)
    acceptedDate = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    nameOfIssuer = Column(String(100))
    titleOfClass = Column(String(100))
    tickercusip = Column(String(100))
    value = Column(BigInteger)
    shares = Column(BigInteger)
    link = Column(String(200))
    finalLink = Column(String(200))


class Fmp4Schema(Base):
    """Fmp form 4 schema
    """
    __tablename__ = 'fmp_4_stg'
    cik = Column(String(100), primary_key=True, nullable=False)
    report_date = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)

class FmpInstHolderSchema(Base):
    """Fmp Institutional Holder schema
    """
    __tablename__ = 'fmp_inst_holder_stg'
    ticker = Column(String(100), primary_key=True, nullable=False)
    holder = Column(String(100), primary_key=True, nullable=False)
    dateReported = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    shares = Column(BigInteger)
    change = Column(BigInteger)

class FmpEtfHolderSchema(Base):
    """Fmp ETF Holder schema
    """
    __tablename__ = 'fmp_etf_holder_stg'
    asset = Column(String(100), primary_key=True, nullable=True, default='')
    holder = Column(String(100), primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    sharesNumber = Column(BigInteger, nullable=True)
    weightPercentage = Column(Float)

class FmpInsiderTradingSchema(Base):
    """Fmp Insider Trading schema
    """
    __tablename__ = 'fmp_insider_trading_stg'
    symbol = Column(String(100), primary_key=True, nullable=False)
    holder = Column(String(100), primary_key=True, nullable=False)
    transactionDate = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)

    reportingCik = Column(String(100))
    transactionType = Column(String(100))
    securitiesOwned = Column(Float)
    companyCik = Column(String(100))
    reportingName = Column(String(100))
    typeOfOwner = Column(String(100))
    acquistionOrDisposition = Column(String(100))
    formType = Column(String(100))
    securitiesTransacted = Column(Float)
    price = Column(Float)
    securityName = Column(String(100))
    link = Column(String(100))