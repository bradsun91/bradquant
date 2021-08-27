from sqlalchemy import Column, Date, BigInteger, Float, String
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Sec13FHRSchema(Base):
    """SEC form 13F-HR schema
    """
    __tablename__ = 'sec_13fhr_stg'
    cik = Column(String(100), primary_key=True, nullable=False)
    document_id = Column(String(100), primary_key=True, nullable=False)
    report_date = Column(Date, primary_key=True, nullable=False)
    date_as_of_change = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    name = Column(String(200))
    title_of_class = Column(String(100))
    cusip = Column(String(100))
    value = Column(BigInteger)
    ssh_prn_amt = Column(BigInteger)
    ssh_prn_type = Column(String(50))
    put_call = Column(String(100))
    other_manager = Column(String(100))
    investment_discretion = Column(String(50))
    voting_authority_sole = Column(BigInteger)
    voting_authority_shared = Column(BigInteger)
    voting_authority_none = Column(BigInteger)


class Sec13FHRASchema(Base):
    """SEC form 13F-HRA schema
    """
    __tablename__ = 'sec_13fhra_stg'
    cik = Column(String(100), primary_key=True, nullable=False)
    document_id = Column(String(100), primary_key=True, nullable=False)
    report_date = Column(Date, primary_key=True, nullable=False)
    date_as_of_change = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    name = Column(String(200))
    title_of_class = Column(String(100))
    cusip = Column(String(100))
    value = Column(BigInteger)
    ssh_prn_amt = Column(BigInteger)
    ssh_prn_type = Column(String(50))
    put_call = Column(String(100))
    other_manager = Column(String(100))
    investment_discretion = Column(String(50))
    voting_authority_sole = Column(BigInteger)
    voting_authority_shared = Column(BigInteger)
    voting_authority_none = Column(BigInteger)



class Sec4NonDerivativeSchema(Base):
    """SEC form 4 Non-Derivative schema
    """
    __tablename__ = 'sec_4_nonderivative_stg'
    cik = Column(String(100), primary_key=True, nullable=False)
    document_id = Column(String(100), primary_key=True, nullable=False)
    report_date = Column(Date, primary_key=True, nullable=False)
    date_as_of_change = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    security_title = Column(String(200))
    transaction_date = Column(Date)
    deemed_execution_date = Column(Date)
    transaction_coding_form_type = Column(String(50))
    transaction_coding_code = Column(String(50))
    transaction_coding_equity_swap_involved = Column(BigInteger)
    transaction_shares = Column(BigInteger)
    transaction_price_per_share = Column(Float)
    transaction_acquired_disposed_code = Column(String(50))
    shares_owned_following_transaction = Column(BigInteger)
    direct_or_indirect_ownership = Column(String(50))
    nature_of_ownership = Column(String(500))


class Sec4DerivativeSchema(Base):
    """SEC form 4 Derivative schema
    """
    __tablename__ = 'sec_4_derivative_stg'
    cik = Column(String(100), primary_key=True, nullable=False)
    document_id = Column(String(100), primary_key=True, nullable=False)
    report_date = Column(Date, primary_key=True, nullable=False)
    date_as_of_change = Column(Date, primary_key=True, nullable=False)
    row_num = Column(BigInteger, primary_key=True, nullable=False)
    security_title = Column(String(200))
    conversion_or_exercise_price = Column(Float)
    transaction_date = Column(Date)
    deemed_execution_date = Column(Date)
    transaction_coding_form_type = Column(String(50))
    transaction_coding_code = Column(String(50))
    transaction_coding_equity_swap_involved = Column(BigInteger)
    transaction_shares = Column(BigInteger)
    transaction_price_per_share = Column(Float)
    transaction_acquired_disposed_code = Column(String(50))
    shares_owned_following_transaction = Column(BigInteger)
    exercise_date = Column(Date)
    expiration_date = Column(Date)
    underlying_security_title = Column(String(200))
    underlying_security_shares = Column(BigInteger)
    direct_or_indirect_ownership = Column(String(50))
    nature_of_ownership = Column(String(500))