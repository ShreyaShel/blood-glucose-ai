from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base
import datetime

class GlucoseRecord(Base):
    __tablename__ = "glucose_records"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    value = Column(Float)
    bolus = Column(Float, default=0.0)
    carbs = Column(Float, default=0.0)
    activity = Column(Float, default=0.0)
    is_actual = Column(Integer, default=1) # 1 for actual, 0 for predicted

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    predicted_value = Column(Float)
    actual_value = Column(Float, nullable=True)
    status = Column(String) # low, normal, high
