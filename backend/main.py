from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models, database, simulation
from pydantic import BaseModel
from typing import List, Optional
import datetime

app = FastAPI(title="Glucose AI API")

# CORS for React/Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init DB
models.Base.metadata.create_all(bind=database.engine)

class ChatRequest(BaseModel):
    message: str
    glucose: float

class SimulationResponse(BaseModel):
    timestamp: datetime.datetime
    actual: float
    predictions: List[float] # List of 30m predictions
    prediction_timestamps: List[str]
    status: str
    bolus: float
    carbs: float
    activity: float

@app.post("/start")
def start_simulation(db: Session = Depends(database.get_db)):
    simulation.simulator.current_index = 0
    # Clear history for demo
    db.query(models.GlucoseRecord).delete()
    db.query(models.PredictionHistory).delete()
    db.commit()
    return {"status": "Simulation reset", "initial": simulation.simulator.get_next()}

@app.get("/next", response_model=SimulationResponse)
def get_next_step(db: Session = Depends(database.get_db)):
    data = simulation.simulator.get_next()
    if not data:
        raise HTTPException(status_code=404, detail="End of simulation data")
    
    # Save to history
    record = models.GlucoseRecord(
        timestamp=data['timestamp'],
        value=data['actual'],
        bolus=data['bolus'],
        carbs=data['carbs'],
        activity=data['activity']
    )
    pred = models.PredictionHistory(
        timestamp=data['timestamp'],
        predicted_value=data['predictions'][0], # Store next 5min prediction
        actual_value=data['actual'],
        status=data['status']
    )
    db.add(record)
    db.add(pred)
    db.commit()
    
    return data

@app.post("/chat")
def chat_ai(request: ChatRequest):
    g = request.glucose
    msg = request.message.lower()
    
    if "eat" in msg or "hungry" in msg:
        if g < 70:
            return {"response": "Your glucose is low (70 mg/dL). I suggest consuming 15g of fast-acting carbs like juice or glucose tablets immediately."}
        elif g > 180:
            return {"response": "Your glucose is high. If you eat, consider a low-carb snack and check your correction dose."}
        else:
            return {"response": "Your levels are currently stable. A balanced meal with complex carbs and protein is recommended."}
            
    if "safe" in msg or "status" in msg:
        if g < 70:
            return {"response": "Alert: You are in a hypoglycemic range. Please take action to raise your blood sugar."}
        if g > 250:
            return {"response": "Alert: You are in a very high range (Hyperglycemia). Check for ketones and ensure insulin delivery is working."}
        return {"response": "Your glucose levels are within a safe/target range. Keep up the good work!"}

    return {"response": f"I see your current glucose is {g} mg/dL. How can I help you manage your T1DM today?"}

@app.get("/history")
def get_history(db: Session = Depends(database.get_db)):
    records = db.query(models.GlucoseRecord).order_by(models.GlucoseRecord.timestamp.desc()).limit(50).all()
    return records

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
