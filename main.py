from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Body
from pydantic import BaseModel
import pickle 
import pandas as pd



app = FastAPI(); 

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"], 
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


class Data(BaseModel): #pydantic model for data that will be used for prediction 
   likes_puzzles:int 
   enjoys_reading:int 
   loves_nature:int 
   prefers_solitude:int
   enjoys_socializing:int 
   loves_technology:int 
   loves_adventure:int 
   prefers_routine:int 
   loves_animals:int 
   enjoys_traveling:int 


with open("model.pkl","rb") as f: 
  model = pickle.load(f) 


@app.get("/welcome") 
async def root():
    return {
       "message":"Welcome to this prediction api",
       "authon":"Nikhil",
       }



@app.post("/prediction/v1/character")
async def get_predicted_character(new_data:Data, response:Response):
  try:
     new_data = pd.DataFrame([new_data.model_dump()]) 
     prediction = model.predict(new_data)
     response.status_code = status.HTTP_200_OK
     data = {
            "message":"success", 
            "character":prediction[0] 
        }
     return JSONResponse(content=data) 
  
  except Exception as e: 
     error_message = {"message": "error", "detail": str(e)}
     response.status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
     return JSONResponse(content=error_message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)



