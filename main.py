from fastapi import FastAPI, Request, Response, status, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
from joblib import dump, load

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def prediction(request: Request, age: int = Form(...), annual_income: int = Form(...), gender: int = Form(...),):
    try:
        data = [[
            age,
            annual_income,
            gender,
        ]]
        model = load('model.joblib') 
        prediction = (model.predict(data)[0]) * 100
        print(prediction)

        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
    except Exception as e:
        return HTMLResponse(content=f"<html><body>Error: {str(e)}</body></html>", status_code=500)