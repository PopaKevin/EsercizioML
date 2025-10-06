from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = FastAPI()


model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")


df = pd.read_csv("student_pe_performance.csv")
df.drop("ID", axis=1, inplace=True)
feature_columns = df.drop("Performance", axis=1).columns


ordinal_cols = ['Class_Participation_Level', 'Motivation_Level',
                'Final_Grade', 'Previous_Semester_PE_Grade', 'Grade_Level']
ordinal_categories = [
    ['Low', 'Medium', 'High'],
    ['Low', 'Medium', 'High'],
    ['F', 'D', 'C', 'B', 'A'],
    ['F', 'D', 'C', 'B', 'A'],
    ['9th', '10th', '11th', '12th']
]

preprocessor = ColumnTransformer(transformers=[
    ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_cols),
    ('ohe', OneHotEncoder(drop='first'), ['Gender'])
], remainder='passthrough')

preprocessor.fit(df.drop("Performance", axis=1))

performance_map_inv = {0: 'Low Performer', 1: 'Average Performer', 2: 'High Performer'}

def load_html_form(result=None):
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()
    if result:
        html = html.replace("{{ result }}", f"<strong>Predicted Performance:</strong> {result}")
    else:
        html = html.replace("{{ result }}", "")
    return html

@app.get("/", response_class=HTMLResponse)
async def form_get():
    return HTMLResponse(content=load_html_form(), status_code=200)

@app.post("/", response_class=HTMLResponse)
async def form_post(
    Age: float = Form(...),
    Gender: str = Form(...),
    Grade_Level: str = Form(...),
    Strength_Score: float = Form(...),
    Endurance_Score: float = Form(...),
    Flexibility_Score: float = Form(...),
    Speed_Agility_Score: float = Form(...),
    BMI: float = Form(...),
    Health_Fitness_Knowledge_Score: float = Form(...),
    Skills_Score: float = Form(...),
    Class_Participation_Level: str = Form(...),
    Attendance_Rate: float = Form(...),
    Motivation_Level: str = Form(...),
    Overall_PE_Performance_Score: float = Form(...),
    Improvement_Rate: float = Form(...),
    Final_Grade: str = Form(...),
    Previous_Semester_PE_Grade: str = Form(...),
    Hours_Physical_Activity_Per_Week: float = Form(...)
):
    input_data = [Age, Gender, Grade_Level, Strength_Score, Endurance_Score,
                  Flexibility_Score, Speed_Agility_Score, BMI,
                  Health_Fitness_Knowledge_Score, Skills_Score,
                  Class_Participation_Level, Attendance_Rate,
                  Motivation_Level, Overall_PE_Performance_Score,
                  Improvement_Rate, Final_Grade, Previous_Semester_PE_Grade,
                  Hours_Physical_Activity_Per_Week]

    df_student = pd.DataFrame([input_data], columns=feature_columns)
    X_encoded = preprocessor.transform(df_student)
    X_scaled = scaler.transform(X_encoded)
    pred = model.predict(X_scaled)
    result = performance_map_inv[pred[0]]

    return HTMLResponse(content=load_html_form(result=result), status_code=200)
