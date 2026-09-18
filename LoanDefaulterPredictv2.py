from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import io
import base64
import uvicorn
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI(title="Bank Loan Predict Defaulter API")


# Reuse the model and FEATURE_COLUMNS trained earlier
# The model and FEATURE_COLUMNS are already in the kernel's global scope from the Dash app cell.
# If this were a standalone FastAPI app, you'd put the training logic here:
#
train_df = pd.read_csv('BANK LOAN.csv')
X_train = train_df.drop(columns=['DEFAULTER','SN'])
y_train = train_df['DEFAULTER']
model = RandomForestClassifier(
     n_estimators=500,
     oob_score=True,
     random_state=42,
     n_jobs=-1
 )
model.fit(X_train, y_train)
FEATURE_COLUMNS = X_train.columns.tolist()

# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {"message": "Bank loan defaulter prediction API is running"}

# Helper function to parse CSV from uploaded file
#def parse_uploaded_csv(file_content: bytes) -> pd.DataFrame:
#    s = io.StringIO(file_content.decode('utf-8'))
#    return pd.read_csv(s)

# -------------------------------------------------
# Predict defaulters function
# -------------------------------------------------


def generate_prediction_summary(
   # file: UploadFile = File(..., description="CSV file containing test data"),
   # threshold: float = Form(0.5, description="Probability threshold for default prediction")
):
    """
    Predicts credit default for the uploaded test data.

    Args:
        file (UploadFile): A CSV file containing the features for prediction.
        threshold (float): The probability threshold to classify a default.

    Returns:
        dict: A dictionary containing prediction results, including probabilities, 
              predicted labels, counts of defaulters/non-defaulters, and a preview.
    """
    
   # try:
   #     contents = await file.read()
   #     test_df = parse_uploaded_csv(contents)
   # except Exception as e:
   #     return {"error": f"Error processing uploaded file: {e}"}

    # Validate columns
   # missing_cols = set(FEATURE_COLUMNS) - set(test_df.columns)
   # if missing_cols:
   #     return {"error": f"Missing columns in test file: {missing_cols}. Expected: {FEATURE_COLUMNS}"}

    threshold = 0.5
    test_df = pd.read_csv('BANK LOAN_TEST.csv')
    X_test = test_df[FEATURE_COLUMNS]

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]

    test_df['Default_Prob'] = probs.tolist() # Convert to list for JSON serialization
    test_df['Pred_Label'] = (probs >= threshold).astype(int).tolist() # Convert to list

    # Counts
    defaulters = int((test_df['Pred_Label'] == 1).sum())
    non_defaulters = int((test_df['Pred_Label'] == 0).sum())

    # Table Preview
    filtered_df = test_df[test_df['Default_Prob'] >= threshold]

    preview_data = []
    if not filtered_df.empty:
        # Convert DataFrame to a list of dictionaries for JSON serialization
        preview_data = filtered_df[['Default_Prob', 'Pred_Label']].round(3).head(10).to_dict(orient='records')
                  
    summary_df = pd.DataFrame({
            "message": "Prediction successful",
            "threshold_selected": threshold,
            "predicted_defaulters": defaulters,
            "predicted_non_defaulters": non_defaulters,
            "total_records": len(test_df)},index=[0])
            #  "preview_table": preview_data,
           #"all_default_probabilities": test_df['Default_Prob'].tolist(),
           #"all_predicted_labels": test_df['Pred_Label'].tolist(),
            
                
    return summary_df

# -------------------------------------------------
# API endpoint returning JSON data
# -------------------------------------------------
@app.get("/Predict")
def get_prediction_summary():
    df = generate_prediction_summary()
    return df.to_dict(orient="records")
           
    
   
@app.get("/", response_class=HTMLResponse)
async def read_root():
 html_content = """
            <html>
                <head>
                    <title>Bank Predict Loan Defaulter Model</title>
                    <style>
                        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; }
                        h1, h2 { color: #333; }
                        table { border-collapse: collapse; width: 90%; margin: 20px 0; background-color: white; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }
                        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                        th { background-color: #007bff; color: white; }
                        tr:nth-child(even) { background-color: #f2f2f2; }
                        tr:hover { background-color: #ddd; }
                    </style>
                </head>
                <body>
                <h1>Predict Bank Loan Defaulter</h1>    
            
                <button onclick="loadData()">Load Summary</button>
               
                       <table id="summaryTable">
                           <thead>
                               <tr>
                                   <th>message</th>
                                   <th>threshold_selected</th>
                                   <th>predicted_defaulters</th>
                                   <th>predicted_non_defaulters</th>
                                   <th>total_records</th>
                                   
                               </tr>
                           </thead>
                           <tbody></tbody>
                       </table>
               
                       <script>
                           function loadData() {
                               fetch('/Predict')
                                   .then(response => response.json())
                                   .then(data => {
                                       const tbody = document.querySelector('#summaryTable tbody');
                                       tbody.innerHTML = '';
               
                                       data.forEach(row => {
                                           const tr = document.createElement('tr');
                                           tr.innerHTML = `
                                               <td>${row.message}</td>
                                               <td>${row.threshold_selected}</td>
                                               <td>${row.predicted_defaulters}</td>
                                               <td>${row.predicted_non_defaulters}</td>
                                               <td>${row.total_records}</td>
                                               
                                           `;
                                           tbody.appendChild(tr);
                                       });
                                   })
                                   .catch(error => {
                                       alert('Error fetching data');
                                       console.error(error);
                                   });
                           }
                       </script>         
                    """          
               
# html_content += "<h2>Product Usage vs Campaign Response</h2>"

 #html_content += tabvalue.to_html(classes="table table-striped")
            
# html_content += """
#              </body>
#            </html>
#                 """
 return HTMLResponse(content=html_content, status_code=200)
