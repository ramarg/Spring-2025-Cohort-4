# Spring-2025-Cohort-4
**Product Vision:**
The goal of this project is to create an AI-based solution that assists businesses in optimizing their marketing strategy based on historical sales data. Through analysis of trends across customer groups, product groups, and promotion campaigns, we will assist businesses in gaining predictive insights that will allow them to make better marketing decisions, improve sales, and improve profitability. 

**Files:**
Data - Contains our data from kaggle. Superstore.csv will be the main dataset.
Report - Group Project - Milestone 2_ Model Development & Product Integration - Document
Python Notebook - Holds the code of the Model developmentaion and Evaluation
Python File - Holds the code of the Integeration Prototype
index.html - This form sends user input to the /predict endpoint in the Flask app, where it's processed by the XGBoost model to return a profit prediction.
pkl - This is a pre-trained XGBoost model saved as `xgb_model.pkl`. Make sure this file is in the same directory as your `app.py` to ensure the app runs correctly.

**Steps to run the code and web app:**
i) Run the Milestone2_Model.ipynb, Load the Superstore.csv and make sure the data path is changed

ii) Save the pre-trained XGBoost model as `xgb_model.pkl`

iii) Make sure all the file saved in the same directory

iv) Create folder templates and save the index.html inside the folder(Or else app.py won't execute)

v) Open terminal and run this command Streamlit run app.py (Assuming app.py is downloaded in the same working directory)

vi) After the command is executed it will be redirected to a webpage

vii) Give some random inputs and click on predict result
