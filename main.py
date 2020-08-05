# Data Handling
import logging
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pydantic import BaseModel
from time import strftime
from typing import Optional

# Server
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, filename='sample.log')

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "2cd7ff405803dfdd2b92d3cf0472948b3b58a6d454a3b055502ec3c1835d1084"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    },
    "aminah": {
        "username": "aminah",
        "full_name": "Aminah Maemunah",
        "email": "amin_ach@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Data(BaseModel):
    grade: str                        # grade of the loan (categorical)
    sub_grade_num: float              # sub-grade of the loan as a number from 0 to 1
    short_emp:float                   # one year or less of employment
    emp_length_num: float             # number of years of employment
    home_ownership: str               # home_ownership status: own, mortgage or rent
    dti: float                        # debt to income ratio
    purpose: str                      # the purpose of the loan
    payment_inc_ratio: float          # ratio of the monthly payment to income
    delinq_2yrs: float                # number of delinquincies 
    delinq_2yrs_zero: float           # no delinquincies in last 2 years
    inq_last_6mths: float             # number of creditor inquiries in last 6 months
    last_delinq_none: float           # has borrower had a delinquincy
    last_major_derog_none: float      # has borrower had 90 day or worse rating
    open_acc: float                   # number of open credit accounts
    pub_rec: float                    # number of derogatory public records
    pub_rec_zero: float               # no derogatory public records
    revol_util: float                 # percent of available credit being used

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create API server
app = FastAPI(debug=True)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Update get_current_user to receive the same token as before, but this time, using JWT tokens.
# Decode the received token, verify it, and return the current user.
# If the token is invalid, return an HTTP error right away.
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Create a timedelta with the expiration time of the token.
# Create a real JWT access token and return it.
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]

# Initialize files
model = joblib.load('./model/best_model.pkl', 'r')
mapper_fit = joblib.load('./model/mapper_fit.pkl', 'r')

numerical_cols = ['sub_grade_num',
                'short_emp',
                'emp_length_num',
                'dti',
                'payment_inc_ratio',
                'delinq_2yrs',
                'delinq_2yrs_zero',
                'inq_last_6mths',
                'last_delinq_none',
                'last_major_derog_none',
                'open_acc',
                'pub_rec',
                'pub_rec_zero',
                'revol_util']

categorical_cols = ['grade',
                    'home_ownership',
                    'purpose']

new_col = ["prediction",
            "prediction_good_loan_%",
            "prediction_bad_loan_%",
            "input_date",
            "input_time"]

col_output = ["Applicant"] + numerical_cols + categorical_cols + new_col

@app.post("/users/me/predict")
async def predict(data: Data, current_user: User = Depends(get_current_active_user)):
    try:
        # Load data
        data_dict = data.dict()
        data=[x for x in data_dict.values()]
        colz=[x for x in data_dict.keys()]
        dfx=pd.DataFrame(data=[data], columns=colz)

        # Apply mapper
        XX1 = mapper_fit.transform(dfx)

        # Prepare data
        XX2 = dfx[numerical_cols]
        XX = np.hstack((XX1,XX2))

        # Prediction
        prediction = model.predict(XX)
        prediction_good_loan = model.predict_proba(XX)[:,0][0]
        prediction_bad_loan = model.predict_proba(XX)[:,1][0]

        # Create result
        if int(prediction) == 1:
            prediction_result = str("Bad Loan")
        else:
            prediction_result = str("Good Loan")

        if int(prediction) == 1:
            prob_result = str(round(prediction_bad_loan*100,1))+"%"
        else:
            prob_result = str(round(prediction_good_loan*100,1))+"%"

        # Store new input in csv file
        input_date = str(strftime('%Y-%m-%d'))
        input_time = str(strftime('%H:%M:%S'))

        dfx["Applicant"] = str(current_user.username)
        dfx["prediction"] = int(prediction)
        di = {1: "Bad Loan", 0: "Good Loan"}
        dfx["prediction"] = dfx["prediction"].map(di)

        dfx["prediction_good_loan_%"] = round(prediction_good_loan*100,1)
        dfx["prediction_bad_loan_%"] = round(prediction_bad_loan*100,1)

        dfx["input_time"] = input_time
        dfx["input_date"] = input_date

        new_input_path = ('./new_input/new_input.csv')

        if os.path.exists(new_input_path) == True:
            old_data = pd.read_csv(new_input_path,index_col=False)
        else:
            old_data = pd.DataFrame()
            
        updated_data = old_data.append(dfx,ignore_index = True)
        updated_data[col_output].to_csv(new_input_path,header=True,index=False)

        # Create and return prediction
        output = {
            "applicant": current_user.username,
            "prediction":str(prediction_result),
            "prob":str(prob_result)
        }

        return output
    
    except:
        my_logger.error("Something went wrong!")
        return {"prediction": "error"}