# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:08:57 2021

@author: SHREYANSH
"""
# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI,File,UploadFile
from helper import reduce_mem_usage
from Predicter_Function import predicterFunction
import pandas as pd

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, World'}
    
@app.post("/direct_csv_predict")
async def parsecsvdirect(csv_file:UploadFile = File("sample_test.csv")):
    #csv_reader = csv.reader(codecs.iterdecode(csv_file.file,'utf-8'))
    print(1)
    test_data = pd.read_csv(csv_file.file)    
    print(test_data.shape)
    
    prices_df = pd.read_csv(r'sample_test_sell_price.csv')
    print(prices_df.shape)
    #prices_df = pd.read_csv('sell_prices_zip.zip', compression='zip',sep=',')
    calendar_df = pd.read_csv(r'calendar.csv')
    print(calendar_df.shape)
    
    print(2)
    calendar_df = reduce_mem_usage(calendar_df,False)
    prices_df = reduce_mem_usage(prices_df,False)
    print(3)
    
    Val_op,Test_op = predicterFunction(test_data,calendar_df,prices_df)
    print('Forecast sales from days 1914 till 1941 is:')
    print(Val_op)
    print('\nForecast sales from days 1942 till 1969 is:')
    print(Test_op)
    
    #response = StreamingResponse(io.StringIO(test_data.to_csv("fatapi_predictions.csv",index=False)), media_type="text/csv")
    #Val_op.to_csv('Validatio_Predictions.csv')
    
    return Test_op
    
    
    





# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
