# from src.mlproject.logger import logging 
# from src.mlproject.exception import CustomException
# import sys
# from src.mlproject.components.data_ingestion import DataIngestion
# from src.mlproject.components.data_transformation import DataTransformation
# from src.mlproject.components.model_trainer import ModelTrainer



# if __name__=="__main__":
#     logging.info("THe execution has started.")

#     try:
#         data_ingestion=DataIngestion()
#         train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

#         data_transformation=DataTransformation()
#         train_arr,test_arr,_=data_transformation.initiate_data_transormation(train_data_path,test_data_path)

#         model_trainer=ModelTrainer()
#         print(model_trainer.initiate_model_trainer(train_arr,test_arr))

#     except Exception as e:
#         logging.info("Custom Exception")
#         raise CustomException(e,sys)

from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.mlproject.pipelines import prediction_pipeline

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data= prediction_pipeline.CustomData(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_data_frame() 
        print(pred_df)

        predict_pipeline= prediction_pipeline.PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print(results)

        return render_template('home.html', results=results[0])

if __name__ =="__main__":
    app.run(host="0.0.0.0",port="8000", debug=True)