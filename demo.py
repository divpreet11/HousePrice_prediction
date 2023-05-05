from sklearn.pipeline import Pipeline

import os
from housing.pipeline.pipeline import Pipeline
from housing.logger import logging
from housing.config.configuration import configuration
from housing.component.data_transformation import DataTransformation

def main():
    try:
        #logging.info("Pipeline starts")
        config_path=os.path.join("config","config.yaml")
        pipeline=Pipeline(configuration(config_file_path=config_path))
        pipeline.run_pipeline()
        
        #pipeline.start_data_ingestion()
        
        
        #data_ingestion_artifact = pipeline.start_data_ingestion()
        # data_validation_artifact = pipeline.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
        
        # data_transformation_artifact=pipeline.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact= data_validation_artifact)
        # model_trainer_artifact=pipeline.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        # model_eval_artifact= pipeline.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
        #                                 data_validation_artifact=data_validation_artifact,
        #                                 model_trainer_artifact=model_trainer_artifact)
    
        # pipeline.start_model_pusher(model_eval_artifact=model_eval_artifact)
        
        #config=configuration()
        #data_transformation_config=config.get_data_transformation_config()
        #print(data_transformation_config)
        #schema_file_path=r"C:\Users\prana\machine_learning_project_1\ml_projects\config\schema.yaml"
        #file_path=r"C:\Users\prana\machine_learning_project_1\ml_projects\housing\artifact\data_ingestion\2022-07-06_16-17-02\ingested_data\train\housing.csv"
        #df=DataTransformation.load_data(schema_file_path=schema_file_path,file_path=file_path)
        #print(df.columns)
        #print(df.dtypes)
    except Exception as e:
        print(e)

if __name__=="__main__":
    main()