
#from matplotlib.style import available
#from sqlalchemy import false
from housing.logger import logging
from housing.exception import Housing_exception
import sys,os
from housing.entity.config_entity import DataValidationconfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.model_profile import Profile
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import pandas as pd
import numpy as np
import json



class DataValidation:
    
    def __init__(self,data_validation_config:DataValidationconfig, data_ingestion_artifact:DataIngestionArtifact ):
        # we are accepting 2 parameter configuration info and the artifact from data ingestion component
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise Housing_exception (e,sys) from e
        
    def is_train_test_file_exist(self):
        """ In this we will check that if the train and test file exist or not"""
        try:
            logging.info("Checking if Training and Testing file available")
            # we are declaring 2 variable initially it is false , when we found that file exist we make it as true
            is_train_file_exist=False
            is_test_file_exist=False
            
            # getting train and test file path from data_ingestion_artifact
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path
            
            # checking if the file eixst or not
            if os.path.exists(train_file_path):
                is_train_file_exist=True
                
            if os.path.exists(test_file_path):
                is_test_file_exist=True
            
            is_available=is_test_file_exist and is_train_file_exist
            # if  both the file exist then only it will return true because it ia a and operation
            logging.info(f"Is Train and Test file exist?->{is_available}")
            
            # we will check here only if it is not available then we will raise exception here only
            if not is_available:
                training_file_path=self.data_ingestion_artifact.train_file_path
                testing_file_path=self.data_ingestion_artifact.test_file_path
                message=f"Training File : [{training_file_path}] or Testing file path: [{testing_file_path}] is not present"
                raise Exception(message)
            
            return is_available
        except Exception as e:
            raise Housing_exception (e,sys) from e
        
        
        
    def validate_dataset_schema(self):
        try:
            validation_status=False
            
                        
            validation_status=True
            return validation_status
        except Exception as e:
            raise Housing_exception(e,sys) from e
        
    def get_and_save_data_drift_report(self):
         try:
             profile = Profile(sections=[DataDriftProfileSection()])
             train_df,test_df=self.get_train_and_test_df()
             
             profile.calculate(train_df,test_df)
             
             report = json.loads(profile.json())
             
             report_file_path=self.data_validation_config.report_file_path
             report_dir=os.path.dirname(report_file_path)
             os.makedirs(report_dir,exist_ok=True)
             
             with open(report_file_path,"w") as report_file:
                 json.dump(report,report_file,indent=6)
                 
             return report
                 
         except Exception as e:
             raise Housing_exception(e,sys) from e
         
    def get_train_and_test_df(self):
        try:
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df,test_df
            
        except Exception as e:
            raise Housing_exception(e,sys) from e
         
    def save_data_drift_report_page(self):
        try:
            dashboard=Dashboard(tabs=[DataDriftTab()])
            
            train_df,test_df=self.get_train_and_test_df()
             
            dashboard.calculate(train_df,test_df)
            
            report_page_file_path=self.data_validation_config.report_page_file_path
            report_dir=os.path.dirname(report_page_file_path)
            os.makedirs(report_dir,exist_ok=True)
            
            dashboard.save(report_page_file_path)
        except Exception as e:
            raise Housing_exception(e,sys) from e     
        
    def is_data_drift_found(self):
        try:
            report=self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise Housing_exception(e,sys) from e  
        
    def initiate_data_validation(self):
        """ This function will return DataValidationArtifact"""
        try:
            self.is_train_test_file_exist()
            self.validate_dataset_schema()
            self.is_data_drift_found()
              
            data_validation_artifact=DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation Performed Sucessfully."
            )    
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise Housing_exception (e,sys) from e
    