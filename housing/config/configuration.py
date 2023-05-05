from ast import Pass
from doctest import REPORT_CDIFF
#from sympy import EX
from housing.entity.config_entity import *
from housing.util.util import read_yaml
from housing.constants import *
from housing.exception import Housing_exception
from housing.logger import logging


import sys

class configuration:
   
    def __init__(self,config_file_path:str=CONFIG_FILE_PATH, current_time_stamp:str=CURRENT_TIME_STAMP)->None:
        try:
                self.config_info=read_yaml(config_file_path) # return the yaml file in the form of dictionary that we have already seen in tje jupyter notebook
                self.training_pipeline_config=self.get_training_pipeline_config() # iski jarurat tab hogi jab hum artifact_dir k andar har ek component k artifat dir design kr rhe honge check data_ingestion_artifact_dir
                self.time_stamp=current_time_stamp
        except Exception as e:
            raise Housing_exception(e,sys) from e
            
    def get_data_ingestion_config(self):
        """ This function weill return The ->DataIngestionconfig"""
        try:
            
            artifact_dir=self.training_pipeline_config.artifact_dir
            # This will give the outer artifact directort within this we can create our data_ingestion_artifact_dir
            
            data_ingestion_artifact_dir=os.path.join(artifact_dir,DATA_INGESTION_ARTIFACT_DIR,self.time_stamp)
            # This line will make The data_ingestion artifact directory within the main artifact folder
            
            
            data_ingestion_info=self.config_info[DATA_INGESTION_CONFIG_KEY] 
            ####This above line will contain the key of the data injection componenet (returned by the yaml file)
            
            dataset_download_url=data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY] 
            ###by using this we are getting the url information ,This is just a dictionary indexing by using various variable nothing else
            
            tgz_download_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY])
            ###This will create the tgz directory  path
            
            raw_data_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])
            ###This will create the raw data directory path
            
            ingested_data_dir=os.path.join(data_ingestion_artifact_dir,data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            # see this we are  creating a new folder ingested dir iske andar hum train and test csv ko rakhenge
            
            ingested_train_dir=os.path.join(artifact_dir,ingested_data_dir,data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY])   
            ## This will create a folder inside the ingested dir
            
            ingested_test_dir=os.path.join(artifact_dir,ingested_data_dir,data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY])
            ## this will also create a folder inside the ingestd dir folder
            
            
            data_ingestion_config=DataIngestionconfig(
                dataset_download_url=dataset_download_url, 
                tgz_download_dir=tgz_download_dir, 
                raw_data_dir=raw_data_dir, 
                ingested_train_dir=ingested_train_dir, 
                ingested_test_dir=ingested_test_dir
            )
            return data_ingestion_config
            logging.info(f"data_ingestion_config : {data_ingestion_config}")
        except Exception as e:
            raise Housing_exception(e,sys) from e
    
    def get_data_validation_config(self):
        """This will return the ->DataValidationconfig"""
        try:
                logging.info("Data validation started")
                artifact_dir=self.training_pipeline_config.artifact_dir
                # creating a root directory
                
                data_validation_artifact_dir=os.path.join(artifact_dir,DATA_VALIDATION_ARTIFACT_DIR,self.time_stamp)
                logging.info(f"Data validation artifact path is:[ {data_validation_artifact_dir} ]")
                # see thr folder structure inside the root artifact dir we have the folder for the component artifact , thats what we are creating
                
                data_validation_info=self.config_info[DATA_VALIDATION_CONFIG_KEY]
                
                
                schema_file_path=os.path.join(ROOT_DIR,data_validation_info[DATA_VALIDATION_SCHEMA_DIR_KEY],data_validation_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
                
                logging.info(f"Schema file path created sucssfully and the file path is : [ {schema_file_path} ]")
                
                report_file_path=os.path.join(data_validation_artifact_dir,data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY])
                logging.info(f"report file path created sucssfully and the file path is : [ {report_file_path} ]")
                
                report_page_file_path=os.path.join(data_validation_artifact_dir,data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])
                logging.info(f"report page file path created sucssfully and the file path is : [ {report_page_file_path} ]")
                
                data_validation_config=DataValidationconfig(schema_file_path=schema_file_path,
                                                            report_file_path=report_file_path,
                                                            report_page_file_path=report_page_file_path)
                
                logging.info(f"Data validation completed sucessfully")
                
                return data_validation_config 
        except Exception as e:
            raise Housing_exception(e,sys) from e
            
    
    
    
    def get_data_transformation_config(self):
        """ This funtion will return the ->DataTransformationconfig"""
        
        try:
            
            artifact_dir=self.training_pipeline_config.artifact_dir
            data_transformation_artifact_dir=os.path.join(artifact_dir,DATA_TRANSFORMATION_ARTIFACT_DIR,self.time_stamp)
            # Thios will make the  outer artifact directory
            logging.info(f"Outer artifact folder created and the path is: [ {data_transformation_artifact_dir} ]")
            
            data_transformation_info=self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            # tHIS WILL PROVIDE THE KEY SO THET WE CAN ACCESS THE THIS ENTITY 
            
            transformed_dir_key=os.path.join(data_transformation_artifact_dir,data_transformation_info[DATA_TRANSFORMATION_DIR_NAME_KEY])
            # THSI WILLN CREATE A DIRECTORY INSIDE THE ARTIFACT->DATA_TRANSFORMATION-> , TO STORE THE TRANSFORMED  TRAIN AND TEST FILE
            
            transformed_train_data_dir=os.path.join(data_transformation_artifact_dir,transformed_dir_key,data_transformation_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])
            transformed_test_data_dir=os.path.join(data_transformation_artifact_dir,transformed_dir_key,data_transformation_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])
            # THIS WILL CREATE THE TRANSFORMED TRAIN AND TEST FILE PATH
            logging.info(f"Transformed train dir is created and the path is: [ {transformed_train_data_dir} ]")
            logging.info(f"Transformed test dir is created and the path is: [ {transformed_test_data_dir} ]")
            
            add_bedroom_per_room=data_transformation_info[DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY]
            #PREPEARING THE PARAMETER add_bedroom_per_room
            
            preprocessed_object_file_path=os.path.join(data_transformation_artifact_dir,data_transformation_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],data_transformation_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY])
            
            data_transformation_config=DataTransformationconfig(add_bedroom_per_room=add_bedroom_per_room,
                                                                tranformed_train_dir=transformed_train_data_dir,
                                                                transformed_test_dir=transformed_test_data_dir,
                                                                preprocessed_object_file_path=preprocessed_object_file_path)
            logging.info(f"Data Transformation config : {data_transformation_config}")
            return data_transformation_config
  
        except Exception as e:
            raise Housing_exception(e,sys) from e
                
   
        
    def get_model_trainer_config(self):
        """->ModelTrainingconfig:"""
        try:
            # getting pipeline level artifact dir
            artifact_dir=self.training_pipeline_config.artifact_dir
            
            #getting model_training_artifact_dir inside the artifact_dir
            model_training_artifact_dir=os.path.join(artifact_dir,MODEL_TRAINER_ARTIFACT_DIR_KEY,self.time_stamp)
            
            #now we have to get the actual path of the file
            model_trainer_config_info=self.config_info[MODEL_TRAINER_CONFIG_KEY]
            logging.info("Getting Model Trainer File Path")
            model_trainer_file_path=os.path.join(model_training_artifact_dir
                                                 ,model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY]
                                                ,model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY])
            
            logging.info(f"Model trainer file path is :[{model_trainer_file_path}]")
            # now we will have to get the base accuracy
            base_accuracy=model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            
            # we will also provid the path of model.yaml file 
            model_config_file_path = os.path.join(model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY],
                                                 model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY])
            
            # fillinhg the form of modeltrainerconfig
            model_trainer_config=ModelTrainingconfig(trained_model_file_path=model_trainer_file_path,
                                                     base_accuracy=base_accuracy,
                                                     model_config_file_path=model_config_file_path)
            logging.info(f"Model Training Config: [{model_trainer_config}]")
            return model_trainer_config
            
        except Exception as e:
            raise Housing_exception(e,sys) from e
        


    def get_model_evaluation_config(self):
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                        MODEL_EVALUATION_ARTIFACT_DIR, )

            model_evaluation_file_path = os.path.join(artifact_dir,
                                                    model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY])
            response = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                            time_stamp=self.time_stamp)
            
            
            logging.info(f"Model Evaluation Config: {response}.")
            return response
        except Exception as e:
            raise Housing_exception(e,sys) from e


    def get_model_pusher_config(self):#-> ModelPusherConfig:
        try:
            time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            
            # we will create a seperate folder outside
            export_dir_path = os.path.join(ROOT_DIR, model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           time_stamp)

            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)
            logging.info(f"Model pusher config {model_pusher_config}")
            return model_pusher_config

        except Exception as e:
            raise Housing_exception(e,sys) from e
        
    def get_training_pipeline_config(self)->TrainingPipelineconfig:
        try:
            training_pipeline_config=self.config_info[TRAINING_PIPELINE_CONFIG_KEY] # This will return the key of the pipeline , jo yaml file k andar hai
            artifact_dir=os.path.join(ROOT_DIR,training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]) # This will join the root dir( that we have created inside the constant folder) and pipeline name and the artifact dir( which is TRAINING_PIPELINE_ARTIFACT_DIR from constant folder )
            training_pipeline_config=TrainingPipelineconfig(artifact_dir=artifact_dir) # name tuple return krega jisse ki jab hum training_pipeline_config return ko call kre class k through to wo sara info dede for this check the notebbok 
            logging.info(f"Training pipeling config: {training_pipeline_config}")
            return training_pipeline_config
            
        
        except Exception as e:
            raise Housing_exception(e,sys) from e 
            
        
        
        
        
        
#  so in this we can see that we have created the classes that will return the entity(yaml file ko read krenge aur jo structure tha entity ka usko input provide kr k return kr denge )
#  but yaml file ko read kaise krenge??? check the flow Txt FOCUS one thing ki ye configuration class 'koi function perform
#  nhi kr rha hai ye just folders k path ko setup kr rha hai 
        
        
        
