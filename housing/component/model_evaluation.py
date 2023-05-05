from housing.logger import logging
from housing.exception import Housing_exception
from housing.entity.config_entity import ModelEvaluationConfig
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from housing.constants import *
import numpy as np
import os
import sys
from housing.util.util import write_yaml_file, read_yaml, load_object,load_data
from housing.entity.model_factory import evaluate_regression_model




class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise Housing_exception(e, sys) from e

    def get_best_model(self):
        try:
            model = None # first initializing the model=none
            
            #getting the model evaluation file path
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path 

            #there may be a possibility that this filedoesnot exist, so we will first check, if not exist then create it
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model # if not exist then this model will return None
            
            # And if the file exist ,then we weill resd the content of the file
            model_eval_file_content = read_yaml(file_path=model_evaluation_file_path)

            
            #aacha one thing see age file exist krti hui but ye bhi ho sakta hai ki uske andar koi content na ho , to us scenario me we will mmake it empty dict 
            # else we will fill it with the content that is in model_eval_content
            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content


            # now this best_model_key contain the key "best _model " from model_eval_file_content
            # we will check here like if the key is not available in the file ( This means file is there but the best model is missing so in thet case we will return m=None again)
            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            #this will load the content of the file 
            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise Housing_exception(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            ############### IN THIS WE ARE AGAIN GOING TO CHECK THE CONTENT OF EVALUATED MODEL FILE #############################
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            ###########################################################################################################################
            
            
            #   LETS SUPPOSE THE PREVIOUS BEST MODEL IS NONE
            # from below we are getting the previous best model so that if in future we get the best model we will replace it 
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}") # this is the previous content before updation
            #we will maintain the dict that will store the evaluated model details
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            # basically below code se ye pta chl rha hai ki agr phle ka best_model age none nhi hai then hum ek dict prepare krenge(model_history) jisme previous model ka detail hoga 
            #ye krne k bad we will check ki agr phle se hi koi history ho us file me , agr nhi hai then  hum ek dict bnainge jisme HISTORY_KEY ko add krenge aur uski value hogi jo humne 
            # previous_model me dala tha.... Aur agr phle se hi HISTORY_KEY hai then us case me hum HISTORY_KEY ko update kr denge jo hmare pas ab model aya hai usse
            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history) # ab jo nya eval_result aya hai usme history bhi updated ho gya 
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result) # as from line no 93 we can see that noe the eval_result is updated ,it has all the info about history and current best model,so finnaly we will update the content of the page with eval-result
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)# and finally we will write that yaml file

        except Exception as e:
            raise Housing_exception(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            train_dataframe = load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path,
                                                           )
            test_dataframe = load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path,
                                                          )
            schema_content = read_yaml(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]

            metric_info_artifact = evaluate_regression_model(model_list=model_list,
                                                               X_train=train_dataframe,
                                                               y_train=train_target_arr,
                                                               X_test=test_dataframe,
                                                               y_test=test_target_arr,
                                                               base_accuracy=self.model_trainer_artifact.model_accuracy,
                                                               )
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise Housing_exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")
        
        