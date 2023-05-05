from housing.logger import logging
from housing.config.configuration import configuration
from housing.entity.config_entity import DataTransformationconfig
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from housing.exception import Housing_exception
from housing.util.util import read_yaml
from housing.constants import *
from housing.util.util import save_numpy_array_data,load_data,save_object,load_numpy_array_data

import numpy as np
import pandas as pd
import os,sys


# This will be stored inside the constannt folder ,just to visualize i have shown here


# This class is created to generate new feature 
class FeatureGenerator(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room=True,
                 total_rooms_ix=3,
                 population_ix=5,
                 households_ix=6,
                 total_bedrooms_ix=4, 
                 columns=None):
        """
        FeatureGenerator Initialization
        add_bedrooms_per_room: bool
        total_rooms_ix: int index number of total rooms columns
        population_ix: int index number of total population columns
        households_ix: int index number of  households columns
        total_bedrooms_ix: int index number of bedrooms columns
        """
        try:
            
            
            self.columns = columns
            if self.columns is not None:
                total_rooms_ix = self.columns.index(COLUMN_TOTAL_ROOMS) 
                population_ix = self.columns.index(COLUMN_POPULATION)
                households_ix = self.columns.index(COLUMN_HOUSEHOLDS)
                total_bedrooms_ix = self.columns.index(COLUMN_TOTAL_BEDROOM)

            self.add_bedrooms_per_room = add_bedrooms_per_room
            self.total_rooms_ix = total_rooms_ix
            self.population_ix = population_ix
            self.households_ix = households_ix
            self.total_bedrooms_ix = total_bedrooms_ix
        except Exception as e:
            raise Housing_exception(e,sys) from e
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """ This will just create extra column NOTHING ELSE"""
        try:
            room_per_household = X[:, self.total_rooms_ix] / X[:, self.households_ix]
            population_per_household = X[:, self.population_ix] /  X[:, self.households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, self.total_bedrooms_ix] / X[:, self.total_rooms_ix]
                generated_feature = np.c_[X, room_per_household, population_per_household, bedrooms_per_room]
            else:
                generated_feature = np.c_[X, room_per_household, population_per_household]

            return generated_feature
        except Exception as e:
            raise Housing_exception(e,sys) from e



class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationconfig,
                 data_ingestion_artifact=DataIngestionArtifact,
                 data_validation_artifact=DataValidationArtifact) :
        try:
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_artifact=data_validation_artifact
        except Exception as e:
            raise Housing_exception(e,sys) from e
        

        
        
    def get_data_transformer_object(self):
        """ This will return the columnTransformerObject"""
        try:
            schema_file=self.data_validation_artifact.schema_file_path
            dataset_schema=read_yaml(file_path=schema_file)
            
            numerical_column=dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_column=dataset_schema[CATEGORICAL_COLUMN_key]
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('feature_generator', FeatureGenerator(add_bedrooms_per_room=self.data_transformation_config.add_bedroom_per_room,columns=numerical_column)),
                ('scaler', StandardScaler())]
            )
             
            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                 ('one_hot_encoder', OneHotEncoder()),
                 ('scaler', StandardScaler(with_mean=False))
            ]
            )
            logging.info(f"Numerical column={numerical_column}")
            logging.info(f"categorical  column={categorical_column}")
            
            preprocessing=ColumnTransformer([
            ("num_pipeline",num_pipeline,numerical_column),
            ("cat_pipeline",cat_pipeline,categorical_column)])
       
            return preprocessing
        
        except Exception as e:
            raise Housing_exception(e,sys) from e
        
    
    def initiate_data_transformation(self):
        """->DataTransformationArtifact"""
        try:
             # To initiate daata transformation we need following thing
             #step 1):get the preprocessed object 
             #step 2):getting the train and test file path for the transformation
             #step 3)now we will provide our data to a function which will return the dataframe after validating(load_data function in util)
             #step 4) now after loading we will gwt the training and testing but it have target column so we will drop it
             #step 5) Now we will apply the preprocessing object on it
             #step 6) now we will get the transformed array , and then we will concatinate the transformed array with target coumn array 
             #step 7)now we are done with the transformation , now we will store this in our transformed_train_dir and transformed_test_dir
                    # to store that we will have to get the path of dir
                    # After that we will find the file name
            #step 8)now we will save the data inside the transformed_train_dir and transformed_test_dir and this code is written in save_numpy_array_data( in util folder)
            #step 9)see we have saved the transformed data, we will now save our preprocessed object in preprocessed-object_file_path
            # step 10) now we are done with the code now we age good to return our dataTransformationArtifact
             
            #step 1):get the preprocessed object 
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            #step 2):getting the train and test file path for the transformation
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            #step 3)now we will provide our data to a function which will return the dataframe after validating and loading into dataframe(load_data function in util)
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)
            schema = read_yaml(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]


            #step 4) now after loading we will gwt the training and testing but it have target column so we will drop it
            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            #step 5) Now we will apply the preprocessing object on it
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # step 6) now we will get the transformed array , and then we will concatinate the transformed array with target coumn array 
            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            
            #step 7)now we are done with the transformation , now we will store this in our transformed_train_dir and transformed_test_dir
                    # getting transformed_train_dir,transformed_test_dir name
            transformed_train_dir = self.data_transformation_config.tranformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
                    # obtaining file_name
            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")
                    # making The file path
            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
                ###### NOW WE ARE DONE WITH FILE PATH CREATION NOW WE WILL HAVE TO MAKE THE FILE THAT WE HAVE DONE IN  save_numpy_array_data(BELOW)#########



            #step 8)now we will save the data inside the transformed_train_dir and transformed_test_dir and this code is written in save_numpy_array_data( in util folder)
            
            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            #step 9)see we have saved the transformed data, we will now save our preprocessed object in preprocessed-object_file_path
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path
            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)


            # step 10) now we are done with the code now we age good to return our dataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise Housing_exception(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")
    