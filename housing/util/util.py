import yaml
from housing.exception import Housing_exception
import os,sys
import numpy as np
import dill
import pandas as pd
from housing.constants import *

def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise Housing_exception(e,sys)

def read_yaml(file_path:str):
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise Housing_exception(e,sys) from e
    
# Function Tosave numpy array inside a file
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise Housing_exception(e, sys) from e
    
# function to load the numpy array 
def load_numpy_array_data(file_path: str) :#-> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise Housing_exception(e, sys) from e
    
# function to save the object of transformation phase   
def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise Housing_exception(e,sys) from e
    
    
def load_data(file_path:str,schema_file_path:str):
        """ this function will return -> pd.DataFrame """
        try:
            dataset_schema=read_yaml(schema_file_path)
            # This dataset_schema will contain the info in dict format of the schema_file_path
            # but here we need the datatype so we only have to look at the first key That is columns and this we stored in  DATASET_SCHEMA_COLUMNS_KEY
            # Extracting the datatype
            
            schema= dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]
            
            dataframe=pd.read_csv(file_path)
            
            error_message=""
            
            for column in dataframe.columns:
                if column in list(schema.keys()): # This basically checking the column name like if it is matching with the column name with the daataset or not
                    dataframe[column].astype(schema[column]) # astype is used to convert one datatype to another in  panfdas dataframe 
                    # basically this is saying that if new_dataset column is matching with the schema file column then change the new_dataset column datatype into what is specified inside the schema file
                else:
                    error_message=f"{error_message} \ncolumn: [{column}] is not in schema "
                
            if len(error_message) > 0:
                raise Exception(error_message)
            return dataframe
            
        except Exception as e:
            raise Housing_exception(e,sys) from e
        
        
def load_object(file_path:str):
        """
        file_path: str
        """
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            raise Housing_exception(e,sys) from e
        


