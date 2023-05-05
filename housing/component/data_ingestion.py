import imp
from housing.config.configuration import configuration
from housing.entity.config_entity import DataIngestionconfig
from housing.exception import Housing_exception
from housing.logger import logging
import sys,os
from sklearn.model_selection import StratifiedShuffleSplit
from housing.entity.artifact_entity import DataIngestionArtifact
import numpy as np
import tarfile  # Python tarfile module is used to read and write tar archives.
from six.moves import urllib
import pandas as pd
class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionconfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise Housing_exception(e,sys) from e #from e likhne ka matlab ye hai ki jab hume exception aaiga to usme ye na show ho ki wo isi line se exception raise hua hai wo actual jha se exception raise hua hai wha point kre 
        
        
    def download_housing_data(self):
        try:
            #raise Exception("Testing Exception")
            ## accessing the url to download the content
            download_url=self.data_ingestion_config.dataset_download_url
            
            ## Extracting the tgz download directory that we have already created in out config.configuration file
            tgz_download_dir=self.data_ingestion_config.tgz_download_dir
            
        
                
            ## Now focus that the tgz_download_dir is not available now because in component we just have created the path we 
            ## have not created any kind of folder , folder creation will be done in this section 
            ## But there is also one thing that of tgz_download_dir exist but what if it contain a unuseful thing for us
            ## to solve this we will delete that file and then again create it 
            
            if os.path.exists(tgz_download_dir):
                os.remove(tgz_download_dir)
                
            os.makedirs(tgz_download_dir,exist_ok=True) # now we will crete the folder freshly
            
            # we are creating a housing_file_name which we are extracting from the download_url 
            # There is a method in os modeule which will give the name of the file from the url 
            # we can also use the url.split("/")[-1] this will also work
            housing_file_name=os.path.basename(download_url)
            
            # see dont get confused when u see this, because in configuration we just have created the root directory now 
            # we will go inside that and will create the actual functioning for this we will create a file in tgz download dir
            
            tgz_file_path=os.path.join(tgz_download_dir,housing_file_name) # this is the place where we download oue tgz file
            
            logging.info(f"Downloading file from [{download_url}] at location [{tgz_file_path}]")
           
            urllib.request.urlretrieve(download_url,tgz_file_path)  #This will download the content of the url and will store inside the tgz_file_path
            logging.info(f"Downloading completed sucessfully at location : [{tgz_file_path}]")
            
            # now as we have downloaded the file  at tgz_file_path we can return that path
            return tgz_file_path   
            
        except Exception as e:
            raise Housing_exception(e,sys) from e
        
        
    def extract_tgz_file(self,tgz_file_path):
        """ This will take a parameter tgz_file_path and try to extract  the data into the raw_data_dir"""
        try:
            raw_data_dir=self.data_ingestion_config.raw_data_dir # This will give the raw_data_dir where er hsve to store the Extracted data
            
            # it will check the same thing as in above method
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
                
            os.makedirs(raw_data_dir,exist_ok=True)
            
            
            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir [{raw_data_dir}]")
            # NOW WE WILL EXTRACT THE FILE AND THEN WE WILL STORE IN THE raw_data_dir FOR THIS WE WILL USE  THE TARFILE MODULE 
            with tarfile.open(tgz_file_path) as housing_tgz_file_object:
                housing_tgz_file_object.extractall(path=raw_data_dir) # in this we are extracting the data and storing it into the raw_data_dir
            logging.info(f"Extraction completed!!!")
            
        except Exception as e:
            raise Housing_exception(e,sys) from e
    
    def split_data_as_train_test(self):
        """ This function will give -> DataIngestionArtifact as output because this will \
            provode train and test data to the data validation component"""
        try:
            raw_data_dir=self.data_ingestion_config.raw_data_dir
            # This above code is only for directory , but we will have to reach the file name 
            # for that we will navigate to the directory and then extract the file path
            # for that we will use the os.listdir func that will give the list of files present inside the dir
            # from there we will extract the file name and then we will join it with the raw_data_dir to get the complete 
            # path of the csv file so that we can split
            
            file_name=os.listdir(raw_data_dir)[0] # we are using [0] because raw_data_dir have only one dile and we need to extract it so we use 0 as index
            
            housing_file_path=os.path.join(raw_data_dir,file_name)
            
            # now we reached to the actual file where the data is stored now we will read that file 
            
            housing_data_frame=pd.read_csv(housing_file_path)
            
            logging.info(f"Reading csv file: [{housing_file_path}]")

            housing_data_frame["income_cat"] = pd.cut(  # This is the dependent variable 
                housing_data_frame["median_income"],
                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=[1,2,3,4,5]
            )
            

            logging.info(f"Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None
            
            
            
            # Thos split will create a object of StratifiedShuffleSplit will will take n_split as parameter 
            # and the distribution of train and test will remail same
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



            # see from below code we will use the above object  to split the data , but to split the data we will have to
            # use 2 parameter (dependent and independent variable) but housing data frame is not a independent variable 
            # 
            for train_index,test_index in split.split(housing_data_frame, housing_data_frame["income_cat"]):
                strat_train_set = housing_data_frame.loc[train_index].drop(["income_cat"],axis=1)
                strat_test_set = housing_data_frame.loc[test_index].drop(["income_cat"],axis=1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                            file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                        file_name)
            
            # noe we have data as strat_train_set and strat_test_set now we will dump the data at train_file pth and test_file_path
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path,index=False)
            

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                test_file_path=test_file_path,
                                is_ingested=True,
                                message=f"Data ingestion completed successfully."
                                )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact
         
        except Exception as e:
            raise Housing_exception(e,sys) from e
        
    def initiate_data_ingestion(self):
        """This wrill return DataIngestionArtifact"""
        try: 
            tgz_file_path=self.download_housing_data() # This class will return the file path see above code
            self.extract_tgz_file(tgz_file_path=tgz_file_path) # here we are just passing the parameter to the extract_tgz_file() class
            return self.split_data_as_train_test()
        except Exception as e:
            raise Housing_exception(e,sys) from e
        
    def __del__(self):
            logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")
        
   