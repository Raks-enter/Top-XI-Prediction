#python -m src.components.data_transformation

import os
import sys
from src.exception import CustomException
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    import os
    train_data_path: str = os.path.join("artifact", "train.csv") #giving these to the ingestion components to store
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "odb.csv") #Input required for the file

class DataIngestion:
    def __init__(self):
        self.ingestion_congif=DataIngestionConfig() #above 3 paths are stored in ingestion_config

    def inititate_data_ingestion(self):  # if your data is stored in some databases, we will write my code over here to read from the database 
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'C:\Users\kollu\Playing XI predictio\notebook\data\odb.csv')

            logging.info("read the dataset as df")

            os.makedirs(os.path.dirname(self.ingestion_congif.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_congif.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_congif.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_congif.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_congif.train_data_path,
                self.ingestion_congif.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    # Assuming DataIngestion provides train and test paths
    obj = DataIngestion()
    train_path, test_path = obj.inititate_data_ingestion()

    # Call DataTransformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
