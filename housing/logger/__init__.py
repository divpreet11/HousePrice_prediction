import logging
import os
from datetime import date, datetime
import pandas as pd



#creating logging directory
log_dir="Housing_log"


current_time_stamp= f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

#making log file with the name of time and date of creation log
log_file=f"log_{current_time_stamp}.log"

#making directory , if already exist then do nothing otherwise create the directory
os.makedirs(log_dir,exist_ok=True)

#joining the file path with directory by using os module
file_path= os.path.join(log_dir,log_file)


logging.basicConfig(filename=file_path,
filemode="w",
format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
level=logging.INFO
   
    )

def get_log_dataframe(file_path):
    data=[]
    with open(file_path) as log_file:
        for line in log_file.readlines():
            data.append(line.split("^;"))

    log_df = pd.DataFrame(data)
    columns=["Time stamp","Log Level","line number","file name","function name","message"]
    log_df.columns=columns
    
    log_df["log_message"] = log_df['Time stamp'].astype(str) +":$"+ log_df["message"]

    return log_df[["log_message"]]