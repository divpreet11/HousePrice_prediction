from typing import List
from setuptools import setup



#Declaring the variable for setup
project_name="Housing-predictor"
Version="0.0.4"
AUTHOR="Divpreet kaur"
DESCRIPTION="This is Housing price predictor project "
REQUOREMENTS_FILE_NAME="requirements.txt"

def get_requirements_list()->List[str]:
    
    """
    Description: This function is going to return list of requirement 
    mention in requirements.txt file
    return This function is going to return a list which contain name 
    of libraries mentioned in requirements.txt file
    """
    with open(REQUOREMENTS_FILE_NAME) as requirements_file:
        return requirements_file.readline()



setup(
    
name=project_name,
version=Version,
description=DESCRIPTION,
packages=["housing"],
author=AUTHOR,
install_requires=get_requirements_list() 

)
