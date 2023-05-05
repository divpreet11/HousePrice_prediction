import os
import sys
import traceback

class Housing_exception(Exception):
    
    def __init__(self,error_message:Exception,error_details:sys):
        # sys module have all thr info regarding the error msj in which file and in which line it is created
        # this super().__init__(error_message) is same as Exception(error_message), it is providing "error_message" to the __init__ if exception class that is parent class
        super().__init__(error_message)
        self.error_message=Housing_exception.get_detailed_error_message(error_message=error_message,error_details=error_details)
     
        # we are creating a static method  because static methid are those which can be called without creating a object we can directly call it by class name 
        # we do not have to initialize again and again--> see google 
    @staticmethod 
    def get_detailed_error_message(error_message:Exception,error_details:sys)->str: 
        
        """
        error_message: object of Exception class
        error_details: object of sys module
        
        """
        
        #this exe_info will return 3 values in the form of tuple but we want only the traceback so we have ignored the first 2 variable by _,_
        _,_,traceback_of_error=error_details.exc_info()
        
        #This will give line number of the error 
        try_block_line_no=traceback_of_error.tb_lineno
        exception_block_line_number=traceback_of_error.tb_frame.f_lineno
        
        # this will give the file name of the error
        filename_of_error=traceback_of_error.tb_frame.f_code.co_filename
        
        error_message= f"""Error occured in script: [ {filename_of_error} ] 
        in try_block_line_no: [ {try_block_line_no} ],
        exception block line number:[ {exception_block_line_number} ],
        error message is [{error_message}] """
        return error_message
        
    def __str__(self) -> str:
        return self.error_message


    def __repr__(self) -> str:
        return Housing_exception.__name__.str()