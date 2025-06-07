import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    # Handle cases where error_detail might be None
    error_message = "Error occurred"
    if error_detail is not None:
        _, _, exc_tb = error_detail.exc_info()  # Extracting traceback details
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name
            error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
                file_name, exc_tb.tb_lineno, str(error)
            )
        else:
            error_message = f"Error occurred: {str(error)} (No traceback available)"
    else:
        error_message = f"Error occurred: {str(error)} (No traceback information)"
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Calling parent constructor of Exception
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

# Test to raise and handle exception
if __name__ == "__main__":
    try:
        a = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        logging.info("Divide by zero error occurred")  # This logs the error message
        raise CustomException(e, sys)  # Raising the custom exception
