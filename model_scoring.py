import vertica_sdk
import numpy as np
from joblib import load


# number of column including the target
NUMBER_OF_COLUMNS = 6

class ModelScoring(vertica_sdk.ScalarFunction):
    """
    Scalar function which lemmatize its inputs.
    For each input string, each of the tokens of that
    string is lemmatizer to its root.
    """
        
    def setup(self, server_interface, col_types):

        # Parameters Reader
        params = server_interface.getParamReader()
        
        # return the parameters used after USING PARAMETERS statement
        self.params = {}
        # passsing the model name
        if params.containsParameter('model_name'):
            self.params['model_name'] = params.getString('model_name')
        else:
            raise ValueError('Model_name is missing')
            
        dir_path = r'/home/dbadmin/trained_models'            
        # Load the model
        self.my_model = load(dir_path+'/'+self.params['model_name'] + '.joblib')
    


    def processBlock(self, server_interface, arg_reader, res_writer):
        
        
        server_interface.log("Model Scoring UDx - LR")
        num_rows = arg_reader.getNumRows()
        num_cols = arg_reader.getNumCols()
        server_interface.log("Number of rows of the reader {} ".format(num_rows))
        server_interface.log("Number of columns of the reader {} ".format(num_cols))
    
        while True:
            
            # check the condition
            if num_rows == 0:
                raise ValueError("Invalid Data: Empty input data")
                
            elif num_cols != NUMBER_OF_COLUMNS-1:
                raise ValueError("Invalid Data: Input data dimension is not acccurate")  
                              
            else:    
                row = []           
                for i in range(num_cols):
                    # read all the data in each row
                    value_i = arg_reader.getFloat(i)
                    if value_i is None:
                        raise ValueError("Invalid Data: NULL value found")
                    else:
                        row.append(value_i)
                # predictions
                pred = self.my_model.predict(np.array([row]))
                # write the results for ean row
                res_writer.setFloat(pred)
                res_writer.next()
                if not arg_reader.next():    
                    break


class ModelScoringFactory(vertica_sdk.ScalarFunctionFactory):


    def getParameterType(self, server_interface, parameterTypes):
        parameterTypes.addVarchar(100,'model_name')
        

    def createScalarFunction(cls, server_interface):
        return ModelScoring()

    def getPrototype(self, server_interface, arg_types, return_type):
    
        # iterate the number of variable
        for _ in range(NUMBER_OF_COLUMNS-1):
            arg_types.addFloat()
        return_type.addFloat()
        
    def getReturnType(self, server_interface, arg_types, return_type):
        return_type.addFloat("Predictions")