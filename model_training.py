import vertica_sdk
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump
import time
import os


# number of column including the target
NUMBER_OF_COLUMNS = 6

class LinearRegressionTrain(vertica_sdk.TransformFunction):
    """

    """

    def setup(self, server_interface, col_types):

        # Parameters Reader
        params = server_interface.getParamReader()
        
        # return the parameters used after USING PARAMETERS statement
        self.params = {}
        
        if params.containsParameter('fit_intercept'):
            self.params['fit_intercept'] = params.getBool('fit_intercept')
            
        if params.containsParameter('copy_X'):
            self.params['copy_X'] = params.getBool('copy_X')
        
        if params.containsParameter('n_jobs'):
            self.params['n_jobs'] = params.getInt('n_jobs')
        
        if params.containsParameter('positive'):
            self.params['positive'] = params.getBool('positive')
            
        # Initialize the model
        self.linear_regressor_ = LinearRegression(**self.params)

    def processPartition(self, server_interface, arg_reader, res_writer):

        server_interface.log("Model Training UDx - LR")
        num_rows = arg_reader.getNumRows()
        num_cols = arg_reader.getNumCols()
        server_interface.log("Number of rows of the reader {} ".format(num_rows))
        server_interface.log("Number of columns of the reader {} ".format(num_cols))
        X = []
        count = 0
        while True:
            # check the condition
            if num_rows == 0:
                raise ValueError("Invalid Data: Empty input Data")
                
            elif num_cols != NUMBER_OF_COLUMNS:
                raise ValueError("Invalid Data: input data dimension is not accurate")
            
            else:
                count += 1
                l = []
                for i in range(num_cols):
                    value_i = arg_reader.getFloat(i)
                    if value_i is None:
                      raise ValueError("Invalid Data: Null value")
                    else: 
                      l.append(value_i)
                X.append(l)
                if not arg_reader.next():
                    break
        # covert the input to array
        X = np.array(X)
        # select target (the target should be of index 0 column wise)
        y = X[:, 0]
        # select features
        X = X[:, 1:]
        server_interface.log("Training Started...")
        # model training
        start = time.time()
        self.linear_regressor_.fit(X, y)  # perform linear regression
        # compute the training time
        time_end = 1000 * (time.time() - start)
        server_interface.log("Time model training {} ms on {} rows".format(time_end, count))
        
        # assign a name to the model
        model_name = 'trained_model'
        
        # create a directory to save the model
        dir_path = r'/home/dbadmin/trained_models' 
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # save the model in the trained_models dir
        full_path = dir_path+ '/' + model_name + '.joblib'
        dump(self.linear_regressor_, full_path)
        server_interface.log("Model Saved")
        
        # return the path of trained models
        res_writer.setString(0, 'model_path : '+full_path)
        res_writer.next()
        
    def destroy(self, server_interface, col_types):
        pass


class LinearRegressionTrainFactory(vertica_sdk.TransformFunctionFactory):
    """_summary_

    Args:
        vertica_sdk (_type_): _description_
    """

   
    def getParameterType(self, server_interface, parameterTypes):
    
        parameterTypes.addBool('fit_intercept')
        parameterTypes.addBool('copy_X')
        parameterTypes.addInt('n_jobs')
        parameterTypes.addBool('positive')


    def getPrototype(self, server_interface, arg_types, return_type):
          
        # iterate on the number of variables + target
        for _ in range(NUMBER_OF_COLUMNS):
            arg_types.addFloat()
        return_type.addVarchar()

    
    def getReturnType(self, server_interface, arg_types, return_type):
    
        return_type.addVarchar(300, 'Training Finished')
        

    def createTransformFunction(cls, server_interface):
        
        return LinearRegressionTrain()
        
        