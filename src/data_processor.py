#!/bin/usr/python

class DataProcesser(object):

    def __init__(self):
        pass

    def read_data_from_folder(self, folder_path, data_type):
        ''' Read data from a folder
        param:
        folder_path (str): path to the folder
        data_type (enum): the type of the data
        return:
        input_data: the data read from the folder
        '''
        pass

    def read_data_from_hdf5(self, hdf5_file):
        ''' Read data from a hdf5 file
        param:
        hdf5_file: The hdf5 file handle
        return:
        input_data: The data read from the hdf5 file
        '''
        pass

    def read_data_from_file(self, file_handle, data_type):
        ''' Read data from a file
        param:
        file_handle: a file reader handle
        data_type (enum): the type of the data to read
        return:
        input_data: the data read from the file handle
        '''
        pass

    def save_data_to_hdf5(self, data, hadf5_file):
        ''' Save data to a hdf5 file
        param:
        data: The data to be saved
        hdf5_file: The file handle to the hdf5 file
        '''
        pass

    def validate_data_point(self, data_point):
        ''' Validate a data point 
        param:
        data_point: the data point to validate
        '''
        pass

    def prepare_training_data(self, data_set):
        ''' Prepare training data for tensor flow network
        param:
        data_set: The whole data set
        return:
        training_data: The data for training
        '''
        pass

    def prepare_testing_data(self, data_set):
        ''' Prepare testing data for tensor flow network
        param:
        data_set: The whole data set
        return:
        testing_data: The data for testing
        '''
        pass 

    def next_training_batch(self, batch_size):
        ''' Generate the next batch of data for training
        param:
        batch_size: How large the batch will be
        return:
        training_batch: The batch of training data
        '''
        pass

    def next_validate_batch(self, batch_size):
        ''' Generate the next batch of data for validation
        param:
        batch_size: How large the batch will be
        return:
        validation_batch: The batch of validation data
        '''
        pass
