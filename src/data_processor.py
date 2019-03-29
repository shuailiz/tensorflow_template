#!/bin/usr/python

class DataProcesser(object):

    def __init__(self, class_names):
        self.class_names = class_names
        self.total_num_data_points = 0
        self.data_loaded = False

        self.training_data_start_ind = 0
        self.training_data_end_ind = 0
        self.valid_data_start_ind = 0
        self.valid_data_start_ind = 0
        self.training_ind_in_epoch = 0
        self.valid_ind_in_epoch = 0

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

    def divide_training_valid_data(self, training_size):
        ''' divide the data into training and validation for tensor flow network
        param:
        training_size (double): The the size of the training set in the whole data set
        return:
        training_data: The data for training
        '''
        training_data_num = int(training_size * self.total_num_data_points)
        self.training_data_start_ind = 0
        self.training_data_end_ind = training_data_num - 1
        self.valid_data_start_ind = training_data_num
        self.valid_data_end_ind = self.total_num_data_points - 1
        self.training_ind_in_epoch = self.training_data_start_ind
        self.valid_ind_in_epoch = self.valid_data_start_ind

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
