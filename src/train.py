import tf
import os

from data_processor import DataProcesser
from graph_constructor import CNNGraph

class TrainCNN(object):
    def __init__(self, input_x, input_y, input_channel,
                 num_classes, conv_filter_size, conv_nums_filters, fc_layer_size):
        self.total_iter = 0
        self.model_dir = os.path.join(os.getcwd(), '../models')
        self.batch_size = 50

        self.input_x = input_x
        self.input_y = input_y
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.conv_filter_size = conv_filter_size
        self.conv_nums_filters = conv_nums_filters
        self.fc_layer_size = fc_layer_size
        self.optimizer = None
        self.cost = None
        self.cross_entropy = None
        self.data_process = DataProcesser()
        self.graph_constructor = CNNgraph()

    def initialize_graph(self):
        self.graph_constructor.create_graph(self.input_x, self.input_y, self.input_channel, self.num_classes,
                                            self.conv_filter_size, self.nums_filters, self.fc_layer_size)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.graph_constructor.last_layer,
                                                                     labels=self.graph_constructor.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    def read_total_iter(self):
        tt_iter_file = os.path.join(self.model_dir, 'total_iteration.txt')
        with open(tt_iter_file, 'r') as f:
            try:
                self.total_iter = int(f.read())
            except:
                print('Error reading the total number of iterations. Assigning zero')
                self.total_iter = 0

    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss, session):
        acc = session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))
                
    def train(self, num_iterations):
        ''' Train the network for certain number of iterations
        param:
        num_iterations: The number of iterations to train
        '''
        self.initialize_graph()
        self.read_total_iter()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for i in range(self.total_iter, self.total_iter + num_iterations):
                x_batch_train, y_batch_train = self.data_process.next_training_batch(self.batch_size)
                x_batch_valid, y_batch_valid = self.data_process.next_validate_batch(self.batch_size)
                feed_dict_train = {x: x_batch_train
                                   y_true: y_batch_train}
                feed_dict_valid = {x: x_batch_valid,
                                   y_true: y_batch_valid}
                session.run(self.optimizer, feed_dict=feed_dict_train)
                if i % int(data.train.num_examples/batch_size) == 0: 
                    val_loss = session.run(cost, feed_dict=feed_dict_val)
                    epoch = int(i / int(data.train.num_examples/batch_size))    
            
                    show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, session)
                    saver.save(session, 'dogs-cats-model') 


                total_iterations += num_iteration
