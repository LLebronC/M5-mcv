import math
import time
import numpy as np
import tensorflow as tf
import cv2
import os
import scipy.misc
from keras.engine.training import GeneratorEnqueuer
#from model_factory import Model_Factory
from tools.save_images import save_img3
from tools.yolo_utils import yolo_postprocess_net_out, yolo_draw_detections


"""
Interface for normal (one net) models and adversarial models. Objects of
classes derived from Model are returned by method make() of the Model_Factory
class.
"""
class Model():
    def train(self, train_gen, valid_gen, cb):
        pass

    def predict(self, test_gen, tag='pred'):
        pass

    def test(self, test_gen):
        pass


"""
Wraper of regular models like FCN, SegNet etc consisting of a one Keras model.
But not GANs, which are made of two networks and have a different training
strategy.
In this class we implement the train(), test() and predict() methods common to
all of them.
"""
# TODO: Better call it Regular_Model ?
class One_Net_Model(Model):
    def __init__(self, model, cf, optimizer):
        self.cf = cf
        self.optimizer = optimizer
        self.model = model

    # Train the model
    def train(self, train_gen, valid_gen, cb):
        if (self.cf.train_model):
            print('\n > Training the model...')
            hist = self.model.fit_generator(generator=train_gen,
                                            samples_per_epoch=self.cf.dataset.n_images_train,
                                            nb_epoch=self.cf.n_epochs,
                                            verbose=1,
                                            callbacks=cb,
                                            validation_data=valid_gen,
                                            nb_val_samples=self.cf.dataset.n_images_valid,
                                            class_weight=None,
                                            max_q_size=10,
                                            nb_worker=1,
                                            pickle_safe=False)
            print('   Training finished.')

            return hist
        else:
            return None

    # Predict the model
    def predict(self, test_gen, tag='pred', max_q_size=10, workers=1, pickle_safe=False, wait_time = 0.01):
        if self.cf.pred_model and test_gen is not None:
            # TODO fix model predict method
            print('\n > Predicting the model...')
            aux =  'image_result'
            result_path = os.path.join(self.cf.savepath,aux)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            # Load best trained model
            # self.model.load_weights(os.path.join(self.cf.savepath, "weights.hdf5"))
            self.model.load_weights(self.cf.weights_file)
            priors = self.cf.dataset.priors
            anchors = np.array(priors)
            thresh = 0.6
            nms_thresh = 0.3
            classes = self.cf.dataset.classes
            # Create a data generator
            data_gen_queue = GeneratorEnqueuer(test_gen, pickle_safe=pickle_safe)
            data_gen_queue.start(workers, max_q_size)
            # Process the dataset
            start_time = time.time()
            image_counter = 1
            for _ in range(int(math.ceil(self.cf.dataset.n_images_train/float(self.cf.batch_size_test)))):
                data = None
                while data_gen_queue.is_running():
                    if not data_gen_queue.queue.empty():
                        data = data_gen_queue.queue.get()
                        break
                    else:
                        time.sleep(wait_time)               
                x_true = data[0]
                y_true = data[1].astype('int32')

                # Get prediction for this minibatch
                y_pred = self.model.predict(x_true)
                if self.cf.model_name == "yolo" or self.cf.model_name == "tiny-yolo":
                    for i in range(len(y_pred)):                  
                        boxes = yolo_postprocess_net_out(y_pred[i], anchors, classes, thresh, nms_thresh)
                        '''print len(boxes)
                        print (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h, boxes[0].c, boxes[0].probs)'''
                        #img = x_true[i]*255
                        
                        im = yolo_draw_detections(boxes, x_true[i], anchors, classes, thresh, nms_thresh)
                        out_name = os.path.join(result_path, 'img_' + str(image_counter).zfill(4)+ '.png')
                        scipy.misc.toimage(im).save(out_name)
                        image_counter = image_counter+1
                '''save_img3(x_true, y_true, y_pred, self.cf.savepath, 0,
                          self.cf.dataset.color_map, self.cf.dataset.classes, tag+str(_), self.cf.dataset.void_class)'''

            # Stop data generator
            data_gen_queue.stop()

            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Predicting time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))

    # Test the model
    def test(self, test_gen):
        if self.cf.test_model and test_gen is not None:
            print('\n > Testing the model...')
            # Load best trained model
            self.model.load_weights(self.cf.weights_test_file)

            # Evaluate model
            start_time = time.time()
            test_metrics = self.model.evaluate_generator(test_gen,
                                                         self.cf.dataset.n_images_test,
                                                         max_q_size=10,
                                                         nb_worker=1,
                                                         pickle_safe=False)
            total_time = time.time() - start_time
            fps = float(self.cf.dataset.n_images_test) / total_time
            s_p_f = total_time / float(self.cf.dataset.n_images_test)
            print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time, fps, s_p_f))
            metrics_dict = dict(zip(self.model.metrics_names, test_metrics))
            print ('   Test metrics: ')
            for k in metrics_dict.keys():
                print ('      {}: {}'.format(k, metrics_dict[k]))

            if self.cf.problem_type == 'segmentation':
                # Compute Jaccard per class
                metrics_dict = dict(zip(self.model.metrics_names, test_metrics))
                I = np.zeros(self.cf.dataset.n_classes)
                U = np.zeros(self.cf.dataset.n_classes)
                jacc_percl = np.zeros(self.cf.dataset.n_classes)
                for i in range(self.cf.dataset.n_classes):
                    I[i] = metrics_dict['I'+str(i)]
                    U[i] = metrics_dict['U'+str(i)]
                    jacc_percl[i] = I[i] / U[i]
                    print ('   {:2d} ({:^15}): Jacc: {:6.2f}'.format(i,
                                                                     self.cf.dataset.classes[i],
                                                                     jacc_percl[i]*100))
                # Compute jaccard mean
                jacc_mean = np.nanmean(jacc_percl)
                print ('   Jaccard mean: {}'.format(jacc_mean))
                
    def logistic_activate_tensor(x):
        return 1. / (1. + tf.exp(-x))