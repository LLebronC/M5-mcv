import os
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
from tools.yolo_utils import *
from keras.preprocessing import image
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
            print('\n > Predicting the model...')
            aux =  'image_result'
            result_path = os.path.join(self.cf.savepath,aux)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            # Load best trained model
            # self.model.load_weights(os.path.join(self.cf.savepath, "weights.hdf5"))
            self.model.load_weights(self.cf.weights_file)
            
            if self.cf.problem_type == 'detection':
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
                            #Process the YOLO output to obtain final BBox per image                  
                            boxes = yolo_postprocess_net_out(y_pred[i], anchors, classes, thresh, nms_thresh)
                            #Draw the Bbox in the image to visualize
                            im = yolo_draw_detections(boxes, x_true[i], anchors, classes, thresh, nms_thresh)
                            out_name = os.path.join(result_path, 'img_' + str(image_counter).zfill(4)+ '.png')
                            scipy.misc.toimage(im).save(out_name)
                            image_counter = image_counter+1
                    elif self.cf.model_name == "ssd":
                        results = self.cf.bbox_util.detection_out(y_pred)
                        for j in range(len(results)):
                            # Parse the outputs.
                            if np.any(results[j]):
                                det_label = results[j][:, 0]
                                det_conf = results[j][:, 1]
                                det_xmin = results[j][:, 2]
                                det_ymin = results[j][:, 3]
                                det_xmax = results[j][:, 4]
                                det_ymax = results[j][:, 5]
                            
                                # Get detections with confidence higher than 0.6.
                                top_indices = [i for i, conf in enumerate(det_conf) if conf >= thresh]
                                top_conf = det_conf[top_indices]
                                top_label_indices = det_label[top_indices].tolist()
                                top_xmin = det_xmin[top_indices]
                                top_ymin = det_ymin[top_indices]
                                top_xmax = det_xmax[top_indices]
                                top_ymax = det_ymax[top_indices]
                                out_name = os.path.join(result_path, 'img_' + str(image_counter).zfill(4)+ '.png')
                                if top_indices:
                                    im = self.cf.bbox_util.ssd_draw_detections(top_conf, top_label_indices, top_xmin, top_ymin,
                                                                           top_xmax, top_ymax, x_true[j], classes, out_name)
                                
                                #scipy.misc.toimage(im).save(out_name)
                                #im.savefig(out_name)
                                image_counter = image_counter+1
                    else:
                        raise ValueError("No model name defined or valid: " + self.model_name)
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
            start_time_global = time.time()
            test_metrics = self.model.evaluate_generator(test_gen,
                                                         self.cf.dataset.n_images_test,
                                                         max_q_size=10,
                                                         nb_worker=1,
                                                         pickle_safe=False)
            if self.cf.problem_type == 'detection':
                # Dataset and the model used
                dataset_name = self.cf.dataset_name 
                #model_name = self.cf.model_name 
                # Net output post-processing needs two parameters:
                detection_threshold = 0.6 # Min probablity for a prediction to be considered
                nms_threshold       = 0.2 # Non Maximum Suppression threshold
                # IMPORTANT: the values of these two params will affect the final performance of the netwrok
                #            you are allowed to find their optimal values in the validation/train sets
                
                if dataset_name == 'TT100K_detection':
                    classes = ['i2','i4','i5','il100','il60','il80','io','ip','p10','p11','p12','p19','p23','p26','p27','p3','p5','p6','pg','ph4','ph4.5','ph5','pl100','pl120','pl20','pl30','pl40','pl5','pl50','pl60','pl70','pl80','pm20','pm30','pm55','pn','pne','po','pr40','w13','w32','w55','w57','w59','wo']
                elif dataset_name == 'Udacity':
                    classes = ['Car','Pedestrian','Truck']
                else:
                    print "Error: Dataset not found!"
                    quit()
                priors = [[0.9,1.2], [1.05,1.35], [2.15,2.55], [3.25,3.75], [5.35,5.1]]
                
                input_shape = (self.cf.dataset.n_channels,
                        self.cf.target_size_test[0],
                        self.cf.target_size_test[1])
                

                
                test_dir = test_gen.directory
                imfiles = [os.path.join(test_dir,f) for f in os.listdir(test_dir) 
                                    if os.path.isfile(os.path.join(test_dir,f)) 
                                    and f.endswith('jpg')]
                inputs = []
                img_paths = []
                chunk_size = 128 # we are going to process all image files in chunks
                
                ok = 0.
                total_true = 0.
                total_pred = 0.
                for i, img_path in enumerate(imfiles):
                    img = image.load_img(img_path, target_size=(input_shape[1], input_shape[2]))
                    img = image.img_to_array(img)
                    img = img / 255.
                    inputs.append(img.copy())
                    img_paths.append(img_path)
                    
                    if len(img_paths)%chunk_size == 0 or i+1 == len(imfiles):
                        inputs = np.array(inputs)
                        start_time_batch = time.time()
                        net_out = self.model.predict(inputs, batch_size = 16, verbose = 1)
                        print ('{} images predicted in {:.5f} seconds. {:.5f} fps').format(len(inputs), 
                               time.time() - start_time_batch, 
                                (len(inputs)/(time.time() - start_time_batch)))
                        # Find correct detections (per image)
                        for i, img_path in enumerate(img_paths):
                            boxes_pred = yolo_postprocess_net_out(net_out[i], priors, classes, detection_threshold, nms_threshold)
                            boxes_true = []
                            label_path = img_path.replace('jpg','txt')
                            gt = np.loadtxt(label_path)
                            if len(gt.shape) == 1:
                                gt = gt[np.newaxis,]
                            for j in range(gt.shape[0]):
                                bx = BoundBox(len(classes))
                                bx.probs[int(gt[j,0])] = 1.
                                bx.x, bx.y, bx.w, bx.h = gt[j,1:].tolist()
                                boxes_true.append(bx)
                            
                            total_true += len(boxes_true)
                            true_matched = np.zeros(len(boxes_true))
                            for b in boxes_pred:
                                if b.probs[np.argmax(b.probs)] < detection_threshold:
                                    continue
                                total_pred += 1.
                                for t,a in enumerate(boxes_true):
                                    if true_matched[t]:
                                        continue
                                    if box_iou(a, b) > 0.5 and np.argmax(a.probs) == np.argmax(b.probs):
                                        true_matched[t] = 1
                                        ok += 1.
                                        break
                            # You can visualize/save per image results with this:
                            #im = cv2.imread(img_path)
                            #im = yolo_draw_detections(boxes_pred, im, priors, classes, detection_threshold, nms_threshold)
                            #cv2.imshow('', im)
                            #cv2.waitKey(0)
                        inputs = []
                        img_paths = []
                    
                        #print 'total_true:',total_true,' total_pred:',total_pred,' ok:',ok
                        p = 0. if total_pred == 0 else (ok/total_pred)
                        r = ok/total_true
                        print('Precission = ' + str(p))
                        print('Recall     = ' + str(r))
                        f = 0. if (p + r) == 0 else (2*p*r/(p + r))
                        print('F-score    = '+str(f))
    
    
            total_time_global = time.time() - start_time_global
            fps = float(self.cf.dataset.n_images_test) / total_time_global
            s_p_f = total_time_global / float(self.cf.dataset.n_images_test)
            print ('   Testing time: {}. FPS: {}. Seconds per Frame: {}'.format(total_time_global, fps, s_p_f))
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
