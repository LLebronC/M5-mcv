import numpy as np
from keras import backend as K
import tensorflow as tf

dim_ordering = K.image_dim_ordering()
if dim_ordering == 'th':
    import theano
    from theano import tensor as T
else:
    import tensorflow as tf
    from tensorflow.python.framework import ops

def cce_flatt(void_class, weights_class):
    def categorical_crossentropy_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)

        if dim_ordering == 'th':
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        else:
            y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01

        # remove void classes from cross_entropy
        if len(void_class):
            for i in range(len(void_class)):
                # get idx of non void classes and remove void classes
                # from y_true and y_pred
                idxs = K.not_equal(y_true, void_class[i])
                if dim_ordering == 'th':
                    idxs = idxs.nonzero()
                    y_pred = y_pred[idxs]
                    y_true = y_true[idxs]
                else:
                    y_pred = tf.boolean_mask(y_pred, idxs)
                    y_true = tf.boolean_mask(y_true, idxs)

        if dim_ordering == 'th':
            y_true = T.extra_ops.to_one_hot(y_true, nb_class=y_pred.shape[-1])
        else:
            y_true = tf.one_hot(y_true, K.shape(y_pred)[-1], on_value=1, off_value=0, axis=None, dtype=None, name=None)
            y_true = K.cast(y_true, 'float32')  # b,01 -> b01
        out = K.categorical_crossentropy(y_pred, y_true)

        # Class balancing
        if weights_class is not None:
            weights_class_var = K.variable(value=weights_class)
            class_balance_w = weights_class_var[y_true].astype(K.floatx())
            out = out * class_balance_w

        return K.mean(out)  # b01 -> b,01
    return categorical_crossentropy_flatt


def IoU(n_classes, void_labels):
    def IoU_flatt(y_true, y_pred):
        '''Expects a binary class matrix instead of a vector of scalar classes.
        '''
        if dim_ordering == 'th':
            y_pred = K.permute_dimensions(y_pred, (0, 2, 3, 1))
        shp_y_pred = K.shape(y_pred)
        y_pred = K.reshape(y_pred, (shp_y_pred[0]*shp_y_pred[1]*shp_y_pred[2],
                           shp_y_pred[3]))  # go back to b01,c
        # shp_y_true = K.shape(y_true)
        y_true = K.cast(K.flatten(y_true), 'int32')  # b,01 -> b01
        y_pred = K.argmax(y_pred, axis=-1)

        # We use not_void in case the prediction falls in the void class of
        # the groundtruth
        for i in range(len(void_labels)):
            if i == 0:
                not_void = K.not_equal(y_true, void_labels[i])
            else:
                not_void = not_void * K.not_equal(y_true, void_labels[i])

        sum_I = K.zeros((1,), dtype='float32')

        out = {}
        for i in range(n_classes):
            y_true_i = K.equal(y_true, i)
            y_pred_i = K.equal(y_pred, i)

            if dim_ordering == 'th':
                I_i = K.sum(y_true_i * y_pred_i)
                U_i = K.sum(T.or_(y_true_i, y_pred_i) * not_void)
                # I = T.set_subtensor(I[i], I_i)
                # U = T.set_subtensor(U[i], U_i)
                sum_I = sum_I + I_i
            else:
                U_i = K.sum(K.cast(tf.logical_and(tf.logical_or(y_true_i, y_pred_i), not_void), 'float32'))
                y_true_i = K.cast(y_true_i, 'float32')
                y_pred_i = K.cast(y_pred_i, 'float32')
                I_i = K.sum(y_true_i * y_pred_i)
                sum_I = sum_I + I_i
            out['I'+str(i)] = I_i
            out['U'+str(i)] = U_i

        if dim_ordering == 'th':
            accuracy = K.sum(sum_I) / K.sum(not_void)
        else:
            accuracy = K.sum(sum_I) / tf.reduce_sum(tf.cast(not_void, 'float32'))
        out['acc'] = accuracy
        return out
    return IoU_flatt



"""
    YOLO loss function
    code adapted from https://github.com/thtrieu/darkflow/
"""

def logistic_activate_tensor(x):
    return 1. / (1. + tf.exp(-x))

def YOLOLoss(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,object_scale=5.0,noobject_scale=1.0,coord_scale=1.0,class_scale=1.0):

  # Def custom loss function using numpy
  def _YOLOLoss(y_true, y_pred, name=None, priors=priors):

      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = logistic_activate_tensor(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2]))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = logistic_activate_tensor(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.pow(coords[:,:,:,2:4], 2) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU, set 0.0 confidence for worse boxes
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
      #best_box = tf.grater_than(iou, 0.5) # LLUIS ???
      best_box = tf.to_float(best_box)
      confs = tf.multiply(best_box, _confs)

      # take care of the weight terms
      conid = noobject_scale * (1. - confs) + object_scale * confs
      weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
      cooid = coord_scale * weight_coo
      weight_pro = tf.concat(num_classes * [tf.expand_dims(confs, -1)], 3)
      proid = class_scale * weight_pro

      true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs ], 3)
      wght = tf.concat([cooid, tf.expand_dims(conid, 3), proid ], 3)

      loss = tf.pow(adjusted_net_out - true, 2)
      loss = tf.multiply(loss, wght)
      loss = tf.reshape(loss, [-1, h*w*b*(4 + 1 + num_classes)])
      loss = tf.reduce_sum(loss, 1)

      return .5*tf.reduce_mean(loss)

  return _YOLOLoss


"""
    YOLO detection metrics
    code adapted from https://github.com/thtrieu/darkflow/
"""
def YOLOMetrics(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3):

  def _YOLOMetrics(y_true, y_pred, name=None):

      net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

      _,h,w,c = net_out.get_shape().as_list()
      b = len(priors)
      anchors = np.array(priors)

      _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes,1,4,1,2,2], axis=3)
      _confs = tf.squeeze(_confs,3)
      _areas = tf.squeeze(_areas,3)

      net_out_reshape = tf.reshape(net_out, [-1, h, w, b, (4 + 1 + num_classes)])
      # Extract the coordinate prediction from net.out
      coords = net_out_reshape[:, :, :, :, :4]
      coords = tf.reshape(coords, [-1, h*w, b, 4])
      adjusted_coords_xy = logistic_activate_tensor(coords[:,:,:,0:2])
      adjusted_coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4]) * np.reshape(anchors, [1, 1, b, 2]) / np.reshape([w, h], [1, 1, 1, 2]))
      coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

      adjusted_c = logistic_activate_tensor(net_out_reshape[:, :, :, :, 4])
      adjusted_c = tf.reshape(adjusted_c, [-1, h*w, b, 1])

      adjusted_prob = tf.nn.softmax(net_out_reshape[:, :, :, :, 5:])
      adjusted_prob = tf.reshape(adjusted_prob, [-1, h*w, b, num_classes])

      adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

      wh = tf.pow(coords[:,:,:,2:4], 2) *  np.reshape([w, h], [1, 1, 1, 2])
      area_pred = wh[:,:,:,0] * wh[:,:,:,1]
      centers = coords[:,:,:,0:2]
      floor = centers - (wh * .5)
      ceil  = centers + (wh * .5)

      # calculate the intersection areas
      intersect_upleft   = tf.maximum(floor, _upleft)
      intersect_botright = tf.minimum(ceil, _botright)
      intersect_wh = intersect_botright - intersect_upleft
      intersect_wh = tf.maximum(intersect_wh, 0.0)
      intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])

      # calculate the best IOU and metrics 
      iou = tf.truediv(intersect, _areas + area_pred - intersect)
      best_ious     = tf.reduce_max(iou, [2], True)
      recall        = tf.reduce_sum(tf.to_float(tf.greater(best_ious,0.5)), [1])
      sum_best_ious = tf.reduce_sum(best_ious, [1])
      gt_obj_areas  = tf.reduce_mean(_areas, [2], True)
      num_gt_obj    = tf.reduce_sum(tf.to_float(tf.greater(gt_obj_areas,tf.zeros_like(gt_obj_areas))), [1])
      avg_iou       = tf.truediv(sum_best_ious, num_gt_obj)
      avg_recall    = tf.truediv(recall, num_gt_obj)
 
      return {'avg_iou':tf.reduce_mean(avg_iou), 'avg_recall':tf.reduce_mean(avg_recall)}
  return _YOLOMetrics

"""
    SSD detection metrics
"""
def SSDLoss(input_shape=(3,300,300),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3,alpha=1.0, neg_pos_ratio=3.0, background_label_id=0, negatives_for_hard=100.0):
    def _l1_smooth_loss(y_true, y_pred):
        """Compute L1-smooth loss.
        # Arguments
            y_true: Ground truth bounding boxes,
                tensor of shape (?, num_boxes, 4).
            y_pred: Predicted bounding boxes,
                tensor of shape (?, num_boxes, 4).
        # Returns
            l1_loss: L1-smooth loss, tensor of shape (?, num_boxes).
        # References
            https://arxiv.org/abs/1504.08083
        """
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(y_true, y_pred):
        """Compute softmax loss.
        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, num_classes).
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, num_classes).
        # Returns
            softmax_loss: Softmax loss, tensor of shape (?, num_boxes).
        """
        y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)
        softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                      axis=-1)
        return softmax_loss

    def compute_loss(y_true, y_pred):
        """Compute mutlibox loss.
        # Arguments
            y_true: Ground truth targets,
                tensor of shape (?, num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                y_true[:, :, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                y_true[:, :, -7:] are all 0.
            y_pred: Predicted logits,
                tensor of shape (?, num_boxes, 4 + num_classes + 8).
        # Returns
            loss: Loss for prediction, tensor of shape (?,).
        """
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])

        # loss for all priors
        conf_loss = _softmax_loss(y_true[:, :, 4:-8],
                                       y_pred[:, :, 4:-8])
        loc_loss = _l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # get positives loss
        num_pos = tf.reduce_sum(y_true[:, :, -8], axis=-1)
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -8],
                                     axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -8],
                                      axis=1)

        # get negatives loss, we penalize only confidence here
        num_neg = tf.minimum(neg_pos_ratio * num_pos,
                             num_boxes - num_pos)
        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0, values=[num_neg,
                                [(1 - has_min) * negatives_for_hard]])
        num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)
        confs_start = 4 + background_label_id + 1
        confs_end = confs_start + num_classes - 1
        max_confs = tf.reduce_max(y_pred[:, :, confs_start:confs_end],
                                  axis=2)
        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]),
                                 k=num_neg_batch)
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                        tf.reshape(indices, [-1]))
        # full_indices = tf.concat(2, [tf.expand_dims(batch_idx, 2),
        #                              tf.expand_dims(indices, 2)])
        # neg_conf_loss = tf.gather_nd(conf_loss, full_indices)
        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

        # loss is sum of positives and negatives
        total_loss = pos_conf_loss + neg_conf_loss
        total_loss /= (num_pos + tf.to_float(num_neg_batch))
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                            tf.ones_like(num_pos))
        total_loss += (alpha * pos_loc_loss) / num_pos
        return total_loss
    return compute_loss
    
"""YOLO f-score detection metric"""

def YOLOFscore(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3):

  # Def custom metric using numpy
  def _YOLOFscore(y_true, y_pred, name=None):
    with ops.name_scope( name, "YOLOFscore", [y_true,y_pred] ) as name:
      fscore = tf.py_func(YOLOFscore_np,
                          [y_true,y_pred,num_classes,np.array(priors),
                           max_truth_boxes,thresh,nms_thresh],
                          [tf.float32], name=name)
    return {'fscore':tf.reduce_mean(fscore[0])}

  return _YOLOFscore


def YOLOFscore_np(y_true, y_pred, num_classes, priors, max_truth_boxes, thresh, nms_thresh):
    batch_size = y_pred.shape[0]
    fscore = np.zeros(batch_size, dtype='f')
    num_priors = priors.shape[0]
    num_coords  = 4
    data_size = num_coords+num_classes+1

    y_pred = yolo_activate_regions(y_pred, num_priors, num_classes)
    boxes,probs = yolo_get_region_boxes(y_pred, priors, num_classes, thresh)
    boxes,probs = yolo_do_nms_sort(boxes, probs, num_classes, nms_thresh)


    # for each image in batch
    for i in range(batch_size):
        b = boxes[i,:,:]
        p = probs[i,:,:]
        num_boxes   = b.shape[0]
        num_classes = p.shape[1]
        # put GT boxes for this image in a list
        gt_boxes = []
        for h_i in range(y_true.shape[2]):
          for w_i in range(y_true.shape[3]):
            if y_true[i,0,h_i,w_i] >= 0:
              gt_boxes.append(y_true[i,:,h_i,w_i])
        num_gt = len(gt_boxes)
        ok    = 0.
        total = 0.
        # for each detected bounding box in this image, find the class with maximum prob
        for j in range(num_boxes):
            max_class = np.argmax(p[j,:])
            pb = p[j,max_class]
            if(pb > thresh):
                total += 1.
                bb = b[j,:]
                # count as TP if the box overlaps more than 50% with a GT object and the class is correct
                for idx,gt in enumerate(gt_boxes):
                    if gt[0] == max_class and yolo_box_iou(bb,gt[1:5]) > 0.5:
                        ok += 1.
                        gt_boxes = gt_boxes[0:idx]+gt_boxes[idx+1:]
                        break
        recall = ok / num_gt
        precision = 0.
        if total > 0.:
          precision = ok / total
        if (recall+precision) > 0:
          fscore[i] = 2 * ((recall*precision) / (recall+precision))

    return fscore
    
    
    
    
"""SSD f-score detection metric"""

def SSDFscore(input_shape=(3,640,640),num_classes=45,priors=[[0.25,0.25], [0.5,0.5], [1.0,1.0], [1.7,1.7], [2.5,2.5]],max_truth_boxes=30,thresh=0.6,nms_thresh=0.3):

  # Def custom metric using numpy
  def _SSDFscore(y_true, y_pred, name=None):
    with ops.name_scope( name, "SSDFscore", [y_true,y_pred] ) as name:
      fscore = tf.py_func(SSDFscore_np,
                          [y_true,y_pred,num_classes,np.array(priors),
                           max_truth_boxes,thresh,nms_thresh],
                          [tf.float32], name=name)
    return {'fscore':tf.reduce_mean(fscore[0])}

  return _SSDFscore


def SSDFscore_np(y_true, y_pred, num_classes, priors, max_truth_boxes, thresh, nms_thresh):
    batch_size = y_pred.shape[0]
    fscore = np.zeros(batch_size, dtype='f')
    num_priors = priors.shape[0]
    num_coords  = 4
    data_size = num_coords+num_classes+1

    y_pred = yolo_activate_regions(y_pred, num_priors, num_classes)
    boxes,probs = yolo_get_region_boxes(y_pred, priors, num_classes, thresh)
    boxes,probs = yolo_do_nms_sort(boxes, probs, num_classes, nms_thresh)


    # for each image in batch
    for i in range(batch_size):
        b = boxes[i,:,:]
        p = probs[i,:,:]
        num_boxes   = b.shape[0]
        num_classes = p.shape[1]
        # put GT boxes for this image in a list
        gt_boxes = []
        for h_i in range(y_true.shape[2]):
          for w_i in range(y_true.shape[3]):
            if y_true[i,0,h_i,w_i] >= 0:
              gt_boxes.append(y_true[i,:,h_i,w_i])
        num_gt = len(gt_boxes)
        ok    = 0.
        total = 0.
        # for each detected bounding box in this image, find the class with maximum prob
        for j in range(num_boxes):
            max_class = np.argmax(p[j,:])
            pb = p[j,max_class]
            if(pb > thresh):
                total += 1.
                bb = b[j,:]
                # count as TP if the box overlaps more than 50% with a GT object and the class is correct
                for idx,gt in enumerate(gt_boxes):
                    if gt[0] == max_class and yolo_box_iou(bb,gt[1:5]) > 0.5:
                        ok += 1.
                        gt_boxes = gt_boxes[0:idx]+gt_boxes[idx+1:]
                        break
        recall = ok / num_gt
        precision = 0.
        if total > 0.:
          precision = ok / total
        if (recall+precision) > 0:
          fscore[i] = 2 * ((recall*precision) / (recall+precision))

    return fscore
    
    """
    SSD Metrics function
    Code adapted from:  
"""
def SSDMetrics():
  def _SSDMetrics(y_true, y_pred):
    return {'avg_iou': tf.reduce_mean(np.asarray([2]))}

  return _SSDMetrics