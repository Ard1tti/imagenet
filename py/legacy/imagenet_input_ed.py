import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
import urllib
import Image
from StringIO import StringIO
import numpy as np
import threading
import time

class Input(object):
    """ DataSet object """
    
    def __init__(self, files, labels, capacity, class_num,
                 IMG_DIR=None, padding=20000, threads=4, size=(224,224)):
        self.IMG_DIR = IMG_DIR
        self.files = files
        self.labels = labels
        self.IMG_N = len(labels)
        self.CLASS_NUM = class_num
        self.size=size
        
        #self.f_queue = tf.train.range_input_producer(len(labels), capacity=capacity*3)
        #self.f_dequeue=self.f_queue.dequeue()
        
        self.images = [tf.placeholder(tf.uint8)]*threads
        self.labels = [tf.placeholder(tf.int32)]*threads
                
        self._queue = tf.FIFOQueue(capacity, [tf.uint8, tf.int32],
                                   shapes=[[size[0],size[1],3],[]])
        self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
        self._close_op = self._queue.close()
        self._queue_closed_exception_types=(errors.OutOfRangeError,)
        self._lock = threading.Lock()
        self._exceptions_raised = []
        self._threads = threads
        
        self._enqueue_ops=[(image,label,self._queue.enqueue([image,label]))
                           for image,label in zip(self.images,self.labels)]
        
        tf.train.add_queue_runner(self)
        np.random.seed(int(time.time()))
        
    def _close_on_stop(self, sess, cancel_op, coord):
        coord._registered_threads.add(threading.current_thread())
        coord.wait_for_stop()
        try:
            sess.run(cancel_op)
        except Exception as e:
            logging.vlog(1, "Ignored excpetion: %s", str(e))
    
    def _run(self, sess, ops, coord=None):
        
        if coord:
            coord._registered_threads.add(threading.current_thread())
        
        try:
            while True:
                if coord and coord.should_stop():
                    break
                
                try:
                    i = np.random.randint(0, IMG_N)
                    file_name=self.files[i]
                    label=self.labels[i]
            
                    if self.IMG_DIR is None:
                        f = None
                        try:
                            f=urllib.request.urlopen(file_name, timeout=5)
                        except Exception as e:
                            continue
                        if f and f.info().type in ['image/jpg','image/jpeg','image/pjpeg',
                                             'image/gif','image/png']:
                            img=Image.open(StringIO(f.read()))
                            img=random_resize_and_crop(img, crop=self.size)
                            image=PIL2array(img)
                            if image.shape[-1] is 3:
                                sess.run(ops[2], {ops[0]:image,ops[1]:label})
                    else:
                        with open(self.IMG_DIR+file_name) as f:
                            img=Image.open(f)
                            img=random_resize_and_crop(img, crop=self.size)
                            image=PIL2array(img)
                            sess.run(ops[2], {ops[0]:image,ops[1]:label})
                except self._queue_closed_exception_types:
                    pass
                    #with self._lock:
                        #self._threads -= 1
                        #if self._threads ==0:
                            #try:
                                #sess.run(self._close_op)
                            #except Exception as e:
                                #logging.vlog(1, "Ignored exception: %s", str(e))         
        except Exception as e:
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
                 
    def create_threads(self, sess, coord=None, daemon=False, start=False):
        threads = [threading.Thread(target=self._run, args=(sess, ops, coord))
                  for ops in self._enqueue_ops]
        if coord:
            threads.append(threading.Thread(target=self._close_on_stop,
                                          args=(sess, self._cancel_op, coord)))
        for t in threads:
            if daemon: t.daemon = True
            if start: t.start()
            print(t)
        return threads
    
    def dequeue(self):
        image, label = self._queue.dequeue()
        return image, label
    
def label2vec(label, CLASS_NUM):
    v = np.zeros([CLASS_NUM])
    v[label]=1.0
    return v

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape([img.size[0],img.size[1],-1])

def random_resize_and_crop(img, short=(256,481), crop=(224,224)):
    np.random.seed(int(time.time()))
    width, height=img.size
    target_short=np.random.random_integers(short[0],short[1])
    rate=target_short/float(min([width, height]))
    
    target_height=int(height*rate)
    target_width=int(width*rate)
    
    cut_height=np.random.random_integers(0,target_height-crop[0])
    cut_width=np.random.random_integers(0,target_width-crop[1])
    
    img=img.resize((target_width, target_height), resample=Image.BILINEAR)
    img=img.crop((cut_width,cut_height,cut_width+crop[0],cut_height+crop[1]))
    return img
                 
    
    
        