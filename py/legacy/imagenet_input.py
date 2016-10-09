import tensorflow as tf
from tensorflow.python.framework import errors
import urllib
import Image
from StringIO import StringIO
import numpy as np
import threading

class Input(object):
    """ DataSet object """
    
    def __init__(self, files, labels, capacity, class_num,
                 IMG_DIR=None, padding=20000, threads=4, shuffle=True):
        self.IMG_DIR = IMG_DIR
        self.files = files
        self.labels = labels        
        self.CLASS_NUM = class_num
        
        self.f_queue = tf.RandomShuffleQueue(capacity=capacity*3+padding,
                                             min_after_dequeue=padding,
                                             dtypes=[files.dtype,labels.dtype],
                                             shapes=[files.shape[1:],labels.shape[1:]])
        self.f_enqueue_op=self.f_queue.enqueue_many([self.files,self.labels])
        
        self.key = tf.placeholder(tf.string)
        self.image = tf.placeholder(tf.uint8)
        self.jpeg = tf.image.decode_jpeg(self.key)
        self.png = tf.image.decode_png(self.key)
        self.label = tf.placeholder(tf.float32)
                
        self._queue = tf.FIFOQueue(capacity, [tf.uint8, tf.float32])
        self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
        self._close_op = self._queue.close()
        self._queue_closed_exception_types=(errors.OutOfRangeError,)
        
        self.jpeg_enqueue=self._queue.enqueue([self.jpeg,self.label])
        self.png_enqueue=self._queue.enqueue([self.png,self.label])
        self._enqueue_op=self._queue.enqueue([self.image, self.label])
         
        tf.train.add_queue_runner(self)
        tf.train.add_queue_runner(tf.train.QueueRunner(self.f_queue,[self.f_enqueue_op]))
       
    def __init__deprecated(self, IMG_DIR, XML_DIR, f_list, capacity, shuffle=True):
        self.IMG_DIR = IMG_DIR
        self.XML_DIR = XML_DIR
        self.f_list=f_list
        self.f_queue = tf.train.string_input_producer(f_list, shuffle=shuffle,
                                                      seed=int(time.time()),
                                                      capacity = capacity)
        
        self.key = tf.placeholder(tf.string)
        self.image = tf.image.decode_jpeg(self.key)
        self.label = tf.placeholder(tf.float32)
        
        self._queue = tf.FIFOQueue(capacity, [tf.uint8, tf.float32])
        self._enqueue_ops = [self._queue.enqueue([self.image, self.label])]
        self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
        self._close_op = self._queue.close()
        self._queue_closed_exception_types=(errors.OutOfRangeError,)
        
        self._dict = self._dict()
        self.CLASS_NUM = len(self._dict)
        
        tf.train.add_queue_runner(self)
        
    def _close_on_stop(self, sess, cancel_op, coord):
        coord.register_thread(threading.current_thread())
        coord.wiat_for_stop()
        try:
            sess.run(cancel_op)
        except Exception as e:
            logging.vlog(1, "Ignored excpetion: %s", str(e))
    
    def _run(self, sess, coord=None):
        if coord:
            coord.register_thread(threading.lcurrent_thread())
            
        while True:
            if coord and coord.should_stop():
                break
            
            file_name, label = sess.run(self.f_queue.dequeue())
            
            if self.IMG_DIR is None:
                try:
                    f=urllib.urlopen(file_name)
                    if f.info().type in ['image/jpg','image/jpeg','image/pjpeg',
                                         'image/gif','image/png']:
                        img=Image.open(StringIO(f.read()))
                        image=PIL2array(img)
                        label_vec=label2vec(label, self.CLASS_NUM)
                        sess.run(self._enqueue_op, {self.image:image,self.label:label_vec})
                except:
                    pass
                              
            else:
                with open(self.IMG_DIR+file_name) as f:
                    img=Image.open(f)
                    image=PIL2array(img)
                    label_vec=label2vec(label, self.CLASS_NUM)
                    sess.run(self._enqueue_op, {self.image:image,self.label:label_vec})
    
    def _run_deprecated(self, sess, coord=None):
        if coord:
            coord.register_thread(threading.lcurrent_thread())
            
        while True:
            if coord and coord.should_stop():
                break
            
            file_name, label = sess.run(self.f_queue.dequeue())
            
            if self.IMG_DIR is None:
                try:
                    f=urllib.urlopen(file_name)
                    f_type=f.info().type
                    key=f.read()
                    print("downloaded")
                    if f_type in ['image/jpg','image/jpeg','image/pjpeg']:
                        label_vec = label2vec(label, self.CLASS_NUM)
                        sess.run(self.jpeg_enqueue,feed_dict={self.key:key,self.label:label_vec})
                    elif f_type in ['image/png']:
                        image=tf.image.decode_png(key)
                        label_vec = label2vec(label, self.CLASS_NUM)
                        sess.run(self.png_enqueue,feed_dict={self.key:key,self.label:label_vec})
                    
                except:
                     pass
                              
            else:
                with open(self.IMG_DIR+file_name) as f:
                    key = f.read()
                image=tf.image.decode_jpeg(key)
                label_vec = label2vec(label, self.CLASS_NUM)
                sess.run(self.jpeg_enqueue,feed_dict={self.key:key,self.label:label_vec})
                 
    def create_threads(self, sess, coord=None, daemon=False, start=False):
        threads = [threading.Thread(target=self._run, args=(sess, coord))]*self._threads
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
    
        