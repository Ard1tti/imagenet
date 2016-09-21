import tensorflow as tf
from tensorflow.python.framework import errors
import xml.etree.ElementTree as ET
import numpy as np
import time
import threading

def label_to_vec(label, LABEL_NUM):
    v = np.zeros([LABEL_NUM])
    v[label]=1.0
    return v

class Input(object):
    """ DataSet object """
    
    def __init__(self, IMG_DIR, XML_DIR, f_list, capacity, shuffle=True):
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
            
    def _dict(self):
        nwid = [ET.parse(self.XML_DIR+f+".xml").getroot().find("object").find("name").text
               for f in self.f_list] 
        nwid_set = set(nwid)
        nwid_dict={wid:i for i,wid in zip(xrange(len(nwid_set)),list(nwid_set))}
        return nwid_dict
    
    def _run(self, sess, coord=None):
        while True:
            if coord and coord.should_stop():
                break
            file_name = sess.run(self.f_queue.dequeue())
            
            with open(self.IMG_DIR+file_name+".JPEG") as f:
                key = f.read()
            
            f=ET.parse(self.XML_DIR+file_name+".xml")
            nwid = f.getroot().find("object").find("name").text
            label = label_to_vec(self._dict[nwid], self.CLASS_NUM)
            
            sess.run(self._enqueue_ops[0], feed_dict={self.key:key, self.label:label})
                 
    def create_threads(self, sess, coord=None, daemon=False, start=False):
        threads = [threading.Thread(target=self._run, args=(sess, coord))]
        if coord:
            threads.append(threading.Thread(target=self._close_on_stop,
                                          args=(sess, self._cancel_op, coord)))
        for t in threads:
            if daemon: t.daemon = True
            if start: t.start()
        return threads
    
    def dequeue(self):
        image, label = self._queue.dequeue()
        return image, label
    
    
        