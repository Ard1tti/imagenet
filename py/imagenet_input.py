import tensorflow as tf
from tensorflow.python.training import queue_runner
import xml.etree.ElementTree as ET
import PIL

class DataSet(object):
    """ DataSet object """
    
    def __init__(self, IMAGE_DIR, XML_DIR, file_name, batch_size):
        self.IMAGE_DIR = IMAGE_DIR
        self.XML_DIR = XML_DIR
        self.file_queue = tf.train.string_input_producer(file_name)
        self.data_queue = tf.FIFOQueue(batch_size*3, [tf.float32, tf.int32])
        self.wnid_dict = self.create_dict()
        
    def create_dict(self):
        nwid = []
        for f in self.file_queue:
            doc = ET.parse(XML_DIR+f+".xml")
            root=doc.getroot()
            nwid.append(root.find("object").find("name").text)
        
        ret_dict={}
        for i, wid in zip(xrange(len(set(nwid))),list(set(nwid))):
            ret_dict.update({wid:i})
            
        return ret_dict
    
    def _enqueue(sess, coord):
        file_name = sess.run(self.file_queue.dequeue())
        
            
