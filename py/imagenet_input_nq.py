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
    
    def __init__(self, files, labels, class_num, capacity=100, threads=16,
                 IMG_DIR=None, size=(224,224)):
        self.IMG_DIR = IMG_DIR
        self.files = files
        self.labels = labels

        self.IMG_N = len(labels)
        self.CLASS_NUM = class_num
        self.size=size
        
        self.capacity=capacity
        self._img_queue=[]
        self._lab_queue=[]
        self._threads=threads
        self._lock=threading.Lock()
        self._runs=0
    
    def train_batch(self, BATCH_NUM, coord):
        while True:
            assert BATCH_NUM > capacity
            if coord and coord.should_stop():
                break
            if len(self._lab_queue) > BATCH_NUM:
                images = self._img_queue[0:BATCH_NUM],
                labels = self._lab_queue[0:BATCH_NUM]
                images = [augment(image) for image in images]
                
                self._img_queue=self._img_queue[BATCH_NUM:]
                self._lab_queue=self._lab_queue[BATCH_NUM:]
                return np.asarray(images,dtype=np.float32), np.asarray(labels,dtype=np.int32)
            
    def _run(self, coord=None):
        if coord:
            coord.register_thread(threading.current_thread())
        try:
            while True:
                if coord and coord.should_stop():
                    break
                
                if len(self._lab_queue)>self.capacity:
                    continue
                
                i = np.random.randint(0, self.IMG_N)
                file_name=self.files[i]
                label=self.labels[i]
            
                if self.IMG_DIR is None:
                    f = None
                    try:
                        f=urllib.urlopen(file_name)
                        img=Image.open(StringIO(f.read()))
                        image=PIL2array(img)
                        if image.shape[-1] is 3:
                            self._img_queue.append(image)
                            self._lab_queue.append(label)
                        
                    except Exception as e:
                        continue
                else:
                    with open(self.IMG_DIR+file_name) as f:
                        img=Image.open(f)
                        image=PIL2array(img)
                        if image.shape[-1] is 3:
                            self._img_queue.append(image) 
                            self._lab_queue.append(label)
        except Exception as e:
            if coord:
                logging.error(str(e))
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                raise
                
    def _close_on_stop(self, coord):
        coord.register_thread(threading.current_thread())
        coord.wait_for_stop()
                
    def create_threads(self, coord=None, daemon=False, start=False):
        threads = [threading.Thread(target=self._run, args=(coord,)) 
                   for _ in xrange(self._threads)]
        #if coord: threads.append(threading.Thread(target=self._close_on_stop, args=(coord,)))

        for t in threads:
            if daemon: t.daemon = True
            if start: t.start()
        return threads
    
    def augment(img, short=(256,481), crop=(224,224)):
        np.random.seed(int(time.time()))
        width, height=img.size
        target_short=np.random.random_integers(short[0],short[1])
        rate=target_short/float(min([width, height]))
        
        target_height=int(height*rate)
        target_width=int(width*rate)
        
        cut_height=np.random.random_integers(0,target_height-crop[0])
        cut_width=np.random.random_integers(0,target_width-crop[1])
        
        bool_flip=np.random.choice([True,False])
        
        img=img.resize((target_width, target_height), resample=Image.BILINEAR)
        img=img.crop((cut_width,cut_height,cut_width+crop[0],cut_height+crop[1]))
        if bool_flip: img=img.transpose(Image.FLIP_LEFT_RIGHT)
            
        image = PIL2array(img)
        image = image-np.mean(image,axis=(0,1))    
        return image
    
def label2vec(label, CLASS_NUM):
    v = np.zeros([CLASS_NUM])
    v[label]=1.0
    return v

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape([img.size[0],img.size[1],-1])

       