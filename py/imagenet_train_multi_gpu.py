from imagenet_input import Input
import ResNet_imagenet as model
import tensorflow as tf
import os

BATCH_SIZE=16
EVAL_SIZE=1000
NUM_GPUS=8
CKPT_DIR="../../ckpt/"+model.__name__+"/"

def tower_loss(batch, scope, keep_prob, reuse):
    images, labels = batch
    
    logits = model.inference(images, keep_prob, True, reuse)
    _ = model.loss(logits, labels)
    
    losses = tf.get_collection('losses', scope)
    
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

def average_grads(list_grads):
    grads = []
    for i in range(len(list_grads[0])):
        if list_grads[0][i][0] is None:
            grads.append((None, list_grads[0][i][1]))
        else:
            grads.append((tf.reduce_mean([list_grads[j][i][0] for j in range(len(list_grads))],[0]),
                         list_grads[0][i][1]))
    return grads
    

def train():
    # Train in multi GPU
    print('Training '+model.__name__+' model')
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        train_data, test_data = cifar10_input()
        
        train_batch = train_data.dist_batch(BATCH_SIZE, 20000)
        images, labels = test_data.dist_batch(EVAL_SIZE, 4000)
        
        keep_prob = tf.placeholder(tf.float32)
        
        logit = model.inference(images, keep_prob, False, False)
        accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logit, labels, 1),tf.float32))
        tf.get_variable_scope().reuse_variables()
        
        tower_grads = []
        tower_losses = []
        
        lr = tf.placeholder(tf.float32)
        opt = tf.train.GradientDescentOptimizer(lr)
        
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    loss = tower_loss(train_batch, scope, keep_prob, True)
                    grads = opt.compute_gradients(loss)
                    tower_losses.append(loss)
                    tower_grads.append(grads)
                    
        grads = average_grads(tower_grads)
        mean_loss = tf.reduce_mean(tower_losses)
       
        train_op = opt.apply_gradients(grads)
    
        sess = tf.Session()
        
        rate = 0.1
        mean = 0.
        highest = 0.
        
        saver = tf.train.Saver(tf.all_variables())
        if not os.path.exists(CKPT_DIR):
            os.makedirs(CKPT_DIR)
        
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Variables are restored from "+ CKPT_DIR)
        else:
            sess.run(tf.initialize_all_variables())
            print("Variables are initialized")
        
        tf.train.start_queue_runners(sess=sess)
        for i in range(40000):
            _, cross_entropy = sess.run([train_op, mean_loss], feed_dict = {keep_prob: 0.5,
                                                                       lr:rate})
            mean = mean + cross_entropy/200
            if (i+1)%200 == 0 and i>0:
                print("step %d, cross_entropy %g"%(i, mean))
                mean=0.
        
            if (i+1)%200 == 0 and i>0:
                eval_accuracy = sess.run(accuracy)
                if eval_accuracy > highest:
                    highest = eval_accuracy
                print("test accuracy %g"%(eval_accuracy))
                
            if (i+1)%2000 == 0 and i>0:
                checkpoint_path = os.path.join(CKPT_DIR, 'model.ckpt')
                saver.save(sess, checkpoint_path)
                print("Model saved in "+CKPT_DIR)
                
        print("highest accuracy: %g"%(highest))
                
def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
