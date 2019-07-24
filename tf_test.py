import tensorflow as tf

a = tf.constant([[1,2,3],[3,4,5]]) # shape (2,3)
b = tf.constant([[7,8,9],[10,11,12]]) # shape (2,3)
c = tf.constant([[6,8,9],[10,11,12]])
ab = tf.stack([a,b], axis=1)
with tf.Session() as sess:
    result1 = sess.run(ab)
    print(result1)
print(ab)