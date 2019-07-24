import tensorflow as tf
import numpy as np
import os
from generate_word_data import generator,BiLSTM_Model


learning_rate = 0.01
training_iters = 100000
embeddings_size=200
# batch_size=128
batch_size=128
display_step=10
save_ckpt_name = 'bilstm_crf_cn.ckpt'

data_path="./data/total.txt"
gen=generator(data_path)
vocab_size=len(gen.word_dict)
sess = tf.Session()
#词嵌入矩阵
# embeddings=tf.Variable(tf.truncated_normal([vocab_size,embeddings_size],stddev=0.05))
# embeddings.initializer.run(session=sess)
# rt=sess.run(embeddings)
lstm_model=BiLSTM_Model(vocab_size=vocab_size,embeddings_size=embeddings_size)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(lstm_model.loss)
# 分类准确率
def viterbi_decode_iter(score_, trans_):
    viterbi_sequence, _ = tf.contrib.crf.crf_decode(score_, trans_)
    return viterbi_sequence

def viterbi_decode(score_, trans_):
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(score_, trans_)
    return viterbi_sequence
def evaluate_iter(scores_, lengths_, trans_, targets_):
    correct_seq = 0.
    total_len = 0.
    for ix, score_ in enumerate(scores_):
        score_real = score_[:lengths_[ix]]
        target_real = targets_[ix][:lengths_[ix]]
        pre_sequence = viterbi_decode(score_real, trans_)
        correct_seq += np.sum((np.equal(pre_sequence, target_real)))
        total_len += lengths_[ix]
    return correct_seq/total_len

def evaluate(scores_, lengths_, trans_, targets_):
    correct_seq = 0.
    total_len = 0.
    for ix, score_ in enumerate(scores_):
        score_real = score_[:lengths_[ix]]
        target_real = targets_[ix][:lengths_[ix]]
        pre_sequence = viterbi_decode(score_real, trans_)
        correct_seq += np.sum((np.equal(pre_sequence, target_real)))
        total_len += lengths_[ix]
    return correct_seq/total_len

# accuracy=evaluate(lstm_model.logits,lstm_model.seqlen,lstm_model.trans,lstm_model.y_target)
# correct_pred = tf.equal(tf.argmax(lstm_model.logits,1), tf.cast(lstm_model.y_target,tf.int64))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
max_accuracy=0
step = 1
while step * batch_size < training_iters:
    batch_x, batch_y, batch_seqlen = gen.next(batch_size)
    train_feed={lstm_model.x_input: batch_x, lstm_model.y_target: batch_y,
                                   lstm_model.seqlen: batch_seqlen}
    # 每run一次就会更新一次参数
    sess.run(optimizer, feed_dict=train_feed)
    # emd=sess.run(lstm_model.emd,feed_dict=train_feed)
    if step % display_step == 0:
        # 在这个batch内计算准确度
        logits,trans=sess.run([lstm_model.logits, lstm_model.trans],train_feed)
        acc=evaluate(logits,batch_seqlen,trans,batch_y)
        # acc = sess.run(accuracy, feed_dict=train_feed)
        # 在这个batch内计算损失
        loss = sess.run(lstm_model.loss, feed_dict=train_feed)
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

        if acc >= max_accuracy :
            max_accuracy = acc
            saver.save(sess, os.path.join("./temp", save_ckpt_name))
            print('Generation # {}. --model saved--'.format(step))
    step += 1
print("Optimization Finished!")
import time

# 最终，我们在测试集上计算一次准确度
test_data = gen.data
test_label = gen.labels
test_seqlen = gen.seqlen
logits,trans=sess.run([lstm_model.logits, lstm_model.trans],feed_dict={lstm_model.x_input: test_data, lstm_model.y_target: test_label,
                                  lstm_model.seqlen: test_seqlen})
print("Testing Accuracy:", \
     evaluate(logits,test_seqlen,trans,test_label))

