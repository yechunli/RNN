import tensorflow as tf
import numpy as np
import jieba
import collections

flag = tf.flags.FLAGS
tf.flags.DEFINE_string('model', 'small', '......')
tf.flags.DEFINE_string('data_path', 'F:\python_project\\tfword2vec_with_nec\\280.txt', '.......')
tf.flags.DEFINE_bool('use_float', False, '......')


class Smallconfig():
    batch_size = 10
    iterator = 10000

class Largeconfig():
    batch_size = 20
    iterator = 20000

def get_config():
    if flag.model == 'small':
        return Smallconfig
    if flag.model == 'large':
        return Largeconfig

def get_type():
    if flag.use_float:
        return tf.float16
    else:
        return tf.float32

def remove_signal(line):
    if ' ' in line:
        line = line.replace(' ', '')
    if '\n' in line:
        line = line.replace('\n', '')
    if '，' in line:
        line = line.replace('，', '')
    if '“' in line:
        line = line.replace('“', '')
    if '”' in line:
        line = line.replace('”', '')
    if '。' in line:
        line = line.replace('。', '')
    if '-' in line:
        line = line.replace('-', '')
    if '\\' in line:
        line = line.replace('\\', '')
    if '/' in line:
        line = line.replace('/', '')
    if '.' in line:
        line = line.replace('.', '')
    return line
def data_process(batch_size, train_dir):
    with open('F:\python_project\\tfword2vec_with_nec\stop_words.txt', encoding='utf-8') as stop_word_file:
        stop_word = []
        line = stop_word_file.readline()
        while line:
            stop_word.append(line)
            line = stop_word_file.readline()
        stop_word = set(stop_word)
        print('loading stop_word sucess')
    file = open(train_dir)
    line = file.readline()
    word_list = []
    train_data = []
    label_data = []
    while line:
        if line != '\n':
            line = remove_signal(line)
            line_list = list(jieba.cut(line))
            sentence_list = []
            for word in line_list:
                if word not in stop_word and word not in ['qingkan520','www','com','http']:
                    word_list.append(word)
                    sentence_list.append(word)
            if len(sentence_list) > batch_size + 1:
                i = 0
                while i+batch_size < len(sentence_list)-1:
                    train_batch = sentence_list[i:i+batch_size:]
                    label_batch = sentence_list[i+1:i+batch_size+1]
                    train_data.append(train_batch)
                    label_data.append(label_batch)
                    i = i + batch_size
                # end = len(sentence_list)
                # train_batch = sentence_list[:end-1:]
                # label_batch = sentence_list[1::]
                # train_data.append(train_batch)
                # label_data.append(label_batch)
        line = file.readline()
    word = collections.Counter(word_list)
    common_word = word.most_common(58799)
    dict_list = [x[0] for x in common_word]
    dict = {}
    for i in range(len(dict_list)):
        dict[dict_list[i]] = i
    train_batch_final = []
    label_batch_final = []
    for i in range(len(train_data)):
        train_line = []
        label_line = []
        for j in range(len(train_data[i])):
            train = dict.get(train_data[i][j])
            label = dict.get(label_data[i][j])
            if train == None or label == None:
                train = -1
                label = -1
                print('error')
            train_line.append(train)
            label_line.append(label)
        train_batch_final.append(train_line)
        label_batch_final.append(label_line)
    return train_batch_final, label_batch_final, dict

class RNN_model():
    def __init__(self, rnn_layers, time_step, embedding_size, learning_rate, keep, dict):
        self.rnn_layers = rnn_layers
        self.input_data = tf.placeholder(tf.int32, [1, time_step])
        self.time_step = time_step
        self.embedding_size = embedding_size
        self.label = tf.placeholder(tf.float32, [1, time_step])

        self.learning_rate = learning_rate
        self.keep = keep
        self.dict = dict

        self.model()
    def model(self):
        length = len(self.dict)
        #embedding_dict = tf.Variable(tf.truncated_normal([length, self.embedding_size], stddev=0.1, dtype=get_type()))
        embedding_dict = tf.get_variable('dict', [58799, self.embedding_size], dtype=get_type())
        input_data = tf.nn.embedding_lookup(embedding_dict, self.input_data)
        input_data = tf.reshape(input_data, [1, self.time_step, self.embedding_size])

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.embedding_size, forget_bias=0, state_is_tuple=True)#forget_bias是遗忘门的系数，0表示都忘了
        if self.keep < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.keep)
        cells = tf.nn.rnn_cell.MultiRNNCell(self.rnn_layers * [cell], state_is_tuple=True)
        self.init_state = cells.zero_state(1, tf.float32)
        state = self.init_state

        #with tf.variable_scope('RNN'):
            # for time_step in range(self.input_data_size):
        output, final_state = tf.nn.dynamic_rnn(cells, input_data, initial_state=state)
            #for time_step in range(self.input_data_size):
            # if time_step > 0 :
            #     tf.get_variable_scope().reuse_variables()
            # tmp_output, state = cells.call(tf.reshape(input_data1[0][time_step],[1, self.embedding_size]), state)#按照顺序一个个往里送,这个用法np.array数组可以，列表不行，看bug
            # output.append(tmp_output)
            # time_step = time_step + 1
        output = tf.reshape(output, [self.time_step, self.embedding_size])#输入是batch*embedding，输出也是
        weight = tf.Variable(tf.truncated_normal([self.embedding_size, 1]), dtype=get_type())
        bias = tf.Variable(tf.zeros(shape=[1, 1]), dtype=get_type())
        self.act = act = tf.add(tf.matmul(output, weight), bias)
        #self.result = result = tf.nn.softmax(act)
        loss = (tf.transpose(self.label) - act) ** 2
        # self.loss1 = loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(self.label), logits=act)
        #loss = (tf.transpose(self.label) - result)**2
        with tf.name_scope('Loss'):
            self.loss = final_loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', self.loss)
        tvars = tf.trainable_variables()
        gradiant, _ = tf.clip_by_global_norm(tf.gradients(final_loss, tvars), 5)
        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)#.minimize(self.loss)
        #self.train_op = optimizer.apply_gradients(zip(gradiant,tvars))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

if __name__ == '__main__':
    if flag.data_path:
        print(flag.data_path)
    config = get_config()

    train_dir = 'F:\python_project\\tfword2vec_with_nec\\280.txt'
    train, label, dict = data_process(config.batch_size, train_dir)
    print("data processing finished!")
    model = RNN_model(rnn_layers=2, time_step=config.batch_size, embedding_size=200, learning_rate=0.01, keep=1, dict=dict)
    merge = tf.summary.merge_all()

    feed = [model.act, model.train_op, model.loss, merge]

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('F:\python_project\log\RNN', graph=sess.graph)
        init = tf.initialize_all_variables()
        sess.run([init, model.init_state])
        for j in range(config.iterator):
            for i in range(len(train)):
                train_i = np.reshape(train[i], [1, config.batch_size])
                label_i = np.reshape(label[i], [1, config.batch_size])
                feed_dict = {model.input_data: train_i, model.label: label_i}
                act, _, loss, summary = sess.run(feed, feed_dict=feed_dict)
                #print(np.transpose(act), label_i)
                if i%100 == 10:
                    print(loss)
                    writer.add_summary(summary)
                    #break
            #break
