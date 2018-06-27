# -*- coding: utf-8 -*-
import tensorflow as tf
#定义IP和端口
strip_hosts = "10.67.64.22:1681"
strworker_hosts = "10.67.64.22:1682,10.67.64.22:1683"
#定义角色名称
strjob_name = 'ps'
task_index = 0
#将字符串转成数组
ps_hosts = strip_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})
#创建server
server = tf.train.Server(
                        {'ps':ps_hosts,'worker':worker_hosts},
                        job_name = strjob_name,
                        task_index = task_index)
#角色使用join进行等待
if strjob_name == 'ps':
    print('wait')
    server.join()
with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d"%task_index,
        cluster=cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    #模型参数
    W = tf.Variable(tf.random_normal([1]),name="weight")
    b = tf.Variable(tf.zeros([1]),name='bias')
    
    global_step = tf.train.get_or_create_global_step() #获取迭代次数
    
    #前向结构体
    z = tf.multiply(X,W) + b
    tf.summary.histogram('z',z)   #将预测值以直方图显示
    #反向优化
    cost = tf.reduce_mean(tf.square(Y-z))
    tf.summary.scalar("loss_function",cost) #将损失以标量显示
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step = global_step) #梯度下降
    
    saver = tf.train.Saver(max_to_keep = 1)
    merged_summary_op = tf.summary.merge_all() #合并所有summary
    
    init = tf.global_variables_initializer()
#定义参数
training_epochs = 2200
display_step = 2

sv = tf.train.Supervisor(is_chief=(task_index == 0 ),#0号worker 为 chief
                         logdir="log/super/",
                         init_op = init,
                         summary_op = None,
                         saver = saver,
                         global_step = global_step,
                         save_model_sece = 5)

#连接目标角色创建session
with sv.managed_session(server.target) as sess:
    print("sess ok")
    print(global_step.eval(session=sess))
    
    for epoch in range(global_step.eval(session = sess),training_epochs*len(train_X)):
        
        for(x,y) in zip(train_X, train_Y):
            _,epoch = sess.run([optimizer,global_step],feed_dict={X:x,Y:y})
            #生成summary
            summary_str = sess.run(merged_summary_op,feed_dict={X:x,Y:y})
            #将summary 写入文件
            sv.summary_computed(sess,summary_str,global_step=epoch)
            if( epoch % display_step == 0):
                loss = sess.run(cost,feed_dict={X;train_X,Y:train_Y})
                print("Epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
                if not (loss == 'NA'):
                    plotdata['batchsize'].append(epoch)
                    plotdata['loss'].append(loss)
                    
    print('Finished!')
    sv.saver.save(sess,"log/mnist_with_summaries/"+"sv.cpk",global_step = epoch)
    
sv.stop()            
            
    