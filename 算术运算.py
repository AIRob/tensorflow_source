# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:10:35 2018

@author: lWX379138
"""
import tensorflow as tf
import numpy  as np

x = tf.constant(2)
y = tf.constant(5)

def f1():
    return tf.multiply(x,17)
    pass
def f2():
    return tf.add(y,23)
    pass

def main():

    
    with tf.Session() as sess:
        #tf.assign(x,y)
    
            
    
        print(sess.run(x))
        print(sess.run(y))        
        
        print(sess.run(tf.add(x,y)))
        print(sess.run(tf.subtract(x,y)))   #减法
        print(sess.run(tf.multiply(x,y)))   #乘法
        print(sess.run(tf.divide(x,y)))     #除法
        print(sess.run(tf.mod(x,y)))        #取模,x/y的余数
        print(sess.run(tf.abs(tf.constant(-4)))) #求绝对值
        print(sess.run(tf.negative(y)))     #取负
        print(sess.run(tf.sign(3)))         #返回输入的符号，负数则为-1,0为0,正数为1
        print(sess.run(tf.sign(0)))
        print(sess.run(tf.sign(-3)))
        #print(sess.run(tf.inv(y)))          #取反
        print(sess.run(tf.square(x)))       #平方
        print(sess.run(tf.round([0.9,2.5,2.3,1.5,-4.5]))) #舍入最接近的整数 [1.0,2.0,2.0,2.0,-4.0]        
        print(sess.run(tf.sqrt(0.04)))         #平方根,float
        a = [[2,2],[3,3]]
        b = [[8,16],[2,3]]
        print(sess.run(tf.pow(a,b)))        #幂次方运算[[2的8次方，2的16次方],[3的2次方，3的3次方]]
        print(sess.run(tf.exp(2.0)))  #e的次方 float
        print(sess.run(tf.log(10.0)))  # float 计算log，一个输入计算e的ln,两次输入以第二次输入为底
        #print(sess.run(tf.log(3.0,9.0))) 
        print(sess.run(tf.maximum(x,y)))    #最大值
        print(sess.run(tf.minimum(x,y)))    #最小值 
        print(sess.run(tf.cos(0.0)))         #三角函数
        print(sess.run(tf.sin(0.0))) 
        print(sess.run(tf.tan(45.0)))
        print(sess.run(tf.atan(1.0)))     #ctan三角函数 
        print(sess.run(tf.cond(tf.less(x,y),f1,f2)))    #满足条件则执行f1，否则执行f2
        #print(sess.run(tf())) 
    
    
    pass
pass

if __name__ =='__main__':
    
    
    main()
    
    
    print('end')
    pass
pass

