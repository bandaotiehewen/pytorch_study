# -*- coding: utf-8 -*- 
"""
__version__ = v1.0
__file__ = test_tf.py
__title__ = ''
__author__ = 'lb'
__mtime__ = 2018/4/28 10:41 
__des__=''
"""
import tensorflow as tf
hello = tf.constant('hello, tensorflow')
sess = tf.Session()
print(sess.run(hello))