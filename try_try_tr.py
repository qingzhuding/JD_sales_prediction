# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
monthes = ['2016-08', '2016-09', '2016-10', '2016-11',
           '2016-12', '2017-01', '2017-02', '2017-03', '2017-04']

monthes_sale = ['2016-06','2016-07','2016-08', '2016-09', '2016-10', '2016-11',
           '2016-12', '2017-01']

def convertTime(x):
    # print(x)
    return x[:7]

if __name__ == '__main__':
    # print(shop_month.keys())
    #print (2978438.0/11398239*3089)
    with open("1630.csv", "r") as myfile:
    	reader = csv.DictReader(myfile)
    	for row in reader:

    		print (row['pid'])
    #product = pd.read_csv(r't_order.csv',nrows=50000)
    '''
    print (product[0:10])
    print (len(sorted(set(product['brand']))))
    #print (product['brand'])
    a=product['brand'].value_counts()
    del product
    print (a)
    #a=np.array([1,2,3])
    a[0:100].plot(kind = "bar")
    plt.show()
    '''
    #print (len(sorted(set(product['cate']))))
    #print (len(sorted(set(product['shop_id']))))
    #print (len(sorted(set(product['pid']))))
    #ads = pd.read_csv(r't_ads.csv')
    #comments = pd.read_csv( r't_comment.csv')
    #sales = pd.read_csv(r't_sales_sum.csv')

    '''
    with open(r'shop_month_order.txt', 'rb') as f:
        a = pickle.load(f)
    print (a[2])

    print (product[0:20])

    kinds=sorted(set(product['cate']))
    print (kinds)
    print (len(kinds))
    
    print ('-----------------ads---------------------')
    shops = sorted(set(ads['shop_id']))
    print (len(shops),shops[1:100])
    print ('-----------------product---------------------')
    shops2=sorted(set(product['shop_id']))
    print (len(shops2),shops2[1:100])
    
    print ('-----------------order---------------------')
    shops3=sorted(set(order['shop_id']))
    print (len(shops3),shops3[1:100])

    print ('-----------------sales---------------------')
    shops4=sorted(set(sales['shop_id']))
    print (len(shops4),shops4[1:100])

    print ('-----------------comments---------------------')
    shops4=sorted(set(comments['shop_id']))
    print (len(shops4),shops4[1:100])
    '''