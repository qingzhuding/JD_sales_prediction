# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import pickle
monthes = ['2016-08', '2016-09', '2016-10', '2016-11',
           '2016-12', '2017-01', '2017-02', '2017-03', '2017-04']

monthes_sale = ['2016-06','2016-07','2016-08', '2016-09', '2016-10', '2016-11',
           '2016-12', '2017-01']

def convertTime(x):
    # print(x)
    return x[:7]

if __name__ == '__main__':
	order = pd.read_csv(r't_order.csv')
	product = pd.read_csv(r't_product.csv')
	ads = pd.read_csv(r't_ads.csv')
	comments = pd.read_csv( r't_comment.csv')
	sales = pd.read_csv(r't_sales_sum.csv')


	
	print '-----------------ads---------------------'
	shops = sorted(set(ads['shop_id']))
	print len(shops),shops[1:100]
	print '-----------------product---------------------'
	shops2=sorted(set(product['shop_id']))
	print len(shops2),shops2[1:100]
	
	print '-----------------order---------------------'
	shops3=sorted(set(order['shop_id']))
	print len(shops3),shops3[1:100]

	print '-----------------sales---------------------'
	shops4=sorted(set(sales['shop_id']))
	print len(shops4),shops4[1:100]

	print '-----------------comments---------------------'
	shops4=sorted(set(comments['shop_id']))
	print len(shops4),shops4[1:100]