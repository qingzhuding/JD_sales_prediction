# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import pickle
import csv
import os.path


shoplist=range(1,3001)#所有商家的序号，1-3000

def process_line(line):
	#line=tuple(line.rstrip().split(","))
	addr=line['shop_id']
	print (line)
	#print (addr)
	#with open("%s.txt"%addr, "wb") as myfile:
	#	pickle.dump(line, myfile)

	filename='%s.csv'%addr
	file_exists = os.path.isfile(filename)
	
	with open('%s'%filename, 'a') as csvfile:#, newline=''
		fieldnames = ['rtn_amt', 'sale_amt', 'offer_amt', 'ord_cnt', 'pid', 'user_cnt', 'shop_id', 'rtn_cnt', 'ord_dt', 'offer_cnt']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		print not((file_exists))
		if not(file_exists):
		    writer.writeheader()  # file doesn't exist yet, write a header
		#writer.writeheader()
		writer.writerow(line)


if __name__ == '__main__':

    #with open('t_order.csv') as csvfile:
    #	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #	for i,line in enumerate(spamreader):

    with open('t_order.csv', 'r') as csvfile:
    	reader = csv.DictReader(csvfile)
    	for i,row in enumerate(reader):
    		#print(row['pid'], row['shop_id'])
    		process_line(row)
    		if i ==20:
    			break
    '''
   
    order = pd.read_csv(r't_ads.csv',usecols=(4,8))#,nrows=2000,usecols=(1,4)
    print (product[1:100])
    print (len(product))
    print (len(sorted(set(product['pid']))))
    # 删除含NAN列的行，删除NAN行
    order.dropna(how='all')
    order.dropna()
    print ('-------------保存所有店铺ip到文件----------------')
    shops = sorted(set(order['shop_id']))
    goods = sorted(set(order['pid']))

    print ('total shop number:',len(shops))
    '''
