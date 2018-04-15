# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import csv
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA 

monthes = ['2016-08', '2016-09', '2016-10', '2016-11',
           '2016-12', '2017-01', '2017-02', '2017-03', '2017-04']

monthes_sale = ['2016-06','2016-07','2016-08', '2016-09', '2016-10', '2016-11',
           '2016-12', '2017-01']


def convertTime(x):
    # print(x)
    return x[:7]


def dataPreprocess():
    order = pd.read_csv(
        r't_order.csv')#,nrows=2000
    product = pd.read_csv(
        r't_product.csv')
    ads = pd.read_csv(r't_ads.csv')#,nrows=2000
    comments = pd.read_csv(
        r't_comment.csv')#,nrows=2000
    sales = pd.read_csv(
        #r'G:\data\京东金融大赛\Sales_Forecast_Qualification\t_sales_sum.csv')
        r't_sales_sum.csv')#,nrows=2000

    # 删除含NAN列的行，删除NAN行
    order.dropna(how='all')
    order.dropna()
    product.dropna(how='all')
    product.dropna()
    ads.dropna(how='all')
    ads.dropna()
    comments.dropna(how='all')
    comments.dropna()
    sales.dropna(how='all')
    sales.dropna()

    # 将广告费信息整合
    shop_ads_month = {}
    shops = sorted(set(ads['shop_id']))
    ads['create_dt'] = ads['create_dt'].apply(convertTime)
    #print('shops of ads', shops)
    for shop in shops:
        tmp = {}
        for month in monthes:
            tmp[month] = {}
        shop_ads_month[shop] = tmp
    #print ads.loc[0:10]
    print('length of ads', len(ads))
    for i in range(0, len(ads.index)):
        currentshop = ads.loc[i, 'shop_id']
        # print(currentshop)
        currentmonth = ads.loc[i, 'create_dt']
        tmp = {}
        tmp['charge'] = ads.loc[i, 'charge']
        tmp['consume'] = ads.loc[i, 'consume']

        tmpresult_shop = shop_ads_month[currentshop]
        tmp3 = tmpresult_shop[currentmonth]
        if currentshop == 1980:
            print(tmp3, tmp)
        if tmp3:
            tmp['charge'] = tmp['charge'] + tmp3['charge']
            tmp['consume'] = tmp['consume'] + tmp3['consume']
        tmpresult_shop[currentmonth] = tmp
        shop_ads_month[currentshop] = tmpresult_shop
    
    with open(r'shop_month_ads_new.txt', 'wb') as f:
        pickle.dump(shop_ads_month, f)

    with open(r'shop_month_ads_new.txt', 'rb') as f:
        a = pickle.load(f)
        print(a[1980])

    # 将评价信息整合
    shop_comments_month = {}
    shops = sorted(set(comments['shop_id']))
    #print('shops of comments', shops)

    comments['create_dt'] = comments['create_dt'].apply(convertTime)
    for shop in shops:
        tmp = {}
        for month in monthes:
            tmp[month] = {}
        shop_comments_month[shop] = tmp

    for i in range(0, len(comments.index)):
        currentshop = comments.loc[i, 'shop_id']
        # print(currentshop)
        currentmonth = comments.loc[i, 'create_dt']
        tmp = {}
        tmp['bad_num'] = comments.loc[i, 'bad_num']
        tmp['cmmt_num'] = comments.loc[i, 'cmmt_num']
        tmp['dis_num'] = comments.loc[i, 'dis_num']
        tmp['good_num'] = comments.loc[i, 'good_num']
        tmp['mid_num'] = comments.loc[i, 'mid_num']

        tmpresult_shop = shop_comments_month[currentshop]
        tmp4 = tmpresult_shop[currentmonth]
        if tmp4:
            tmp['bad_num'] = tmp['bad_num'] + tmp4['bad_num']
            tmp['good_num'] = tmp['good_num'] + tmp4['good_num']
            tmp['cmmt_num'] = tmp['cmmt_num'] + tmp4['cmmt_num']
            tmp['dis_num'] = tmp['dis_num'] + tmp4['dis_num']
            tmp['mid_num'] = tmp['mid_num'] + tmp4['mid_num']

        tmpresult_shop[currentmonth] = tmp

        shop_comments_month[currentshop] = tmpresult_shop

    with open(r'shop_month_comments_new.txt', 'wb') as f:
        pickle.dump(shop_comments_month, f)

    with open(r'shop_month_comments_new.txt', 'rb') as f:
        a = pickle.load(f)
        #print(a[1630])

    # 将销售信息整合
    shop_sales_month = {}
    shops = sorted(set(sales['shop_id']))
    #print('shops of sales', shops)
    sales['dt'] = sales['dt'].apply(convertTime)
    for shop in shops:
        tmp = {}
        for month in monthes:
            tmp[month] = {}
        shop_sales_month[shop] = tmp

    for i in range(0, len(sales.index)):
        currentshop = sales.loc[i, 'shop_id']
        # print(currentshop)
        currentmonth = sales.loc[i, 'dt']
        tmp = {}
        tmp['sale_amt_3m'] = sales.loc[i, 'sale_amt_3m']

        tmpresult_shop = shop_sales_month[currentshop]
        tmpresult_shop[currentmonth] = tmp
        shop_sales_month[currentshop] = tmpresult_shop

    with open(r'shop_month_sales_new.txt', 'wb') as f:
        pickle.dump(shop_sales_month, f)

    with open(r'shop_month_sales_new.txt', 'rb') as f:
        a = pickle.load(f)
        print(a[1630])

    # 将订单数据按店铺、月份整合
    shop_month = {}
    # head_order = order.head()
    order['ord_dt'] = order['ord_dt'].apply(convertTime)
    shops = sorted(set(order['shop_id']))
    #print('shops of order', shops)

    for shop in shops:
        tmp = {}
        for month in monthes:
            tmp[month] = {}
        shop_month[shop] = tmp
    for i in range(0, len(order.index)):
        pid = order.loc[i, 'pid']

        currentshop = order.loc[i, 'shop_id']
        # print(currentshop)
        currentmonth = order.loc[i, 'ord_dt']
        # month = time.strptime(sub_month.loc[i, 'ord_dt'], "%Y-%m")
        tmpdit = {}
        tmpdit['sale_amt'] = order.loc[i, 'sale_amt']
        tmpdit['offer_amt'] = order.loc[i, 'offer_amt']
        tmpdit['offer_cnt'] = order.loc[i, 'offer_cnt']
        tmpdit['rtn_cnt'] = order.loc[i, 'rtn_cnt']
        tmpdit['user_cnt'] = order.loc[i, 'user_cnt']
        tmpdit['rtn_amt'] = order.loc[i, 'rtn_amt']
        tmpdit['ord_cnt'] = order.loc[i, 'ord_cnt']

        # 更新字典,注意到同一商品在同一月份可能有多条记录
        tmpresult_shop = shop_month[currentshop]
        tmpresult_month = tmpresult_shop[currentmonth]
        if pid in sorted(tmpresult_month.keys()):
            tmp5=tmpresult_month[pid]
            tmpdit['sale_amt'] += tmp5['sale_amt']
            tmpdit['offer_amt'] += tmp5['offer_amt']
            tmpdit['offer_cnt'] += tmp5['offer_cnt']
            tmpdit['rtn_cnt'] += tmp5['rtn_cnt']
            tmpdit['user_cnt'] += tmp5['user_cnt']
            tmpdit['rtn_amt'] += tmp5['rtn_amt']
            tmpdit['ord_cnt'] += tmp5['ord_cnt']
        tmpresult_month[pid] = tmpdit
        tmpresult_shop[currentmonth] = tmpresult_month
        shop_month[currentshop] = tmpresult_shop
        # print(shop,shop.type)
        # t(shop_month[shop])

    
    # print(shop_month.keys())
    with open(r'shop_month_order_new.txt', 'wb') as f:
        pickle.dump(shop_month, f)

    with open(r'shop_month_order_new.txt', 'rb') as f:
        a = pickle.load(f)
        print(a[1630])
    


def featureExtract_train(featureLength):
    with open(r'shop_month_order_new.txt','rb') as f:
        order_month = pickle.load(f)
    with open(r'shop_month_ads_new.txt', 'rb') as f:
        ads_month = pickle.load(f)
    with open(r'shop_month_comments_new.txt', 'rb') as f:
        comments_month = pickle.load(f)
    with open(r'shop_month_sales_new.txt', 'rb') as f:
        sales_month = pickle.load(f)
    shops = sorted(sales_month.keys())#order_month,0415
    print '-----------------featureExtract_train---------------------'



    print sales_month[1444L]

    # 指定特征提取的时间跨度，可取3个月、2个月、1个月
    #featureLength = 1
    trainFeature = []

    monthes_train = ['2016-08', '2016-09', '2016-10', '2016-11',
                     '2016-12', '2017-01']
    featureMatrix = []
    labels = []
    for i in range(0, len(monthes_train) + 1 - featureLength):
        month_list = monthes_train[i: i + featureLength]
        feature, label = featureofMonth(
            order_month, ads_month, comments_month, sales_month, month_list, shops)
        featureMatrix.extend(feature)
        labels.extend(label)
    print('特征提取成功！')
    print(len(featureMatrix), len(labels), len(featureMatrix[0]))
    # print(featureMatrix[0])
    labels = np.array(labels).reshape(len(labels), 1)

    featureMatrix = np.hstack((labels, np.array(featureMatrix)))
    featureSave(featureLength, featureMatrix, 'train')


def featureExtract_test(featureLength):
    with open(r'shop_month_order_new.txt', 'rb') as f:
        order_month = pickle.load(f)
    with open(r'shop_month_ads_new.txt', 'rb') as f:
        ads_month = pickle.load(f)
    with open(r'shop_month_comments_new.txt', 'rb') as f:
        comments_month = pickle.load(f)
    with open(r'shop_month_sales_new.txt', 'rb') as f:
        sales_month = pickle.load(f)
    shops = sorted(order_month.keys())

    print sales_month[68L]

    #print(shops)
    # 指定特征提取的时间跨度，可取3个月、2个月、1个月
    #featureLength = 1
    trainFeature = []

    monthes_test = ['2017-02', '2017-03', '2017-04', ]
    #featureMatrix = []
    labels = []
    month_list = monthes_test[-featureLength:]

    feature = featureofMonth(
        order_month, ads_month, comments_month, sales_month, month_list, shops, 'test')

    print('特征提取成功！')
    #print(len(featureMatrix), len(labels), len(featureMatrix[0]))
    # print(featureMatrix[0])
    #labels = np.array(labels).reshape(len(labels), 1)

    #featureMatrix = np.hstack((labels, np.array(featureMatrix)))
    featureSave(featureLength, feature, 'test')


def featureofMonth(order_month, ads_month, comments_month, sales_month, month_list, shops, workMode='train'):
    #shops = sorted(order_month.keys())
    features = []
    datas=[]
    for month in month_list:
        dataOfOneMonth = []
        for shop in shops:

            tmp=[]
            orderofCurrentMonth = order_month[shop][month]
            # 获取当月每种商品的销量、销售额、退货量、优惠、顾客数
            sales_amt_per_product = list(
                sale['sale_amt'] for sale in orderofCurrentMonth.values())
            offer_amt_per_product = list(
                sale['offer_amt'] for sale in orderofCurrentMonth.values())
            rtn_cnt_per_product = list(
                sale['rtn_cnt'] for sale in orderofCurrentMonth.values())
            rtn_amt_per_product = list(
                sale['rtn_amt'] for sale in orderofCurrentMonth.values())
            ord_cnt_per_product = list(
                sale['ord_cnt'] for sale in orderofCurrentMonth.values())
            user_cnt_per_product = list(
                sale['user_cnt'] for sale in orderofCurrentMonth.values())

            tmp.append(np.sum(sales_amt_per_product))
            tmp.append(np.sum(offer_amt_per_product))
            tmp.append(np.sum(rtn_cnt_per_product))
            tmp.append(np.sum(rtn_amt_per_product))
            tmp.append(np.sum(ord_cnt_per_product))
            tmp.append(np.sum(user_cnt_per_product))

            
            
            if shop in sorted(comments_month.keys()):
                commentsofCurrentMonth = comments_month[shop][month]
                #print(shop, month)
                if commentsofCurrentMonth:
                    good_num = commentsofCurrentMonth['good_num']
                    bad_num = commentsofCurrentMonth['bad_num']
                    cmmt_num = commentsofCurrentMonth['cmmt_num']
                    mid_num = commentsofCurrentMonth['mid_num']
                    dis_num = commentsofCurrentMonth['dis_num']

                    tmp.append(good_num)
                    tmp.append(bad_num)
                    tmp.append(cmmt_num)
                    tmp.append(mid_num)
                    tmp.append(dis_num)
                    
                else:
                    
                    tmp.append(-0.00001)
                    tmp.append(-0.00001)
                    tmp.append(-0.00001)
                    tmp.append(-0.00001)
                    tmp.append(-0.00001)
            else:
                
                tmp.append(-0.00001)
                tmp.append(-0.00001)
                tmp.append(-0.00001)
                tmp.append(-0.00001)
                tmp.append(-0.00001)
           

            # 缺失值处理
            if shop in sorted(ads_month.keys()):
                adsofCurrentMonth = ads_month[shop][month]
                #print(adsofCurrentMonth, shop, month)
                if adsofCurrentMonth:
                    adscost = adsofCurrentMonth['consume']
                    tmp.append(adscost)
                    
                else:
                    tmp.append(-0.00001)
                    
            else:
                tmp.append(-0.00001)
                

            # 将三部分特征组合起来
            


            dataOfOneMonth.append(tmp)
        datas.append(dataOfOneMonth)

    featureof3Month=np.array(datas[0])+np.array(datas[1])+np.array(datas[2])
    featureof2Month=np.array(datas[1])+np.array(datas[2])
    featureof1Month=np.array(np.array(datas[2]))

    featureFinal=[]
    for i in range(0,len(shops)):
        tmp2=[]
        for j in range(0,len(featureof3Month[0])):
            tmp2.append(featureof3Month[i][j]/3)
            tmp2.append(featureof2Month[i][j]/2)
            tmp2.append(featureof1Month[i][j])

        tmp2.append(featureof3Month[i][0]/90)
        tmp2.append(featureof3Month[i][4]/90)
        tmp2.append(featureof3Month[i][5]/90)
        tmp2.append(featureof3Month[i][1]/featureof3Month[i][0])
        tmp2.append(featureof3Month[i][3]/featureof3Month[i][0])
        tmp2.append(featureof3Month[i][2]/featureof3Month[i][4])
        tmp2.append(featureof3Month[i][0]/featureof3Month[i][4])
        tmp2.append(featureof3Month[i][0]/featureof3Month[i][5])
        tmp2.append(featureof3Month[i][1]/featureof3Month[i][5])

        tmp2.append(featureof3Month[i][6]/featureof3Month[i][8])
        tmp2.append(featureof3Month[i][7]/featureof3Month[i][8])
        tmp2.append(featureof3Month[i][9]/featureof3Month[i][8])
        tmp2.append(featureof3Month[i][10]/featureof3Month[i][8])

        tmp2.append(featureof3Month[i][11]/featureof3Month[i][0])

        tmp2.append(featureof2Month[i][0]/60)
        tmp2.append(featureof2Month[i][4]/60)
        tmp2.append(featureof2Month[i][5]/60)
        tmp2.append(featureof2Month[i][1]/featureof2Month[i][0])
        tmp2.append(featureof2Month[i][3]/featureof2Month[i][0])
        tmp2.append(featureof2Month[i][2]/featureof2Month[i][4])
        tmp2.append(featureof2Month[i][0]/featureof2Month[i][4])
        tmp2.append(featureof2Month[i][0]/featureof2Month[i][5])
        tmp2.append(featureof2Month[i][1]/featureof2Month[i][5])

        tmp2.append(featureof2Month[i][6]/featureof2Month[i][8])
        tmp2.append(featureof2Month[i][7]/featureof2Month[i][8])
        tmp2.append(featureof2Month[i][9]/featureof2Month[i][8])
        tmp2.append(featureof2Month[i][10]/featureof2Month[i][8])

        tmp2.append(featureof2Month[i][11]/featureof2Month[i][0])

        tmp2.append(featureof1Month[i][0]/30)
        tmp2.append(featureof1Month[i][4]/30)
        tmp2.append(featureof1Month[i][5]/30)
        tmp2.append(featureof1Month[i][1]/featureof1Month[i][0])
        tmp2.append(featureof1Month[i][3]/featureof1Month[i][0])
        tmp2.append(featureof1Month[i][2]/featureof1Month[i][4])
        tmp2.append(featureof1Month[i][0]/featureof1Month[i][4])
        tmp2.append(featureof1Month[i][0]/featureof1Month[i][5])
        tmp2.append(featureof1Month[i][1]/featureof1Month[i][5])

        tmp2.append(featureof1Month[i][6]/featureof1Month[i][8])
        tmp2.append(featureof1Month[i][7]/featureof1Month[i][8])
        tmp2.append(featureof1Month[i][9]/featureof1Month[i][8])
        tmp2.append(featureof1Month[i][10]/featureof1Month[i][8])

        tmp2.append(featureof1Month[i][11]/featureof1Month[i][0])

        featureFinal.append(tmp2)

    if workMode == 'train':
        #将当前几个月份的销售额作为特征
        monthpre=monthes_sale[monthes_sale.index(month_list[0])-1]
        monthpre2=monthes_sale[monthes_sale.index(month_list[0])-2]
        #print(monthpre)
        sales_pre=[]
        for shop in shops:
        	print shop
        	sale1=sales_month[shop][monthpre]['sale_amt_3m']
        	sale2=sales_month[shop][monthpre2]['sale_amt_3m']
        	sales_pre.append([sale1,sale2,sale1-sale2,(sale1-sale2)/sale1])
        #sales_pre=np.array(sales_pre).reshape((len(sales_pre),1))
        featureFinal=np.hstack((featureFinal,sales_pre))

        month_last = month_list[-1]
        tmpLabel = []
        for shop in shops:
            tmpLabel.append(sales_month[shop][month_last]['sale_amt_3m'])
        return featureFinal, tmpLabel

    sales_pre=[]
    for shop in shops:
        sale1=sales_month[shop]['2017-01']['sale_amt_3m']
        sale2=sales_month[shop]['2016-12']['sale_amt_3m']
        
            
        sales_pre.append([sale1,sale2,sale1-sale2,(sale1-sale2)/sale1])
        
            
        
    #sales_pre=np.array(sales_pre).reshape((len(sales_pre),4))
    #print(np.array(sales_pre).shape)
    featureFinal=np.hstack((featureFinal,sales_pre))
    return featureFinal


def featureSave(featureLen, features, datatype):
    featurePath = datatype + '_featureLength_' + str(featureLen) + '_new.txt'
    print(featurePath)
    with open(featurePath, 'w') as f:
        for row in features:
            line = ''
            for item in row:
                line += str(item) + ' '
            f.write(line)
            f.write('\n')


def readData(path, workMode):
    if workMode == 'train':
        feature=[]
        label=[]
        with open(path, 'r') as f:
            for line in f:
                line=line.strip().split(' ')
                feature.append(line[1:])
                label.append(line[0])
        return np.array(feature).astype(float),np.array(label).astype(float)
    if workMode == 'test':
        feature=[]
        with open(path, 'r') as f:
            for line in f:
                line=line.strip().split(' ')
                feature.append(line)
        return np.array(feature).astype(float)

def extendedFeatureMatrix(X,y):
    seed = 7
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=seed)

    model = RandomForestRegressor()
    #learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    #param_grid = dict(learning_rate=learning_rate)
    parameters = {
            'n_estimators': [200,400,500],
            'max_features': ['sqrt'],
            'max_depth':[4,6,8,10],
            # 'max_leaf_nodes', 'bootstrap',
            #'min_samples_leaf': [20],
            #'max_leaf_nodes': [2],
            #  'n_estimators', 'min_samples_split', 'min_weight_fraction_leaf', 'criterion', 'random_state', 'min_impurity_split', 'max_features', 'max_depth', 'class_weight'
            #'min_sample_leaf':[30,40,50,60,70,80],
            'n_jobs': [-1],
            #'random_state': [50],
        }
    grid_search = GridSearchCV(model, parameters, scoring="neg_mean_absolute_error")
    grid_result = grid_search.fit(X_train, y_train)

    X_noise=np.random.normal(loc=1,scale=1,size=(np.array(X).shape))
    X_noisy=X+X_noise
    y_predict=grid_search.predict(X_noisy)

    y_predict=np.array(y_predict).reshape((len(y_predict),1))

    return y_predict,grid_search
def trainAndPredict(featureLength):
    traindataPath=r'train_featureLength_'+str(featureLength)+'_hot_new.txt'
    testdataPath=r'test_featureLength_'+str(featureLength)+'_hot_new.txt'

    X,y=readData(traindataPath,'train')

    X[np.isnan(X)]=-0.00001
    pca=PCA(n_components=6)
    X=pca.fit_transform(X) 
    
    
    
    #y[np.isnan(y)]=0

    #imp = Imputer(missing_values=-0.00001, strategy='mean', axis=0) 
    #X=imp.fit_transform(X)
    #for row in X:
        #print(row)

    X_predict=readData(testdataPath,'test')
    X_predict[np.isnan(X_predict)]=-0.00001
    X_predict=pca.fit_transform(X_predict)
    
   
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=seed)

    model = XGBRegressor()
    #learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    #param_grid = dict(learning_rate=learning_rate)
    parameters = {
            'n_estimators': [200,400,600,800,1000],
            'max_depth': [6,8,9,10],
            #'nthread': [-1],
            #'min_child_weight':[4,6,8],
            'subsample':[0.6,0.7,0.75],
            #'colsample_bytree':[0.6,0.7] ,
            'learning_rate':[0.1,0.01,0.03,0.05]
            #objective = 'reg:linear'
            #'random_state': [50],
        }
    grid_search = GridSearchCV(model, parameters, scoring="neg_mean_absolute_error")
    grid_result = grid_search.fit(X_train, y_train)

    best_model=grid_search.best_estimator_
    y_predict=best_model.predict(X_test)

    test_error=mean_absolute_error(y_test, y_predict)

    print('the tset error is: ',test_error)
    print('the weighted tset error is: ',test_error*len(y_test)/np.sum(y_test))

    
    y_hat=best_model.predict(X_predict)

    #resultPath=r'result.csv'
    with open("predict_"+str(test_error)+'_'+str(featureLength)+"_new.csv","w",encoding='utf-8') as csvfile: 
        writer = csv.writer(csvfile,lineterminator='\n')
        
        for i in range(0,3000):
            writer.writerow([i+1,np.abs(y_hat[i])])

    joblib.dump(best_model,str(test_error)+'_'+str(featureLength)+'_xgboost_new.pkl')



def main():
    #dataPreprocess()
    featureLength=3
    featureExtract_train(featureLength)
    #featureExtract_test(featureLength)
    #trainAndPredict(featureLength)


if __name__ == '__main__':
    main()
