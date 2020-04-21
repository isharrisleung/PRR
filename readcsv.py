import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import time
import datetime
from shapely import geometry
import os



# match the location string:"xxxx,xxxxx"
pattern = re.compile(r'([-|\d|\.]+),([-|\d|\.]+)')

def cal_date():
    data = pd.read_csv('./data/porto_taxi_data/porto_taxi_tractory.csv')
    date_list = list(data["TIMESTAMP"])
    length = len(date_list)
    starttime = datetime.datetime(2013, 7, 1)
    li = []
    for i in range(length):
        date_time = date_list[i]
        y = time.gmtime(date_time).tm_year
        m = time.gmtime(date_time).tm_mon
        d = time.gmtime(date_time).tm_mday

        date1=datetime.datetime(y, m, d)
        # print(date1)
        # print(date1-starttime)
        # input()
        li.append((date1-starttime).days)
    sns.distplot(li)
    plt.show()

def filter_tractory(f):
    print("filter trajectory")
    """filter tractory
    """
    data = pd.read_csv('./data/porto_taxi_data/porto_user_tractory.csv')
    li = []
    data_list = list(data["POLYLINE"])
    call_list = list(data["CALL_TYPE"])
    length = len(data)
    print(length)
    for i in range(length):
        if (i+1) % 1000 == 0:
            print("完成{}/{}，{}%".format(i+1, length, round((i+1)/length*100, 2)))
        tra = data_list[i]
        locations = re.findall(pattern, tra)
        if len(locations) > 50 or len(locations) < 10:
            # 筛选轨迹长度
            li.append(i) 
        elif call_list[i] != "A":
            li.append(i)
        else :
            for j in locations:
                # 筛选轨迹边缘
                if float(j[0]) > -8.549 or float(j[0]) < -8.718 or float(j[1]) < 41.0936 or float(j[1]) > 41.22:
                    li.append(i)
    data.drop(li, inplace=True)
    df = data[["ORIGIN_CALL", "TIMESTAMP", "POLYLINE"]]
    length -= len(li)
    data_filter = []
    user = {}
    tra_list = list(df["POLYLINE"])
    uid_list = list(df["ORIGIN_CALL"])
    tim_list = list(df["TIMESTAMP"])
    for i in range(length):
        uid = uid_list[i]
        if uid not in user:
            user[uid] = [i]
        else:
            user[uid].append(i)
    for u in user:
        tra_index_li = user[u]
        if len(tra_index_li) < 10 or len(tra_index_li) > 50:
            # 筛选轨迹条数
            continue
        else:
            start_tim_li = []
            tra_li = []
            for tra_index in tra_index_li:
                start_tim_li.append(tim_list[tra_index])
                ps = re.findall(pattern, tra_list[tra_index])
                t = [[p[0], p[1]] for p in ps]
                tra_li.append(t) 
            dic = {"uid" : u, "start_time": start_tim_li, "polyline": tra_li}
            data_filter.append(dic)
    print(len(data_filter))
    with open(f, "wb") as file:
        pickle.dump(data_filter, file)

def show_loc():
    # 查看location的位置范围
    locations = []
    pattern = re.compile(r'([-|\d|\.]+),([-|\d|\.]+)')
    data = pd.read_csv('./data/porto_taxi_data/porto_taxi_tractory.csv')
    tra_list = list(data["POLYLINE"])
    length = len(tra_list)
    lat_list = []
    lng_list = []
    for i in range(length):
        tra = tra_list[i]
        location = re.findall(pattern, tra)
        for i in location:
            lat_list.append(float(i[0]))
            lng_list.append(float(i[1]))
    lat_list.sort()
    lng_list.sort()
    print(lat_list[0:10])
    print(lat_list[-10:])
    print(lng_list[0:10])
    print(lng_list[-10:])

def convert2GpsTrajectCsv(f):
    print("convert trajecory to GpsTrajectCsv")
    data = None
    n = []
    geom = []
    count = 0
    with open("./data/porto_taxi_data/data_filter.pkl", "rb") as file:
        data = pickle.load(file)
    for i in data:
        for j in i["polyline"]:
            count += 1
            l = [(float(a[0]), float(a[1])) for a in j]
            line = geometry.LineString(l)
            n.append(count)
            geom.append(line)
    dic = {"id": n, "geom": geom}
    df = pd.DataFrame(dic)
    df.to_csv(f, sep=";", index=0)

def convert2GpsPointCsv(f):
    print("convert trajecory to GpsPointsCsv")
    data = None
    n = []
    geomx = []
    geomy = []
    count = 0
    with open("./data/porto_taxi_data/data_filter.pkl", "rb") as file:
        data = pickle.load(file)
    for i in data:
        for j in i["polyline"]:
            count += 1
            l = [(float(a[0]), float(a[1])) for a in j]
            for p in l:
                n.append(count)
                geomx.append(p[0])
                geomy.append(p[1])

    dic = {"id": n, "x": geomx, "y": geomy}
    df = pd.DataFrame(dic)
    df.to_csv(f, sep=";", index=0)

def read_fmm():
    file = open("./data/porto_taxi_data/gps_fmm.txt")
    fmm_list = {}   # 有效的fmm路径 {i_d : [cpath, mageom],}
    for i in file.readlines():
        i_d, cpath, mageom = i.split(";")
        if not cpath == "":
            fmm_list[i_d] = [cpath, mageom]
    with open("./data/porto_taxi_data/data_filter.pkl", "rb") as file:
        data = pickle.load(file)
    count = 0
    tra_lookup = {}
    start_time_list = []
    for i in data:
        uid = i["uid"]
        start_time_list.extend(i["start_time"])
        tra_lookup[uid] = []
        for j in i["polyline"]:
            count += 1
            tra_lookup[uid].append(count)
    with open("./data/porto_taxi_data/gps_fmm_2.txt", 'w') as f:
        f.write("uid;timestamp;capth;mageom\n")
        for uid in tra_lookup:
            ti = tra_lookup[uid]
            for i in ti:
                u = str(uid)
                start_time = str(start_time_list[i-1])
                if str(i) in fmm_list:
                    ts = ";".join(fmm_list[str(i)])
                else:
                    continue
                f.write(";".join([u, start_time, ts]))
    file.close()
    pass

def filter_fmm_data():
    file = open("./data/porto_taxi_data/gps_fmm_2.txt")
    i_d_dic = []
    tim_dic = []
    cpath_dic = []
    mageom_dic = []
    u = {}   
    for i in file.readlines():
        i_d, tim, cpath, mageom = i.strip().split(";")
        if i_d in u:
            u[i_d].append([tim, cpath, mageom])
        else:
            u[i_d] = [[tim, cpath, mageom]]
    file.close()
    for i in u:
        if i == "uid":
            continue
        ta = u[i]
        # print(ta)
        # input()
        if len(ta) >= 10:
            for a in ta:
                # print(a)
                i_d_dic.append(i)
                tim_dic.append(a[0])
                cpath_dic.append(a[1])
                mageom_dic.append(a[2])
    data = {"uid" : i_d_dic, "timestamp" : tim_dic, "cpath" : cpath_dic, "mageom" : mageom_dic}
    df = pd.DataFrame(data)
    df.to_csv("./data/porto_taxi_data/gps_fmm.csv", index=0)


if __name__ == "__main__":
    # cal_date()
    # 筛选用户
    if not os.path.exists("./data/porto_taxi_data/data_filter.pkl"):
        filter_tractory("./data/porto_taxi_data/data_filter.pkl")
    if not os.path.exists("./data/porto_taxi_data/trips.csv"):
        convert2GpsTrajectCsv("./data/porto_taxi_data/trips.csv")
    if not os.path.exists("./data/porto_taxi_data/gps.csv"):
        convert2GpsPointCsv("./data/porto_taxi_data/gps.csv")
    print("Data convert finished.")
    print("Can process Fast map matching")
    if not os.path.exists("./data/porto_taxi_data/gps_fmm_2.txt"):
        read_fmm()
    if not os.path.exists("./data/porto_taxi_data/gps_fmm_2.txt"):
        filter_fmm_data()

            

    