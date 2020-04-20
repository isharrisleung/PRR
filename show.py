import numpy as np  
import matplotlib.pyplot as plt   
import pandas as pd  
import os 
import gmplot
import re
import folium
# 让轨迹在地图上显示


def show_tra():
     # 在地图上显示轨迹
     locations = []
     pattern = re.compile(r'([-|\d|\.]+),([-|\d|\.]+)')
     data = pd.read_csv('./data/porto_taxi_data/porto_taxi_tractory.csv')
     tra_list = list(data["POLYLINE"])
     length = len(tra_list)
     for i in range(100):     # 选前100条轨迹
          tra = tra_list[i]
          locations.append(re.findall(pattern, tra))


     points = []
     for i in locations:
          points.append([[float(j[1]), float(j[0])] for j in i])
     # 初始化 地图 以及 放大倍数
     my_map = folium.Map(location=[points[0][0][0],points[-1][-1][1]], zoom_start=11)
     # loop each point
     # for i in points:
     #      # folium.Marker(i).add_to(my_map)
     for i in points:
          # 将不同的轨迹加入地图中
          folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(my_map)
     my_map.save("somelocs.html")

def show_points():
     # 在网格中显示坐标点
     locations = []
     pattern = re.compile(r'([-|\d|\.]+),([-|\d|\.]+)')
     data = pd.read_csv('./data/porto_taxi_data/porto_taxi_tractory.csv')
     tra_list = list(data["POLYLINE"])
     length = len(tra_list)
     for i in range(100000):     # 选前100条轨迹
          tra = tra_list[i]
          locations.append(re.findall(pattern, tra))

     x = []    # 经
     y = []    # 纬
     for i in locations:
          length = len(i)
          for j in range(length):
               x.append(float(i[j][0]))
               y.append(float(i[j][1]))
               # points.append([[float(j[1]), float(j[0])] for j in i])
     plt.scatter(x,y, color='k', s=25, marker=".")
     plt.xlabel('x')
     plt.ylabel('y')
     plt.title('Interesting Graph\nCheck it out')
     plt.legend()
     plt.show()


if __name__ == "__main__":
#     show_tra()
    show_points()
