import osmnx as ox
from shapely.geometry import shape
import pandas as pd
# p = re.compile(r'([-|\d|\.]+),([-|\d|\.]+)')
# json_file = open("./data/stockholm_boundary.geojson")
# data = json.load(json_file)
# a = data["features"][0]['geometry']['coordinates']
# print(a)
# json_file.close()

# x_min = -8.7200000
# x_max = -8.5500000
# y_min = 41.0936000
# y_max = 41.2200000
# x_diff = x

file = pd.read_csv("./data/x.csv")
x = [float(i) for i in list(file["x"])]
y = [float(i) for i in list(file["y"])]
length = len(x)
p = []
for i in range(length):
    p.append([x[i],y[i]])
p = [p]
dic = {"type": "Polygon", "coordinates": p}
boundary_polygon = shape(dic)
G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
ox.save_graph_shapefile(G, filename='stockholm')


# x = []
# y = []
# for i in a:
#     x.append(i[0])
#     y.append(i[1])

# plt.scatter(x,y, color='k', s=25, marker=".")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Interesting Graph\nCheck it out')
# plt.legend()
# plt.show()
# boundary_polygon = shape(data["features"][0]['geometry'])
