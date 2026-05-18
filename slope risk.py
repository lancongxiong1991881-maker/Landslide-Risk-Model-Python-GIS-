import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data={
    "location"     :["A","B","C","D","E"],
    "slope"        :[30,10,45,20,35],
    "rainfall"     :[200,100,300,150,250],
    "soil_strength":[50,80,40,70,60],
    "x"            :[1,2,3,4,5],
    "y"            :[1,2,1,4,1]
}

df = pd.DataFrame(data)     #DataFrame是指excel里面的表格

#把所有数据变成只有0-1之间
df["slope_n"]         = df["slope"]/df["slope"].max()
df["rainfall_n"]      = df["rainfall"]/df["rainfall"].max()
df["soil_strength_n"] = df["soil_strength"]/df["soil_strength"].max()

#normalize
df["risk_score"]=(
    0.5 * df["slope_n"] +
    0.3 * df["rainfall_n"] -
    0.2 * df["soil_strength_n"]
)

#classification
df["risk_level"] = pd.qcut(
    df["risk_score"],
    3,
    labels=["low","medium","high"]
)

print(df)

#most dangerous
most_dangerous = df.loc[df["risk_score"].idxmax()]                #loc是为了用条件找数据
print("The most dangerous place:")
print(most_dangerous)

#sensitivity model
weights={
    "model1":(0.5,0.3,0.2),
    "model2":(0.4,0.4,0.2),
    "model3":(0.6,0.2,0.2)
}

for name,(ws,wr,wso) in weights.items():
    df[name] = ws*df["slope_n"]+wr*df["rainfall_n"]-wso*df["soil_strength_n"]


print("Model 1 Ranking:")
print(df.sort_values("model1",ascending=False)["location"])

print("Model 2 Ranking:")
print(df.sort_values("model2",ascending=False)["location"])

print("Model 3 Ranking:")
print(df.sort_values("model3",ascending=False)["location"])

#modeling极端的条件(stress test)
df["slope_s"]    = df["slope_n"]*1.2
df["rainfall_s"] = df["rainfall_n"]*1.5
df["soil_s"]     = df["soil_strength_n"]

df["stress_risk"] = (
    df["slope_s"]*0.5+
    df["rainfall_s"]*0.3-
    df["soil_s"]*0.2
)

print(df[["location","risk_score","stress_risk"]])
print("Normalize:")
print(df.sort_values("risk_level",ascending=False)["location"])
print("Stress test:")
print(df.sort_values("stress_risk",ascending=False)["location"])


max_dangerous_stress = df.loc[df["stress_risk"].idxmax()]
print("The most dangerous place:")
print(max_dangerous_stress)

def zone(x):
    if x > 0.6:
        return "High"
    elif x > 0.3:
        return"Medium"
    else:
        return"Low"
    
df["hazard_zone"] = df["risk_score"].apply(zone)
print(df["hazard_zone"])

#做图象
df.groupby("risk_level",observed=False)["location"].count().plot(kind="bar")

plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Location")

plt.show()

plt.scatter(df["x"],df["y"], c=df["risk_score"], cmap="Reds", s=200)

for i , txt in enumerate(df["location"]):
    plt.text(df["x"][i],df["y"][i],txt)

plt.colorbar(label="Risk score")
plt.title("Landslide Hazard Map(Pseduo GIS)")
plt.show()

#scratter part
#创建地图网格
grid_x = np.linspace(0,1,50)
grid_y = np.linspace(0,1,50)

X,Y = np.meshgrid(grid_x,grid_y)        #meshgrid是把X轴和Y轴合成一个地图网格

#创建风险面
Z=(
    0.5*X+
    0.3*Y+
    0.2*(1 - X)                        #因为1-X是反向因子也就是我们所说的安全因子，因为for x来讲它不是全部都是危险的因子也有安全的
)

plt.contourf(X, Y, Z, cmap="Reds")      #contourf把高低不同的风险值画成颜色地图
plt.colorbar(label="Risk Level")

plt.title("Landslide Hazard Raster Map(GIS Concept)")
plt.xlabel("X(Spatial dimension)")
plt.ylabel("Y(Spatial dimension)")

plt.show()

#真实工程上的权重比
#经验公式
risk = (
    df["slope_n"]*0.45+
    df["rainfall_n"]*0.3-
    (1-df["soil_strength_n"])*0.25
)
print(risk)

#做不同模型的modeling
weight_risk={
    "risk1" : (0.45,0.3,0.25),
    "risk2" : (0.4,0.35,0.25),
    "risk3" : (0.5,0.25,0.25),
}

for name1,(rs,rr,rss) in weight_risk.items():
    df[name1] = rs*df["slope_n"]+rr*df["rainfall_n"]+(1-rss*df["soil_strength_n"])

print("Risk 1 Ranking:")
print(df.sort_values("risk1",ascending=False)["location"])
print("Risk 2 Ranking:")
print(df.sort_values("risk2",ascending=False)["location"])
print("Risk 3 Ranking:")
print(df.sort_values("risk3",ascending=False)["location"])

print(df[["risk1","risk2","risk3"]])

#stress testing
df["stress_risk1"]=(
    df["slope_s"]*0.45+
    df["rainfall_s"]*0.3-
    df["soil_s"]*0.2
)

print(df.sort_values("stress_risk1",ascending=False)["location"])