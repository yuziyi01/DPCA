import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy import stats


def general_data(distrNum):
    mean1 = [5, 30]
    cov1 = [[10, 15], [15, 50]]
    data = np.random.multivariate_normal(mean1, cov1, distrNum)

    mean2 = [10, 10]
    cov2 = [[10, -20], [-20, 50]]
    data = np.append(data, np.random.multivariate_normal(mean2, cov2, distrNum), 0)

    mean3 = [25, 25]
    cov3 = [[30, -18], [-18, 30]]
    data = np.append(data, np.random.multivariate_normal(mean3, cov3, distrNum), 0)

    mean4 = [40, 40]
    cov4 = [[50, 25], [25, 30]]
    data = np.append(data, np.random.multivariate_normal(mean4, cov4, distrNum), 0)

    mean5 = [60, 30]
    cov5 = [[60, -50], [-50, 60]]
    data = np.append(data, np.random.multivariate_normal(mean5, cov5, distrNum), 0)
    return data

def getDistCut(distList, distPercent):
    # return max(distList) * distPercent / 100
    abc=[]
    n=len(distList)
    for i in range(n-1):
        for j in range(i+1,n):
            abc.append(distList[i,j])
    aaa=sorted(abc)
    position = round(len(aaa) * distPercent / 100)
    dc = aaa[position - 1]
    # print(aaa[0],dc,aaa[len(aaa)-1])
    return dc

#使用了高斯核计算点密度
def getRho2(n, distMatrix, distCut):
    rho = np.zeros(n, dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rho[i] = rho[i] + np.exp(-(distMatrix[i, j] / distCut) ** 2)
            rho[j] = rho[j] + np.exp(-(distMatrix[i, j] / distCut) ** 2)
    return rho

#正常计算点密度--输出1维数组
def getRho1(n,distMatrix,distCut):
    rho = np.zeros(n,dtype=float)
    for i in range(n-1):
        for j in range(i+1,n):
            if distMatrix[i,j] < distCut:
                rho[i] += 1
                rho[j] += 1
    # print("rho:",rho[:10])
    return rho


# ------------密度峰值聚类------------------#
def DPCA(n, distMatrix, rho,distCut):
    # rho序号  rho由大到小排列
    rhoOrdIndex = np.flipud(np.argsort(rho))
    delta = np.zeros(n, dtype=float)
    leader = np.ones(n, dtype=int) * int(-1)
    # delta就是与其他密度更高的点之间的最小距离δ
    '''获取密度最大样本的Delta和Leader'''
    maxdist = 0
    for ele in range(n):
        if distMatrix[rhoOrdIndex[0], ele] > maxdist:
            maxdist = distMatrix[rhoOrdIndex[0], ele]
    delta[rhoOrdIndex[0]] = maxdist
    #Leader是非密度最大样本对应的具有最小距离的密度更高点的坐标
    '''获取非密度最大样本的Delta和Leader'''
    for i in range(1, n):
        mindist = np.inf
        minindex = -1
        for j in range(i):
            if distMatrix[rhoOrdIndex[i], rhoOrdIndex[j]] < mindist:
                mindist = distMatrix[rhoOrdIndex[i], rhoOrdIndex[j]]
                minindex = rhoOrdIndex[j]
        delta[rhoOrdIndex[i]] = mindist
        leader[rhoOrdIndex[i]] = minindex

    # 决策图
    plt.scatter(rho, delta,s=1)
    plt.show()
    #gamma就是δ*rho 聚类依据--数组
    gamma = delta * rho

    #gamma图
    gg=gamma.tolist()
    gg=sorted(gg)
    gg.reverse()
    print(gg)
    x_index=[]
    for i in range(len(rho)):
        x_index.append(i+1)
    plt.scatter(x_index, gg,s=1)
    plt.show()
    #从大到小
    gammaOrdIdx = np.flipud(np.argsort(gamma))
    '''开始聚类'''
    clusterIdx = np.ones(n, dtype=int) * (-1)
    # ------初始化聚类中心-------#
    blockNum = input("请输入聚类中心个数 ：")
    blockNum=int(blockNum)
    for k in range(blockNum):
        clusterIdx[gammaOrdIdx[k]] = k
    # ------对中心点以外样本进行聚类-----------#
    #按点密度由高到低算--聚类标签
    for i in range(n):
        if clusterIdx[rhoOrdIndex[i]] == -1:
            clusterIdx[rhoOrdIndex[i]] = clusterIdx[leader[rhoOrdIndex[i]]]

    clusterIdx_no_noise=clusterIdx.copy()
    # 计算噪声点
    pb = np.zeros(blockNum, dtype=float)
    for i in range(n-1):
        for j in range(i+1,n):
            if clusterIdx[i]!=clusterIdx[j] and distMatrix[i,j]<distCut:
                pp=(rho[i]+rho[j])/2
                if pp>pb[clusterIdx[i]]:
                    pb[clusterIdx[i]]=pp
                if pp > pb[clusterIdx[j]]:
                    pb[clusterIdx[j]] = pp

    for i in range(n):
        if rho[i]<pb[clusterIdx[i]]:
            clusterIdx[i]=blockNum+1

    return clusterIdx_no_noise,clusterIdx


distrNum=500
# 生成数据
X=general_data(distrNum)
plt.scatter(X[:, 0], X[:, 1], s=1,c='b')
plt.show()
n = len(X)
distPercent = 2.0
distList = pdist(X, metric='euclidean')
distMatrix = squareform(distList)
distCut = getDistCut(distMatrix, distPercent)
print(distCut)
#getRho1使用截断核计算，getRho2使用高斯核
rho = getRho1(n, distMatrix, distCut)
clusterIdx_no_noise,clusterSet = DPCA(n,distMatrix,rho,distCut)

#计算正确率
errorNum=0
d=[]
for i in range(5):
    d.append(stats.mode(clusterIdx_no_noise[i*distrNum:(i+1)*distrNum])[0][0])
for i in range(len(clusterIdx_no_noise)):
    if clusterIdx_no_noise[i]!=d[int(i/distrNum)]:
        errorNum+=1
true_rate=(2500-errorNum)/2500
print("聚类正确率：",true_rate)
plt.scatter(X[:, 0], X[:, 1], s=1,c=clusterIdx_no_noise,cmap='Set1')
plt.show()

plt.scatter(X[:, 0], X[:, 1], s=1,c=clusterSet,cmap='Set1')
plt.show()
