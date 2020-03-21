#请关注微信公众号：小叶说
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sco
import scipy.interpolate as sci

#导入数据
data=pd.read_csv('market.csv',index_col=0)
symbols=data.columns
num=len(symbols)
rets=data.dropna()

prets=[]  # 投资组合的收益率
pvols=[]  # 投资组合的方差
I=100000  # 模拟次数
rets_mean_year=rets.mean()*252/100
rets_cov_year=rets.cov()*252/10000

for p in range(I):
    weights=np.random.random(num)
    weights/=np.sum(weights)
    prets.append(np.sum(rets_mean_year*weights))
    pvols.append(np.sqrt(np.dot(weights.T,np.dot(rets_cov_year,weights))))
prets=np.array(prets)
pvols=np.array(pvols)

prets

#画可行域
plt.figure(figsize=(8,5))
plt.scatter(pvols,prets,c=prets/pvols,marker='o')
plt.grid(True)
plt.xlim(0,0.16)
plt.ylim(0,0.25)
plt.xlabel('standard deviation')
plt.ylabel('return')
plt.colorbar(label='SR')

'''最优组合'''
def func_stats(weights,rets):
    weights=np.array(weights)
    #投资组合的收益率
    pret=np.sum(rets.mean()*weights)*252 
    #投资组合的标准差
    pvol=np.sqrt(np.dot(weights.T,np.dot(rets.cov()*252,weights))) 
    return np.array([pret,pvol,pret/pvol])

def func_min_sr(weights,rets):
#得到夏普比率最大时的期望收益
    return -func_stats(weights,rets)[2]

def func_min_variance(weights,rets):
#优化，最小化风险：方差最小化
    return func_stats(weights,rets)[1]**2

def func_min_vol(weights,rets):
    return func_stats(weights,rets)[1]

#求最大收益波动率比率:
cons=({'type':'eq','fun':lambda x:1-np.sum(x)})
bnds=tuple((0,1) for x in range(num))
opts=sco.minimize(lambda x:func_min_sr(x,rets),num*[1.0/num,],method='SLSQP',bounds=bnds,constraints=cons)
opt_sr_weights=opts['x'].round(3)

#有效前沿
ret_min=min(prets)
ret_max=max(prets)
vol_min=min(pvols)
ind_min_vol=np.argmin(pvols)
ret_start=prets[ind_min_vol]

trets=np.linspace(ret_start,ret_max,100)
tvols=[]
bnds=tuple((0,1) for x in range(num))
for tret in trets:
    cons=({'type':'eq','fun':lambda x: func_stats(x,rets)[0]-tret},
           {'type':'eq','fun':lambda x: 1-np.sum(x)})
    weights=np.random.random(num)
    weights= weights/np.sum(weights)
    res=sco.minimize(lambda x: func_min_vol(x,rets),weights,method='SLSQP',bounds=bnds,constraints=cons)
    tvols.append(res['fun'])
tvols=np.array(tvols) #target volatility

#画有效前沿
plt.figure(figsize=(8,4))
plt.scatter(tvols,trets,c=trets/tvols,marker='x')
plt.grid(True)
plt.xlabel('volatility')
plt.ylabel('return')
plt.colorbar(label='SR')
plt.title('Efficient Frontier')

#市场组合
ind=np.argmin(tvols)
evols=tvols[ind:]
erets=trets[ind:]
tck=sci.splrep(evols,erets)

def func_ef(x,tck):
    return sci.splev(x,tck,der=0)

def func_def(x,tck):
    return sci.splev(x,tck,der=1)

def func_equation(p,tck,rf=0.05):
    eq1=rf-p[0]
    eq2=rf+p[1]*p[2]-func_ef(p[2],tck)
    eq3=p[1]-func_def(p[2],tck)
    return eq1,eq2,eq3

opt=sco.fsolve(lambda s:func_equation(s,tck),[0.05,0.2,0.2],xtol=1e-06)

#画图
plt.figure(figsize=(8,4))
#可行域
plt.scatter(pvols,prets,c=(prets-0.01)/pvols,marker='o')
#有效前沿
plt.plot(evols,erets,'g',lw=2.0)
#切点
plt.plot(opt[2],func_ef(opt[2],tck),'r*')
#市场组合
cx=np.linspace(0.0,max(pvols))
plt.plot(cx,opt[0]+opt[1]*cx,lw=1.5)
#图像背景
plt.grid(True)
plt.axhline(0,color='k',ls='--',lw=2.0)
plt.axvline(0,color='k',ls='--',lw=2.0)
plt.xlabel('volatility')
plt.ylabel('return')

#权重
tpr=func_ef(opt[2],tck)
cons=({'type':'eq','fun':lambda x: func_stats(x,rets)[0]-tpr},
           {'type':'eq','fun':lambda x: 1-np.sum(x)})
bnds=tuple((0,1) for x in range(num))
res=sco.minimize(lambda x:func_min_vol(x,rets),num*[1.0/num],method='SLSQP',bounds=bnds,constraints=cons)
res.x
optwei=[]
optwei.append(res.x.round(3))
