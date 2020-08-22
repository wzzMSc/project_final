# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
# %%
df = pd.read_csv('data_pre/data_pre.csv',header=0,parse_dates=['ISODate'],index_col='ISODate')

prediction_targets = ['BEAMS','BEAMT','BEAMT2']

resample_method = 'H'
df = df.resample(resample_method).mean()
df.drop(labels=['FAA'],axis=1, inplace=True)



# %%
for x in df.columns:
    if not os.path.exists("vis/raw_reg/"+x+'/'):
        os.makedirs("vis/raw_reg/"+x+'/')

for y in df.columns:
    plt.figure(figsize=(150,10))
    plt.plot(df.index,df[y])
    plt.xlabel('Date')
    plt.ylabel(y)
    plt.savefig("vis/raw_reg/"+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()



# %%
for x in prediction_targets:
    if not os.path.exists("vis/raw_cla/"+x+'/'):
        os.makedirs("vis/raw_cla/"+x+'/')


for x in prediction_targets:
    thres = df[x].max()*0.5
    df[x][df[x]<thres] = 0
    df[x][df[x]>=thres] = 1

for y in prediction_targets:
    plt.figure(figsize=(150,10))
    plt.plot(df.index,df[y])
    plt.xlabel('Date')
    plt.ylabel(y)
    plt.savefig("vis/raw_cla/"+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()


# %%
if not os.path.exists("vis/corr/"):
    os.makedirs("vis/corr/")

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),vmin=-1, vmax=1, annot=True).set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
plt.savefig('vis/corr/corr.png',dpi=300)
plt.cla()
plt.close()


# %%
for x in 'HOUR,DAY,WEEKDAY,MONTH,YEAR'.split(','):
    if not os.path.exists("vis/avg/"+x+'/'):
        os.makedirs("vis/avg/"+x+'/')

def draw_plot(x,y,xlabel,ylabel,title,filename):
    plt.figure(figsize=(20,20))
    sns_temp = sns.pointplot(data=df,x=x,y=y).set(xlabel=xlabel,ylabel=ylabel,title=title)
    plt.savefig('vis/avg/'+x+'/'+filename+'.png',dpi=130)
    plt.cla()
    plt.close()

for y in 'BEAMT,BEAMT2,BEAMS,BEAME1,MTEMP,MUONKICKER,HTEMP,TS1_TOTAL_YEST,TS1_TOTAL,REPR,REPR2,TS2_TOTAL,TS2_TOTAL_YEST'.split(','):
    for x in 'HOUR,DAY,WEEKDAY,MONTH,YEAR'.split(','):
        draw_plot(x,y,x,y,x+'-'+y,x+'_'+y)
# %%
for x in df.columns:
    if not os.path.exists("vis/scatter/"+x+'/'):
        os.makedirs("vis/scatter/"+x+'/')

for x in df.columns:
    for y in df.columns:
        plt.scatter(df[x],df[y],s=0.5)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(x+"-"+y)
        plt.savefig("vis/scatter/"+x+'/'+x+"-"+y+".png")
        plt.cla()
        plt.close()



# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('data_pre/data_pre.csv',header=0,parse_dates=['ISODate'],index_col='ISODate')

resample_method = 'H'
df = df.resample(resample_method).mean()
df.drop(labels=['FAA'],axis=1, inplace=True)
df = df.dropna(axis=0)

for x in prediction_targets:
    if not os.path.exists("vis/dr_pca/"+x+'/'):
        os.makedirs("vis/dr_pca/"+x+'/')


for x in prediction_targets:
    thres = df[x].max()*0.5
    df[x][df[x]<thres] = 0
    df[x][df[x]>=thres] = 1


raw_features = "MTEMP,MUONKICKER,HTEMP,TS1_TOTAL_YEST,TS1_TOTAL,REPR,REPR2,TS2_TOTAL,TS2_TOTAL_YEST".split(',')
# raw_features = "MTEMP,MUONKICKER,HTEMP,TS1_TOTAL_YEST,TS1_TOTAL,TS2_TOTAL,TS2_TOTAL_YEST".split(',')
# raw_features = "YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,WEEKDAY,UPTIME".split(',')

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[raw_features].values)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


for i in range(1,pca_result.shape[1]+1):
    df['PCA'+str(i)] = pca_result[:,i-1]


pca_l = ['PCA1','PCA2','PCA3']
for target in prediction_targets:
    for x in pca_l:
        for y in pca_l:
            plt.figure(figsize=(15,15))
            sns.scatterplot(x,y,data=df,hue=target)
            plt.savefig("vis/dr_pca/"+target+'/'+x+"-"+y+".png",dpi=200)
            plt.cla()
            plt.close()


for target in prediction_targets:
    ax = Axes3D(plt.figure(figsize=(15,15)))
    ax.scatter3D(df['PCA1'],df['PCA2'],df['PCA3'],c=df[target])
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    plt.savefig("vis/dr_pca/"+target+'/'+target+"_3d.png",dpi=200)
    plt.cla()
    plt.close()

##################################################################
#######                t-SNE
##################################################################
# %%
for x in prediction_targets:
    if not os.path.exists("vis/dr_tsne/"+x+'/'):
        os.makedirs("vis/dr_tsne/"+x+'/')

tsne = TSNE(n_components=3,verbose=1,n_iter=1000)
tsne_results = tsne.fit_transform(df[raw_features].values)

for i in range(1,tsne_results.shape[1]+1):
    df['tSNE'+str(i)] = tsne_results[:,i-1]

tsne_l = ['tSNE1','tSNE2','tSNE3']
for target in prediction_targets:
    for x in tsne_l:
        for y in tsne_l:
            sns.scatterplot(x,y,data=df,hue=target)
            plt.savefig("vis/dr_tsne/"+target+'/'+x+"-"+y+".png")
            plt.cla()
            plt.close()

for target in prediction_targets:
    ax = Axes3D(plt.figure(figsize=(15,15)))
    ax.scatter3D(df['tSNE1'],df['tSNE2'],df['tSNE3'],c=df[target])
    ax.set_xlabel('tSNE1')
    ax.set_ylabel('tSNE2')
    ax.set_zlabel('tSNE3')
    plt.savefig("vis/dr_tsne/"+target+'/'+target+"_3d.png",dpi=130)
    plt.cla()
    plt.close()
# %%
###########DOWNTIME
import matplotlib

downtime_attrs = ['StartDate','SecondsSinceUp','EndDate','LastedSeconds']

for x in prediction_targets:
    if not os.path.exists("vis/downtime/"+x+'/'):
        os.makedirs("vis/downtime/"+x+'/')

for x in prediction_targets:
    if not os.path.exists("vis/downtime_seq/"+x+'/'):
        os.makedirs("vis/downtime_seq/"+x+'/')

for x in prediction_targets:
    if not os.path.exists("vis/downtime_seconds_since_up/"+x+'/'):
        os.makedirs("vis/downtime_seconds_since_up/"+x+'/')

for target in prediction_targets:
    df_target = pd.read_csv('data_pre/Downtime_'+target+'.csv',header=0)
    df_target['StartDate'] = pd.to_datetime(df_target['StartDate'])
    df_target.set_index('StartDate')
    plt.figure(figsize=(150,10))
    plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d", tz=None))
    plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=None, interval=20, tz=None))
    plt.plot(df_target['StartDate'],df_target['LastedSeconds'])
    plt.xlabel('StartDate')
    plt.ylabel('LastedSeconds')
    plt.savefig("vis/downtime/"+target+'/'+target+'.png',dpi=130)
    plt.cla()
    plt.close()

for target in prediction_targets:
    df_target = pd.read_csv('data_pre/Downtime_'+target+'.csv',header=0)
    df_target['i'] = [x for x in range(1,len(df_target)+1)]
    plt.figure(figsize=(150,10))
    plt.plot(df_target['i'],df_target['LastedSeconds'])
    plt.xlabel('Sequential order')
    plt.ylabel('LastedSeconds')
    plt.savefig("vis/downtime_seq/"+target+'/'+target+'.png',dpi=130)
    plt.cla()
    plt.close()
    df_target.to_csv('data_pre/Downtime_seq_'+target+'.csv')


for target in prediction_targets:
    df_target = pd.read_csv('data_pre/Downtime_seq_'+target+'.csv',header=0)
    max = df_target['SecondsSinceUp'].max()
    bins = [max*x*0.01 for x in range(0,101)]
    labels = [str(int(bins[i]))+'-'+str(int(bins[i+1])) for i in range(0,len(bins)-1)]
    df_target['Bin'] = pd.cut(df_target['SecondsSinceUp'],bins=bins,labels=labels)
    labels_count = [list(df_target['Bin'].values).count(labels[i]) for i in range(len(labels))]
    plt.figure(figsize=(150,10))
    plt.bar(labels,labels_count)
    plt.xlabel('Seconds since up')
    plt.ylabel('Number of downtimes')
    plt.savefig("vis/downtime_seconds_since_up/"+target+'/'+target+'.png',dpi=130)
    plt.cla()
    plt.close()

# %%
