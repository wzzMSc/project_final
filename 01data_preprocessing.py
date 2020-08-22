# %%
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from dateutil import parser

# %%
path_data = "data/isislog"

f_list = [join(path_data,f) for f in listdir(path_data) if isfile(join(path_data,f))]
# %%
attrs_dict = dict()
for f in f_list:
    with open(f,'r') as file:
        try:
            lines = file.readlines()
            attrs = lines[0].split()
        except BaseException:
            continue
        for attr in attrs:
            if attr in attrs_dict:
                attrs_dict[attr] += 1
            else:
                attrs_dict[attr] = 1
columns = list(attrs_dict.keys())

# %%
if not os.path.exists("data_pre/"):
    os.makedirs("data_pre/")

# %%
with open('data_pre/data.csv','a') as data_file:
    data_file.writelines(','.join(columns)+'\n')

# %%
for f in f_list:
    data = [ [] for i in range(len(attrs_dict)) ]
    with open(f,'r') as file:
        try:
            lines = file.readlines()
            attrs = lines[0].split()
        except BaseException:
            print('Decode error')

        for line in lines[1:]:
            for column in columns:
                if column not in attrs:
                    data[columns.index(column)].append('NaN')
                else:
                    try:
                        if int(line.split()[0])<1048763883:
                            raise BaseException
                        data[columns.index(column)].append(line.split()[attrs.index(column)])
                    except BaseException:
                        print('Invalid data')
        data_output = list()
        for i in range(len(data[0])):
            data_row = list()
            for data_col in data:
                try:
                    data_row.append(data_col[i])
                except BaseException:
                    continue
            data_output.append(','.join(data_row)+'\n')

        with open('data_pre/data.csv','a') as data_file:
            data_file.writelines(data_output)
# %%
df = pd.read_csv('data_pre/data.csv',header=0)
# %%
df = df[df['SEC_1970']>=1460415602]
# %%
df['ISODate'] = pd.to_datetime(df['DATE']+' '+df['TIME'])

# %%
df = df.set_index('ISODate')



# %%
cycles_Dates = [
 ['12 Apr 2016','20 May 2016'],
 ['13 Sep 2016','28 Oct 2016'],
 ['15 Nov 2016','16 Dec 2016'],
 ['14 Feb 2017','30 Mar 2017'],
 ['02 May 2017','01 Jun 2017'],
 ['19 Sep 2017','27 Oct 2017'],
 ['06 Feb 2018','25 Mar 2018'],
 ['17 Apr 2018','18 May 2018'],
 ['11 Sep 2018','26 Oct 2018'],
 ['13 Nov 2018','18 Dec 2018'],
 ['05 Feb 2019','29 Mar 2019'],
 ['04 Jun 2019','19 Jul 2019'],
 ['10 Sep 2019','25 Oct 2019'],
 ['12 Nov 2019','20 Dec 2019']
]
ISO_cycles_Dates = list()
for cycle in cycles_Dates:
    start = parser.parse(cycle[0]+" 00:00:00")
    end = parser.parse(cycle[1]+" 00:00:00")
    ISO_cycles_Dates.append([start,end])
# %%
df['YEAR'] = df.index.year
df['MONTH'] = df.index.month
df['DAY'] = df.index.day
df['HOUR'] = df.index.hour
df['MINUTE'] = df.index.minute
df['SECOND'] = df.index.second
df['WEEKDAY'] = df.index.dayofweek


df_list = list()
for cycle in ISO_cycles_Dates:
    df_temp = df[(df.index>=cycle[0]) & (df.index<cycle[1])]
    df_temp['UPTIME'] = df_temp['SEC_1970'] - df_temp['SEC_1970'][0]
    df_list.append(df_temp)
df_concat = pd.concat(df_list)

df_concat.to_csv('data_pre/data_pre.csv')
# %%
df = pd.read_csv('data_pre/data_pre.csv',header=0,parse_dates=['ISODate'])
# %%
prediction_targets = ['BEAMS','BEAMT','BEAMT2']
downtime_attrs = ['StartDate','SecondsSinceUp','EndDate','LastedSeconds']
# %%
for target in prediction_targets:
    df_concat = []
    for cycle in cycles_Dates:
        target_threshold = df[(df['ISODate']>=cycle[0]) & (df['ISODate']<cycle[1])][target].max()*0.8
        first_up=0
        is_down = 0
        up_start = 0
        down_start = 0
        list_df,l=[],[]
        df_list = df[(df['ISODate']>=cycle[0]) & (df['ISODate']<cycle[1])].to_dict('records')
        for doc in df_list:
            if first_up == 0:
                if int(doc[target])>=target_threshold:
                    first_up=1
                    up_start=int(doc['SEC_1970'])
            else:
                if is_down==0:
                    if int(doc[target])<target_threshold:
                        is_down=1
                        down_start=int(doc['SEC_1970'])
                        l.append(doc['ISODate'])
                        l.append(down_start-up_start)
                else:
                    if int(doc[target])>=target_threshold:
                        is_down=0
                        l.append(doc['ISODate'])
                        l.append(int(doc['SEC_1970']) - down_start)
            
            if len(l)==len(downtime_attrs):
                list_df.append(l)
                l=[]
        df_down = pd.DataFrame(list_df,columns=downtime_attrs)
        if len(df_down)!=0:
            df_concat.append(df_down)
    df_down = pd.concat(df_concat)
    df_down.to_csv('data_pre/Downtime_'+target+'.csv',index=False)
# %%
path_data = "data/mcr"

f_list = [join(path_data,f) for f in listdir(path_data) if isfile(join(path_data,f))]

mcr_list = list()
for f in f_list:
    with open(f,'r',encoding='utf-8') as file:
        date = ''
        msg = list()
        for line in file:
            if line[0]!=' ':
                if len(msg)!=0:
                    mcr_list.append([date,msg])
                    msg = list()
                    date=parser.parse(line)
                else:
                    date=parser.parse(line)
            else:
                msg.append(line)

mcr_df = pd.DataFrame(mcr_list,columns=['date','msg'])
mcr_df.sort_values(by='date',ascending=True,inplace=True)
mcr_df.to_csv('data_pre/mcr.csv',index=False)
# %%
