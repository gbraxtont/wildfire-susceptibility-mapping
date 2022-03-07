
import pandas as pd
import numpy as np

def clean():

    data = pd.read_csv('Data Cleaning - Compiled Data.csv',index_col = 0)


    #temp to Nx1 numpy array
    if (data['Temperature (*F)'].dtypes != 'object'):
        temp = data['Temperature (*F)'].to_numpy()
    else:
        temp = data['Temperature (*F)'].tolist()
        temp = [float(x.replace(',','')) for x in temp]
        temp = np.array(temp)
    temp = temp[:,np.newaxis]

    #Max temp to Nx1 numpy array
    if(data['Max Temp (*F)'].dtypes != 'object'):
        maxtemp = data['Max Temp (*F)'].to_numpy()

    else:
        maxtemp = data['Max Temp (*F)'].tolist()
        maxtemp = [float(x.replace(',','')) for x in maxtemp]
        maxtemp = np.array(maxtemp)
    maxtemp = maxtemp[:,np.newaxis]

    #PDSI to Nx1 numpy array
    if(data['PDSI'].dtypes != 'object'):
        pdsi = data['PDSI'].to_numpy()
    else:
        pdsi = data['PDSI'].tolist()
        pdsi = [float(x.replace(',','')) for x in pdsi]
        pdsi = np.array(pdsi)
    pdsi = pdsi[:,np.newaxis]

    #Precipitation to Nx1 numpy array
    if(data['Precipitation (in)'].dtypes != 'object'):
        precipitation = data['Precipitation (in)'].to_numpy()
    else:
        precipitation = data['Precipitation (in)'].tolist()
        precipitation = [float(x.replace(',','')) for x in precipitation]
        precipitation = np.array(precipitation)
    precipitation = precipitation[:,np.newaxis]

    #Forest cover to Nx1 numpy array
    if(data['Forest Cover (%)'].dtypes != 'object'):
        fc = data['Forest Cover (%)'].to_numpy()
    else:
        fc = data['Forest Cover (%)'].tolist()
        fc = [float(x.replace(',','')) for x in fc]
        fc - np.array(fc)
    fc = fc[:,np.newaxis]

    #Population Count to Nx1 numpy array
    if(data['Population Count (ppl)'].dtypes != 'object'):
        pc = data['Population Count (ppl)'].to_numpy()
    else:
        pc = data['Population Count (ppl)'].tolist()
        pc = [int(x.replace(',','')) for x in pc]
        pc = np.array(pc)
    pc = pc[:,np.newaxis]

    #Population density to Nx1 numpy array
    if(data['Population Density (ppl/mi^2)'].dtypes != 'object'):
        popd = data['Population Density (ppl/mi^2)'].to_numpy()
    else:
        popd = data['Population Density (ppl/mi^2)'].tolist()
        popd = [float(x.replace(',','')) for x in popd]
        popd = np.array(popd)
    popd = popd[:,np.newaxis]

    #Avg humidity % to Nx1 numpy array
    if(data['Avg. Humidity (%)'].dtypes != 'object'):
        hum = data['Avg. Humidity (%)'].to_numpy()
    else:
        hum = data['Avg. Humidity (%)'].tolist()
        hum = [float(x.replace(',','')) for x in hum]
        hum = np.array(hum)
    hum = hum[:,np.newaxis]

    #avereage dry to Nx1 numpy array
    if(data['Avg % Dry'].dtypes != 'object'):
        avgDry = data['Avg % Dry'].to_numpy()
    else:
        avgDry = data['Avg % Dry'].tolist()
        avgDry = [float(x.replace(',','')) for x in avgDry]
        avgDry = np.array(avgDry)
    avgDry = avgDry[:,np.newaxis]

    #max dry to Nx1 numpy array
    if(data['Max % Dry'].dtypes != 'object'):
        maxDry = data['Max % Dry'].to_numpy()
    else:
        maxDry = data['Max % Dry'].tolist()
        maxDry = [float(x.replace(',','')) for x in maxDry]
        maxDry = np.array(maxDry)
    maxDry = maxDry[:,np.newaxis]

    #avg wet to Nx1 numpy array
    if(data['Avg % Wet'].dtypes != 'object'):
        avgWet = data['Avg % Wet'].to_numpy()
    else:
        avgWet = data['Avg % Wet'].tolist()
        avgWet = [float(x.replace(',','')) for x in avgWet]
        avgWet = np.array(avgWet)
    avgWet = avgWet[:,np.newaxis]

    #max wet Nx1 numpy array
    if(data['Max % Wet'].dtypes != 'object'):
        maxWet = data['Max % Wet'].to_numpy()
    else:
        maxWet = data['Max % Wet'].tolist()
        maxWet = [float(x.replace(',','')) for x in maxWet]
        maxWet = np.array(maxWet)
    maxWet = maxWet[:,np.newaxis]

    #wildfire count to Nx1 numpy array
    if(data['Wildfire Count (#)'].dtypes != 'object'):
        wc = data['Wildfire Count (#)'].to_numpy()
    else:
        wc = data['Wildfire Count (#)'].tolist()
        wc = [int(x.replace(',','')) for x in wc]
        wc = np.array(wc)
    #wildfire coun are labels
    yData = wc[:,np.newaxis]

    #features
    xData = np.concatenate((temp,maxtemp,pdsi,precipitation,fc,pc,popd,hum,avgDry,maxDry,avgWet,maxWet), axis = 1)

    return yData, xData




