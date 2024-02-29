import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from matplotlib.dates import (YEARLY, DateFormatter,rrulewrapper, RRuleLocator, drange)
from sklearn.metrics import mean_squared_error
import time
import copy

#use the input data without transformation
scaler = StandardScaler()
mse_threshold = 0.02
TrueAndFalse_tolerance = 20 #unité en minutes
Min_duration = 30 # integer value

#1
def find_error_index(input_data, reconstructed_data, mse_threshold):
    column_names = input_data.columns.tolist()
    
    reconstructed_data = pd.DataFrame(reconstructed_data, columns=column_names)
    
    input_data = scaler.fit_transform(input_data)
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    input_data = pd.DataFrame(input_data, columns=column_names) # this will return data scaled with columns name on it
    
    mse_threshold = mse_threshold  # Set your desired MSE threshold
    Error_Index = []
    for index, (actual_row, predicted_row) in enumerate(zip(input_data.iterrows(), reconstructed_data.iterrows())):
        _, actual_data = actual_row
        _, predicted_data = predicted_row

        mse = mean_squared_error(actual_data, predicted_data)



        if mse > mse_threshold:
            # Take action when the MSE exceeds the threshold
            Error_Index.append(index) # +1 throughing an error when I'm trying to use a different threshold
            #print(f"Row {index + 1} MSE: {mse}")
            #print(f"Action taken for Row {index + 1}: MSE exceeds threshold")

        else:
            # Take a different action or do nothing
            pass
    return Error_Index

#2
def get_real_values(normal_input_data,refined_input_data,reconstructed_data):
    #reconstructed_data = copy.copy(reconstructed_data)
    reconstructed_data = scaler.inverse_transform(reconstructed_data)
    column_names = refined_input_data.columns.tolist()
    
    reconstructed_data = pd.DataFrame(reconstructed_data, columns=column_names)
    
    #reconstructed_data = copy.copy(reconstructed_data)
    reconstructed_data['Timestamp']= normal_input_data['Timestamp']
    
    return reconstructed_data

#3
def build_samples_data(Error_Index,reconstructed_data,feature):#Here we have to create datasets : based on the features selected to plot or to simply select as well as filter the error among normal data
    values_to_store = [reconstructed_data .at[idx, feature] for idx in Error_Index]
    values_to_store1 = [reconstructed_data .at[idx, 'Timestamp'] for idx in Error_Index]
    
    column_names = [feature]
    Data_samples = pd.DataFrame(values_to_store, columns=column_names)
    
    #Data_samples.index =values_to_store1
    Data_samples['Timestamp'] = values_to_store1
    Data_samples['Timestamp'] = pd.to_datetime(Data_samples["Timestamp"])
    Data_samples.set_index('Timestamp', inplace=True)
    
    return Data_samples

#4
def visualization(Data_samples, TrueAndFalse_tolerance, Min_duration,feature):
    # Initialize variables
    slices = []
    current_slice = []

    # Iterate through the index (timestamps)
    for i, index in enumerate(Data_samples.index):
        if i == 0:
            current_slice.append(index)
        elif (index - Data_samples.index[i - 1]).total_seconds() / 60 <= TrueAndFalse_tolerance:
            current_slice.append(index)
            #print('The condition was true')
        else:
            slices.append(current_slice.copy())
            current_slice = [index]

    # Append the last slice
    if current_slice:
        slices.append(current_slice)

    # Plot the values in each slice with timestamps on the x-axis

    #B=my_Gen_Bear.Gen_Bear_Temp_Avg.values.reshape(1, -1)
    ##my_Gen_Bear['Gen_Bear_Temp_Avg']=scaler.inverse_transform(B)
    #i=0
    for i, slice_indices in enumerate(slices):
        slice_df = Data_samples.loc[slice_indices]

        if len(slice_indices) >= Min_duration: # environ 2 heures
            # Only plot slices with more than 9 rows
            print(f"Slice {i + 1} (Rows: {len(slice_indices)}):\n{slice_df}")
            print()
            plt.figure(figsize=(8, 4))  # Adjust figure size as needed
            plt.plot(slice_df.index, slice_df[feature], marker='o', linestyle='-', label=f'Slice {i + 1}')
            plt.title(f'Values in Slice {i + 1}')
            plt.xlabel('Timestamp')
            plt.ylabel(feature)
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.legend()
            plt.grid(True)
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
            plt.show()

#5
def get_slices(Data_samples) :
    slices = []
    current_slice = []
    for i, index in enumerate(Data_samples.index):
        if i == 0:
            current_slice.append(index)
        elif (index - Data_samples.index[i - 1]).total_seconds() / 60 <= TrueAndFalse_tolerance:
            current_slice.append(index)
            #print('The condition was true')
        else:
            slices.append(current_slice.copy())
            current_slice = [index]

    # Append the last slice
    if current_slice:
        slices.append(current_slice)
    return slices
def filter_slices(Data_samples,TrueAndFalse_tolerance, Min_duration) :
        error = []
        # Initialize variables
        slices = []
        current_slice = []

        # Iterate through the index (timestamps)
        for i, index in enumerate(Data_samples.index):
            if i == 0:
                current_slice.append(index)
            elif (index - Data_samples.index[i - 1]).total_seconds() / 60 <= TrueAndFalse_tolerance:
                current_slice.append(index)
                #print('The condition was true')
            else:
                slices.append(current_slice.copy())
                current_slice = [index]

        # Append the last slice
        if current_slice:
            slices.append(current_slice)
        for i, slice_indices in enumerate(slices):
            
            slice_df = Data_samples.loc[slice_indices]

            if len(slice_indices) >= Min_duration: # environ 2 heures
                error.append(slice_df)
                #print("The script was triggered")
                # Only plot slices with more than 9 rows
                print(f"Slice {i + 1} (Rows: {len(slice_indices)}):\n{slice_df['Gen_Phase1_Temp_Avg']}")
                #print()
                plt.figure(figsize=(8, 4))  # Adjust figure size as needed
                feature = 'Gen_Phase1_Temp_Avg' #  Gear_Bear_Temp_Avg
                plt.plot(slice_df.index, slice_df[feature], marker='o', linestyle='-', label=f'Slice {i + 1}') 
                plt.title(f'Values in Slice {i + 1}')
                plt.xlabel('Timestamp')
                plt.ylabel(feature)
                plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
                plt.legend()
                plt.grid(True)
                plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
                plt.show()
                
        return(error)
#6eme fonction        
def make_classification(Data_samples,slice_indices, input_data, TrueAndFalse_tolerance, Min_duration,feature):
    # Initialize variables
    list_data_list = []
    #data_listA = []
    data_listAplus = []
    data_listAminus = []
    #data_listB = []
    data_listBplus = [] 
    data_listBminus = []
    slice_df1 = []
    slice_df2 = []
    slice_df3 = []
    slice_df4 = []
    slice_scaled_df1 = []
    slice_scaled_df2 = []
    slice_scaled_df3 = []
    slice_scaled_df4 = []
    scaled_data_list = []
    #data_listC = []
    journal_errors = ['2016-04-30 00:00:00','2016-07-10 04:00:00', '2016-08-23 00:00:00', '2017-06-17 00:00:00', '2017-08-20 00:00:00'] #this is based on visualization function
    #journal_errors = pd.to_datetime(journal_errors)
    journal_errors_df = pd.DataFrame({'Timestamp': pd.to_datetime(journal_errors)})
    # Iterate through the index (timestamps)
    input_data['Timestamp'] = pd.to_datetime(input_data['Timestamp'])
    input_data.set_index('Timestamp', inplace=True)
    df=copy.copy(input_data)
    

    for i, slice_indice in enumerate(slice_indices):  
        #slice_df = Data_samples.loc[slice_indices]
        start_date = slice_indice[0]
        end_date = slice_indice[-1]

        if len(slice_indice) >= Min_duration: # environ 2 heures
            
            timestamp_exists_in_interval = any((journal_errors_df['Timestamp'] >= start_date) & (journal_errors_df['Timestamp'] <= end_date))
            if (timestamp_exists_in_interval == True) :
                df1 = input_data.loc[start_date:end_date,:] # dataframe
                data_listAplus.append(df1)
            else:
                df1 = input_data.loc[start_date:end_date,:]
                data_listAminus.append(df1)
            """
            for i,journal_error in enumerate(journal_errors):
                df1 = []
                test_timestamp = journal_errors[i]
                #print(test_timestamp)
                #print(start_date)
                
                                                                                                                       
                """
            """
                is_within_interval = (test_timestamp >= start_date) & (test_timestamp <= end_date)
                #is_within_interval = Data_samples.loc[start_date:end_date,[feature]]
                """
                
                    #Here we have the known error
                    
                    
                
                    #Here we have the unknown error
                    
                    
                    
                #df= pd.concat([df1,df2])      
    
        elif   (4 <= len(slice_indice)) & (len(slice_indice) < Min_duration ):
            start_date = slice_indice[0]
            end_date = slice_indice[-1]
            #journal_errors = ['2016-04-30 00:00:00','2016-07-10 04:00:00', '2016-08-23 00:00:00', '2017-06-17 00:00:00', '2017-08-20 00:00:00']
            #journal_errors = pd.to_datetime(journal_errors)
            timestamp_exists_in_interval = any((journal_errors_df['Timestamp'] >= start_date) & (journal_errors_df['Timestamp'] <= end_date))
            if (timestamp_exists_in_interval == True) :
                df1 = input_data.loc[start_date:end_date,:]
                data_listBplus.append(df1)
            else:
                df1 = input_data.loc[start_date:end_date,:]
                data_listBminus.append(df1)
            """
            for i,journal_error in enumerate(journal_errors):
                df1 = []
                test_timestamp =journal_errors[i]
                #is_within_interval = Data_samples.loc[start_date:end_date,[feature]]
                #is_within_interval = (test_timestamp >= start_date) & (test_timestamp <= end_date)
            """
        
            
    """
    #Time to separate normal data, know error, unknown error
    df=copy.copy(input_data)
    #print(df)
    #df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    #df.set_index('Timestamp', inplace=True)
    #print(df)
    df = df.reset_index()
    """
    total_rows = 0
    if len(data_listAplus) == 0:
        print("The data_listAplus list is empty")
    else:
        # Anomalie A+ above  Min_duration and registered in log
        for index1, known_error1 in enumerate(data_listAplus):
            #start_timestamp = known_error[0]
            ##end_timestamp = known_error[-1]
            
            #slice_df1 = df2[df2.isin(df1)].dropna()
            #slice_df1 = data_listAplus
            df1 = df[df.isin(known_error1)].dropna()
            df1 = df1.reset_index()
            df1 = df1.drop(columns=['Timestamp'])
            df1 = scaler.fit_transform(df1)
            slice_scaled_df1.append(df1)
            df = df[~df.isin(known_error1)].dropna()
            
            
            #print(df1)
            total_rows = total_rows + len(known_error1)
            #print("df Actual total rows :", len(df))
            
            #slice_df1 = df.loc[start_timestamp:end_timestamp]
            #slice_df = df[(df['Timestamp'] >= interval_to_drop[0]) & (df['Timestamp'] <= interval_to_drop[1])]
            #df = df[~((df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp))]
    print('The total number dropped in data_listAplus : \n',total_rows)        
    total_rows = 0    
    
    if len(data_listAminus) == 0:
        print("The data_listAminus list is empty")
    else:
        #Anomalie A- above Min_duration but not registered in log
        for index2 ,unknown_error in enumerate(data_listAminus):
            #start_timestamp = unknown_error[0]
            #end_timestamp = unknown_error[-1]
            #print(unknown_error)
            #slice_df2 = df.loc[unknown_error]
            #slice_df2 = data_listAminus
            #print("Number of rows going to be dropped",len(unknown_error))
            df1 = df[df.isin(unknown_error)].dropna()
            df1 = df1.reset_index()
            df1 = df1.drop(columns=['Timestamp'])
            df1 = scaler.fit_transform(df1)
            slice_scaled_df2.append(df1)
            df = df[~df.isin(unknown_error)].dropna()
            #print("DataFrame number :",index, "dropped from the normal data")
            total_rows = total_rows + len(unknown_error)
            #print("df Actual total rows :", len(df))
            #slice_df2 = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
            #df = df[~((df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp))]
    print('The total number dropped in data_listAminus : \n',total_rows)
    total_rows = 0
    if len(data_listBminus) == 0:
        print("The list (data_listBminus is empty")
    else:
        # Anomalie B- in between certain value not registered in log
        for index3, unknown_error1 in enumerate(data_listBminus):
            #print(index3)
            #print(unknown_error1)
            #start_timestamp = known_error[0]
            ##end_timestamp = known_error[-1]
            #print(start_timestamp)
            ##slice_df3 = data_listBminus
            #print(slice_df3)
            #slice_df3 = df.loc[known_error]
            #slice_df3 = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
            #df = df[~((df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp))]
            df1 = df[df.isin(unknown_error1)].dropna()
            df1 = df1.reset_index()
            df1 = df1.drop(columns=['Timestamp'])
            df1 = scaler.fit_transform(df1)
            slice_scaled_df3.append(df1)
            #df = df[~df.isin(unknown_error1)].dropna()
            total_rows = total_rows + len(unknown_error1)
            #print("df Actual total rows :", len(df))
            
    print('The total number dropped in data_listBminus : \n',total_rows)
    total_rows = 0
    if len(data_listBplus) == 0:
        print("The list (data_listBplus is empty")
    else:
        for index4, known_error2 in enumerate(data_listBplus):
            ##print(index4)
            #print(known_error2)
            #start_timestamp = known_error[0]
            ##end_timestamp = known_error[-1]
            #print(start_timestamp)
            ##slice_df3 = data_listBminus
            #print(slice_df3)
            #slice_df3 = df.loc[known_error]
            #slice_df3 = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
            #df = df[~((df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp))]
            df1 = df[df.isin(known_error2)].dropna()
            df1 = df1.reset_index()
            df1 = df1.drop(columns=['Timestamp'])
            df1 = scaler.fit_transform(df1)
            slice_scaled_df4.append(df1)
            df = df[~df.isin(known_error2)].dropna()
            total_rows = total_rows + len(known_error2)
            #print("df Actual total rows :", len(df))
    print('The total number dropped in data_listBplus : \n',total_rows)        
    ####df=copy.copy(input_data)
    ###df = df[~df.isin(data_listAplus)].dropna()
    ##df = df[~df.isin(data_listAminus)].dropna()
    #df = df[~df.isin(data_listBminus)].dropna()
    slice_df1 = copy.copy(data_listAplus)
    slice_df2 =copy.copy(data_listAminus)
    slice_df3 = copy.copy(data_listBminus) #data_listBplus
    slice_df4 = copy.copy(data_listBplus)
    
    list_data_list.append(df)
    list_data_list.append(slice_df1)
    list_data_list.append(slice_df2)
    list_data_list.append(slice_df3)
    list_data_list.append(slice_df4)
    #slice_scaled_df1 = copy.copy(data_listAplus)
    ##slice_scaled_df2 =copy.copy(data_listAminus)
    ###slice_scaled_df3 = copy.copy(data_listBminus)
    
    
    #print(slice_df2)
    #list_data_list.append(df, slice_df1, slice_df2, slice_df3
    df = df.reset_index()
    scaled_data_list.append(scaler.fit_transform(df.drop(columns=['Timestamp'])))
    scaled_data_list.append(slice_scaled_df1)
    scaled_data_list.append(slice_scaled_df2)
    scaled_data_list.append(slice_scaled_df3)
    scaled_data_list.append(slice_scaled_df4)
    return list_data_list, scaled_data_list

def make_classification2(error_data) :
    error_array = []
    for error in  error_data:
        error_sample = error
        error_sample = error_sample.drop(columns=["MSE_below_threshold"])
        error_sample = scaler.fit_transform(error_sample)
        error_array.append(error_sample)
    return (error_array)

def show_local_explaination(list_to_explain,
                      original_data_point,
                      invert_transform,
                      data_train,
                      impact_pourcentage = 50 ,
                      plot=False,
                      sort=False,
                      show_pourcentage=False):
    exp_data = pd.DataFrame(list_to_explain)
    exp_data = exp_data.T
    exp_data.columns = data_train.columns
    col_to_scale = list(data_train.columns)
    #print(col_to_scale,len(col_to_scale))
    if invert_transform :
        scaler = StandardScaler() #MinMaxScaler()
        scaler.fit(data_train[col_to_scale])
        exp_inverted = pd.DataFrame()
        org_inverted = pd.DataFrame()
        exp_inverted[col_to_scale] = scaler.inverse_transform(exp_data[col_to_scale])
        org_inverted[col_to_scale] = scaler.inverse_transform(original_data_point[col_to_scale])

        result = pd.concat([org_inverted,exp_inverted]).T
        
    else :
        result = pd.concat([original_data_point[col_to_scale],exp_data[col_to_scale]]).T
        
    result.columns = ['real_data_point','optimal_values']
    #print(result.head(20))
    #Here is the most important part of the code
    #------------------------------------------------------------------------------------------------------------------
    result['impact'] = result.apply(lambda row: np.mean((row[result.columns[0]] - row[result.columns[1]])**2), axis=1)
    result['Polarity'] = result.apply(lambda row: row[result.columns[0]] * row[result.columns[1]], axis=1)
    #result['Polarity'] = result[result.columns].apply(lambda x  : x[0] * x[1] , axis=1)
    #---------------------------------------------------------------------------------------------------------------------
    #result['impact']=np.mean((df_anomalie.Hyd_Oil_Temp_Avg.values[85485] - reconstructed_data.Hyd_Oil_Temp_Avg.values[85485])**2)
    #result['impact'] = result.apply(lambda row: (row['real_data_point'] - row['optimal_values']) / (abs(row['real_data_point']) + abs(row['optimal_values'])), axis=1)
    #max_abs_diff = result[['real_data_point', 'optimal_values']].abs().max().max()
    #result['impact'] = result.apply(lambda row: (row['real_data_point'] - row['optimal_values']) / max_abs_diff, axis=1)
    #result = result.fillna(0) 
    #full_deviation = result.impact.abs().sum()
    #result['pourcentage_impact'] = result['impact'].apply(lambda x : (abs(x)/result.impact.abs().sum())*100 )
    result['pourcentage_impact'] = result['impact'].apply(lambda x : x*1)
    result_copy = result.sort_values(by=['pourcentage_impact'],ascending=False)
    result_copy['pourc_cum'] = result_copy.pourcentage_impact.cumsum()
    #print(result_copy)
    print(result)
    impacting_variable_index = list(np.where(result_copy['pourc_cum'] > impact_pourcentage))[0][0]
    impacting_variable_line = (len(result) - impacting_variable_index ) - 1.4
    
    #Everything will start here since we set invert_transform to false
    
    if sort :
            result.sort_values(by=['pourcentage_impact'],ascending=False , inplace=True)
            result = result[0:10]
            impacting_variable_line = (len(result) - impacting_variable_index ) - 1.4
            result.sort_values(by=['pourcentage_impact'],ascending=True , inplace=True)
    if plot :
        limit = np.max(np.abs(result.impact.values)) 
        limit = limit + 0.1*limit
        fig, ax = plt.subplots(figsize=(10,6))
        ax.set_xlim([0,limit])
        #ax.set_xlim([-limit,limit])
        ax.barh(result.index, result['impact'], color='blue')
        container = ax.containers[0]
        if show_pourcentage :
            ax.bar_label(container, labels=[f'{x:,.2f} %' for x in list(result.pourcentage_impact)])
        ax.plot([-2, 2], [impacting_variable_line, impacting_variable_line], "k--",color='green')
        if show_pourcentage :
            ax.set_xlabel("pourcentage of deviation")
        else :
            ax.set_xlabel("deviation")
        ax.set_ylabel("features")
    else :
        return result
    
    return result