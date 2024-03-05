import sys
sys.path.append('../')

import requests
import numpy as np
from LSBSteg import decode

import tensorflow as tf
import time 
import json

model_path = './Eagle/saved_eagle'

model = tf.saved_model.load(model_path)

infer = model.signatures["serving_default"]


server_ip = '3.70.97.142'
api_base_url = f'http://{server_ip}:5000/eagle/'

team_id='kArHbcl'

def remove_nans(data):
    spectrograms = data

    # Replace 'inf' values with NaN
    spectrograms[np.isinf(spectrograms)] = np.nan

    #print(np.sum(np.isinf(spectrograms)))

    # Iterate over each sample
    for i in range(spectrograms.shape[0]):
        sample = spectrograms[i, :]

        # Create a mask to exclude negative and infinity values
        #valid_mask = (sample >= 0) & ~np.isnan(sample)

        # Replace negative and infinity values with the mean of valid values
        mean_valid = np.mean(sample[sample >= 0])
        #print(mean_valid)
        sample[np.isnan(sample)] = mean_valid
        sample[sample <= 0] = mean_valid

        spectrograms[i, :] = sample

    return spectrograms

def write_to_file(statement):
    with open('eagle_logs5.txt', 'a') as f:
        # Redirect print statements to the file
        print(statement, file=f)

def write_json_testcase(data, cnt):
    
    # Your file path
    file_path = f'./test3.{cnt}.json'

    # Write the JSON object to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


def get_model_prediction(data):
    input_data = tf.constant(data.reshape((1, 1998, 101, 1)), dtype=tf.float32)
    result = infer(input_data)  
    result_value = result['dense_1'].numpy()
    if(result_value[0][0] > 0.5):
        return 'R'
    return 'F'


def get_model_prediction2(data):
    input_data = tf.constant(data.reshape((1, 1998, 101, 1)), dtype=tf.float32)
    result = infer(input_data)

    # Assuming 'dense_1' is not present in the keys, you can adjust accordingly
    result_value = result['dense_1'].numpy()
    return result_value[0][0]


def init_eagle(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    response = requests.post(api_base_url + "start", json={
        "teamId": team_id
    })
    return response.json()['footprint']


def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''
    pass
  
def skip_msg(team_id):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    pass
  
def request_msg(team_id, channel_id):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    pass

def submit_msg(team_id, decoded_msg):
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    pass
  
def end_eagle(team_id):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    pass

def get_channel_intercept(chunk):
    best_score = 0
    best_index = -1
    cleaned = remove_nans(np.array(chunk['1']))
    result = get_model_prediction2(cleaned)
    if(result > best_score):
        best_score = result
        best_index = 1
    
    cleaned = remove_nans(np.array(chunk['2']))
    result = get_model_prediction2(cleaned)
    if(result > best_score):
        best_score = result
        best_index = 2
    
    cleaned = remove_nans(np.array(chunk['3']))
    result = get_model_prediction2(cleaned)
    if(result > best_score):
        best_score = result
        best_index = 3
    
    return best_index if best_score > 0.6 else -1


def submit_eagle_attempt(team_id):
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''
    curr_chunk = init_eagle(team_id)

    while(curr_chunk != "End of message reached"):

        best_channel = get_channel_intercept(curr_chunk)

        if(best_channel == -1):
            response = requests.post(api_base_url + "skip-message", json={
                "teamId": team_id
            })
            if( response.status_code == 400 or response.text == "End of message reached" ):
                break
            curr_chunk = response.json()['nextFootprint']
            
        else:
            response = requests.post(api_base_url + "request-message", json={
                "teamId": team_id,
                "channelId": best_channel
            })
             
            msg = decode(np.array(response.json()['encodedMsg']))
            response = requests.post(api_base_url + "submit-message", json={
                "teamId": team_id,
                "decodedMsg": msg
            })
            if(response.status_code == 400 or response.text == "End of message reached"):
                break
            curr_chunk = response.json()['nextFootprint']

            
    
    response = requests.post(api_base_url + 'end-game', json={
        "teamId": team_id
    })



submit_eagle_attempt(team_id)


#-----------------TESTING----------------------------#
#json_file_path = './test2.0.json'
#with open(json_file_path, 'r') as file:
    #data = json.load(file)

#start = time.time()
#print(get_channel_intercept(data))
#end = time.time()
#print(end - start)

#cleaned = remove_nans(np.array(data['3']))
#print(get_model_prediction2(cleaned))

    

    

    