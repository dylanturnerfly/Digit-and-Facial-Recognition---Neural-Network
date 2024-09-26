import os
import random
import time
from numpy import mean, std
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

##############################
#### ADJUSTABLE VARIABLES ####
##############################

directory = './network_weights/stored_weights_digit' #path to weights directory

imageHeight = 28 #do not change
imageWidth = 28 #do not change

#NN variables
hidden_layer_height = 10
input_hidden_weights = np.random.uniform(-0.5 , 0.5, (hidden_layer_height , 784))
hidden_output_weights = np.random.uniform(-0.5 , 0.5, (10 , hidden_layer_height))
input_hidden_bias = np.zeros((hidden_layer_height,1))
hidden_output_bias = np.zeros((10 , 1))

learnRate = 0.01
correct = 0

Trials = 5
epochs = 200
round_digits = 4

show_indidivual_results = False #debugging info (WARNING: spam)
show_group_results = False #debugging info 

##################################
#### NEURAL NETWORK FUNCTIONS ####
##################################

#Function to read all faces from a file into an array of faces
def read_faces_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    faces = []  #faces array to return
    current_face = []  #temporary current face data
    line_count = 0 
    for line in lines:
        #convert line to list of 1s and 0s
        row = []
        for char in line:
            if char == '#' or char == '+':
                row.append(1)
            if char == ' ':
                row.append(0)    
        current_face.append(row)
        
        line_count += 1
        if len(row) != 28:
            print(len(row))

        #check if we have read 70 lines
        if line_count == imageHeight:
            faces.append(np.reshape(current_face, (imageHeight * imageWidth, 1)))
            current_face = []  
            line_count = 0     

    #add the last face if it wasn't added yet (# of lines was not multiple of 70)
    return faces   

#Function to read labels into a 1D array
def read_labels(filename):
    labels = []
    with open(filename, 'r') as file:
            content = file.read()
    labels = [int(char) for char in content if char in '0123456789']
    return labels 
    
#Function to print a face in the console
def print_face(faces, index):
    if index < len(faces):
        face = np.reshape(faces[index],(28,28))
        for row in face:
            print(''.join('#' if pixel == 1 else ' ' for pixel in row))
    else:
        print("Index out of range.")

def convertMatrix(num):
    arr = np.zeros((10, 1))  # Create a 10x1 array of zeros
    arr[num, 0] = 1     # Set '1' at the specified position
    return arr

def resetWeights():
    global input_hidden_weights
    global hidden_output_weights
    global input_hidden_bias
    global hidden_output_bias
    input_hidden_weights = np.random.uniform(-0.5 , 0.5, (hidden_layer_height , 784))
    hidden_output_weights = np.random.uniform(-0.5 , 0.5, (10 , hidden_layer_height))
    input_hidden_bias = np.zeros((hidden_layer_height,1))
    hidden_output_bias = np.zeros((10 , 1))

def trainImageArray(faces, labels):
    global input_hidden_weights
    global hidden_output_weights
    global input_hidden_bias
    global hidden_output_bias
    for face, label in zip(faces, labels):
      # label.shape += (1,)
        #forward propagtion from input to hidden
        hiddenBasic = input_hidden_weights @ face + input_hidden_bias
        hidden = 1/(1+np.exp(-hiddenBasic)) #normalize
        #forward propagation from hidden to input
        outputBasic = hidden_output_weights @ hidden + hidden_output_bias
        output = 1/(1+np.exp(-outputBasic)) #normalize
        #backpropagation output to hidden
        deltaOutput = output - convertMatrix(label)
        hidden_output_weights += -learnRate * deltaOutput @ np.transpose(hidden)
        hidden_output_bias += -learnRate *deltaOutput
        #backpropagation hidden to output
        delta_h = np.transpose(hidden_output_weights) @ deltaOutput *(hidden*(1-hidden))
        input_hidden_weights += -learnRate * delta_h @ np.transpose(face)
        input_hidden_bias += -learnRate * delta_h

def testImageArray(faces, labels):
    global correct  
    correct = 0
    for face, label in zip(faces, labels):
       # label.shape += (1,)
         # forward propagtion from input to hidden
        hiddenBasic = input_hidden_weights @ face + input_hidden_bias
        hidden = 1/(1+np.exp(-hiddenBasic)) #normalize
        #forward propagation from hidden to input
        outputBasic = hidden_output_weights @ hidden + hidden_output_bias
        output = 1/(1+np.exp(-outputBasic)) #normalize
        #checks if correct
        if np.argmax(output) == label:
            correct += 1
    # print(str(correct / len(faces)) + " accuracy")
    return correct / len(faces)

def testOneImage(face):
    global correct  
    correct = 0
    hiddenBasic = input_hidden_weights @ face + input_hidden_bias
    hidden = 1/(1+np.exp(-hiddenBasic)) #normalize
    #forward propagation from hidden to input
    outputBasic = hidden_output_weights @ hidden + hidden_output_bias
    output = 1/(1+np.exp(-outputBasic)) #normalize
    #checks if correct
    return np.argmax(output)

def sample_percentage_two_arrays(faces, labels, percentage):
    if len(faces) != len(labels):
        raise ValueError("both arrays must be of the same size.")
    
    #number of items to sample
    sample_size = int(len(faces) * percentage / 100)
    
    #indicies to select from both arrays
    sampled_indices = random.sample(range(len(faces)), sample_size)
    
    #use the indices to sample items from both arrays
    sampled_data1 = [faces[i] for i in sampled_indices]
    sampled_data2 = [labels[i] for i in sampled_indices]
    
    return sampled_data1, sampled_data2

def createImage(num):
    label = convertMatrix(num)
    deltaO = label
    deltaH = np.transpose(hidden_output_weights) @ deltaO
    deltaI = np.transpose(input_hidden_weights) @ deltaH
    sns.heatmap(np.reshape(deltaI, (28,28)), annot=False, cmap='Greys', cbar=True)
    plt.title('Neural Network Weights Visualization for Digit ' + str(num) + ":\n" + str(epochs) + " Epochs, " + str(learnRate) + " Learning Rate")
    plt.show()

#########################
#### STORING WEIGHTS ####
#########################

#stores current weights and additional parameters
def store_weights(file_name, weights1, weights2, bias1, bias2, Trials, epochs, learningRate):
    data_to_store = {
        'weights1': weights1,
        'weights2': weights2,
        'bias1': bias1,
        'bias2': bias2,
        'learningRate': learningRate,
        'Trials': Trials,
        'epochs': epochs,
    }
    with open(os.path.join(directory, file_name + '.pkl'), 'wb') as f:
        pickle.dump(data_to_store, f)
    print(f"\nWeights and parameters saved to {file_name + '.pkl'}\n")

#retrieves current weights and parameters
def retrieve_weights(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            data_loaded = pickle.load(f)
        return data_loaded 
    else:
        raise FileNotFoundError(f"No such file: {file_name}\n")
    
#lists all current available weights
def list_weight_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    if not files:
        print("\nNo weight files found.\n")
        return None
    for idx, file in enumerate(files, start=1):
        print(f"{idx}: {file}")
    return files

#returns selected weights
def select_and_load_weights(directory):
    files = list_weight_files(directory)
    if files is None:
        return None

    while True:
        try:
            choice = int(input("\nWhich weights would you like to retrieve? Enter the number: "))
            if 1 <= choice <= len(files):
                file_name = files[choice - 1]
                return retrieve_weights(os.path.join(directory, file_name))
            else:
                print("Invalid choice. Please choose a number from the list.\n")
        except ValueError:
            print("Please enter a valid number.\n")

#######################
#### TESTING SUITE ####
#######################

#Variables can be adjusted at the top of the script!
training_done = False
full_training_done = False

# Load data once
training_faces = read_faces_from_file("./data/digitdata/trainingimages.txt")
training_labels = read_labels("./data/digitdata/traininglabels.txt")
testing_faces = read_faces_from_file("./data/digitdata/testimages.txt")
testing_labels = read_labels("./data/digitdata/testlabels.txt")

#train from scratch if requested
from_scratch = input("\nWould you like to train from scratch (y/n): ")
if(from_scratch == 'y'):

    #determine what percentage of the training data to use
    percentage = input("\nWhat percentage of data should be used for training? (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, all): ")
    if(percentage == 'all'):
        percent_loops = 11
        choice = 0.1
        percentages = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    else: 
        percent_loops = 2
        choice = float(percentage)
        percentages = [0, float(percentage)]

    average_accuracies = []
    average_times = []
    average_stds = []

    #Percentage loop
    for i in range(11):
        total_accuracy = []
        total_time = []

        #Trial loop
        for j in range(Trials):
            resetWeights()

            #Randomly shuffle the training data
            x,y = sample_percentage_two_arrays(training_faces, training_labels, i * 10) #shuffled and sampled
            start_train_time = time.time() #training starts
            
            #epoch loop
            for _ in range(epochs):
                trainImageArray(x, y)
            train_time = time.time() - start_train_time #training ends

            #Append accuracy and time for trial
            total_accuracy.append(testImageArray(testing_faces, testing_labels))
            total_time.append(train_time)

        #calculate results for this trial
        average_accuracy = mean(total_accuracy)
        average_time = mean(total_time)
        std_accuracy = std(total_accuracy)
        std_time = std(total_time)
        total_time = sum(total_time)

        #Display all results in terminal
        print("\n-------------------------------------------------------------------")
        print(str(Trials) + " Trials, " + str(epochs) + " Epochs, " + str(learnRate) + " learning rate, " + str(round(i * 0.1 * 100,3)) 
            + " percent of data.") 
        print("-------------------------------------------------------------------")
        print("Average accuracy: " + str(round(average_accuracy,round_digits)) + " percent.") 
        print("Standard Deviation of accuracy: " + str(round(std_accuracy,round_digits)) + " percent.\n") 

        print("Average training time: " + str(round(average_time,round_digits)) + " seconds.")
        print("Standard Deviation of training time: " + str(round(std_time,round_digits)) + " seconds.") 
        print("Total Training Time: " + str(round(total_time,round_digits)) + " seconds.")
        print("-------------------------------------------------------------------\n")

        #Save trial averages for plotting
        average_accuracies.append(average_accuracy)
        average_times.append(average_time)
        average_stds.append(std_accuracy)
        # trained_weights.append(weights)   
        training_done = True
        full_training_done = True 

    #store new weights
    store = input("\nWould you like to save these weights? (y/n): ")
    if(store == 'y'):
        name = input("\nWhat would you like the name them?: ")
        store_weights(name, input_hidden_weights, hidden_output_weights, input_hidden_bias, hidden_output_bias, Trials, epochs, learnRate)
else:
    #load saved weights
    data = select_and_load_weights(directory)
    
    if data is not None:
        input_hidden_weights = data['weights1']
        hidden_output_weights = data['weights2']
        input_hidden_bias = data['bias1']
        hidden_output_bias = data['bias2']
        learnRate = data['learningRate']
        Trials = data['Trials']
        epochs = data['epochs']
        print("Loaded weights.\n")
        training_done = True

        #display weight map of loaded weights
        show_weight = input("\nWould you like to see the weight map? (y/n): ")
        if(show_weight == 'y'):
            #Weights visualization heatmap
            for i in range(10):
                createImage(i)
            
#Show plots if user requests
if(full_training_done): #only plot if they trained through every percentage
    show_plot = input("\nWould you like to plot results? (y/n): ")
    if(show_plot == 'y'):
        #Create a plot of the average results
        plt.figure()
        plt.plot(percentages, average_accuracies, label= 'Average Accuracy (Percentage)')  
        plt.plot(percentages, average_stds, label = 'Standard Deviation of Accurac')
        plt.title('Accuracy Results: \n' + str(Trials) + " Trials, " + str(epochs) + " Epochs, " + str(learnRate) + " learning rate ")
        plt.xlabel('Percentage of Training Data')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([i * 0.1 for i in range(11)]) 
        plt.yticks([i * 0.1 for i in range(11)])
        plt.legend()
        plt.show()

        plt.figure()
        plt.title('Train Time Results: \n' + str(Trials) + " Trials, " + str(epochs) + " Epochs, " + str(learnRate) + " learning rate ")
        plt.plot(percentages, average_times, label= 'Average Train Time (Seconds)') 
        plt.ylabel('Average Train Time (seconds)')
        plt.xlabel('Percentage of Training Data')
        plt.xlim(0, 1)
        plt.xticks([i * 0.1 for i in range(11)]) 
        plt.legend()
        plt.show()

        for i in range(10):
            createImage(i)

#Tests trained perceptron on specified index
def test_one(index):
    result = testOneImage(testing_faces[index])
    print("Network Guess: " + str(result) + "   Correct Label: " + str(testing_labels[index]))

#Promt user for request
if(training_done):
    cont = True
    while(cont):
        index = input("\nWhich image to test? (0-999, s to exit): ")
        if(index == "s"):
            cont = False
        else:
            if(abs(int(index)) < len(testing_faces)):
                print_face(testing_faces, int(index))
                test_one(int(index))
            else:
                print("Index out of range.\n")

print('\n\nHave a great day!\n\n')