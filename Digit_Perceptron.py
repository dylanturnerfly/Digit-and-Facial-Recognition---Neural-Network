import os
import random
import time
from numpy import mean, std
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

##############################
#### ADJUSTABLE VARIABLES ####
##############################

directory = './perceptron_weights/stored_weights_digit' #path to weights directory

imageHeight = 28 #do not change
imageWidth = 28 #do not change

rowFeatures = 28 #features per row
colFeatures = 28 #features per column

Trials = 15 #how many trials for each percentage of dataset
epochs = 3 #how many epochs for each trial
round_digits = 4 #how many digits to round display results to

show_indidivual_results = False #debugging info (WARNING: spam)
show_group_results = False #debugging info

##############################
#### PERCEPTRON FUNCTIONS ####
##############################

#Function to read all digits from a file into an array of digits
def read_digits_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    digits = []  #digits array to return
    current_digit = []  #temporary current digit data
    line_count = 0 

    for line in lines:
        #convert line to list of 1s and 0s
        row = [1 if (char != ' ' and char!= "\n") else 0 for char in line]
        current_digit.append(row)
        line_count += 1

        #check if we have read 28 lines
        if line_count == imageHeight:
            digits.append(current_digit)
            current_digit = []  
            line_count = 0 

    #add the last digit if it wasn't added yet (# of lines was not multiple of 70)
    if current_digit: 
        digits.append(current_digit)

    return digits

#Function to read labels into a 1D array
def read_labels(filename):
    labels = []
    with open(filename, 'r') as file:
            content = file.read()
    labels = [int(char) for char in content if char in '0123456789']
    return labels 
    
#Function to print a digit in the console
def print_digit(digits, index):
    if index < len(digits):
        digit = digits[index]
        for row in digit:
            print(''.join('#' if pixel == 1 else ' ' for pixel in row))
    else:
        print("Index out of range.\n")

#Function to extract the value for each feature in a digit
def feature_extract(digit, rows, cols):
    feature_height = len(digit) // rows
    feature_width = len(digit[0]) // cols
    feature_counts = [[0] * cols for _ in range(rows)]  #2D list for feature counts

    for i in range(len(digit)):
        for j in range(len(digit[0])):
            if digit[i][j] == 1:
                feature_row = i // feature_height
                feature_col = j // feature_width
                #check boundaries for features at the edges (for when its not perfectly divisible)
                if feature_row >= rows:
                    feature_row = rows - 1
                if feature_col >= cols:
                    feature_col = cols - 1
                feature_counts[feature_row][feature_col] += 1
    #PRINT FOR TESTING
    # for row in feature_counts:
    #     for item in row:
    #         print(item, end=' ')  
    #     print() 
    #PRINT FOR TESTING
    linear_feature_counts = sum(feature_counts, []) #linearize 2D array so its easy to do later calculations
    
    return linear_feature_counts

#Randomly initializes 9 sets of weights between -1 and 1 
def initialize_weights(feature_count):
    all_weights = []
    for i in range(10):
        random_weights = [0] * (feature_count +  1)
        for j in range(feature_count):
            random_weights[j] = round(random.uniform(-1, 1), round_digits)
        all_weights.append(random_weights)
    return all_weights

#Add together all the (weight, features) pairs and return the index of the weight set with the highest results
def whichDigit(all_weights, features):
    totals = [0] * len(all_weights)
    for i in range(len(all_weights)):
        weights = all_weights[i]
        for j in range(len(weights) - 1):
            totals[i] += weights[j] * features[j]
        totals[i] += weights[-1]
    return (totals.index(max(totals))
)

#update the weights through training with a given percentage of the input data
def train_once(all_weights, digits, labels, percentage):
    #loop through every digit and update weights
    for i in range(int(len(digits) * percentage)):
        digit = digits[i]
        features = feature_extract(digit,rowFeatures,colFeatures)
        result = whichDigit(all_weights, features)
        correct = labels[i]
        if show_indidivual_results: 
            print("Training Result " + str(i) + ": " + str(result) + "   Correct Label: " + str(correct))

        #update weights
        if(result != correct):
            for i in range(len(all_weights[0]) - 1):
                all_weights[result][i] -= features[i]
                all_weights[correct][i] += features[i]
            all_weights[result][-1] -= 1
            all_weights[correct][-1] += 1
    return all_weights

#tests the perceptron on the digit data given and report accuracy
def test_once(weights, digits, labels):
    total_correct = len(digits)
    total = len(digits)
    #loop through every digit and test accuracy
    for i in range(total):
        digit = digits[i]
        features = feature_extract(digit,rowFeatures,colFeatures)
        result = whichDigit(weights, features)
        correct = labels[i]
        if show_indidivual_results: 
            print("Test Result: " + str(result) + "   Correct Label: " + str(correct))
        if result != correct:
            total_correct -= 1
    if show_group_results:
        print(str(round(total_correct / total, round_digits)) + " percent correct --> " + str(total_correct) + " out of " + str(total) + ".\n")
    return total_correct / total

#########################
#### STORING WEIGHTS ####
#########################

#stores current weights and additional parameters
def store_weights(weights, file_name, rowFeatures, colFeatures, Trials, epochs):
    data_to_store = {
        'weights': weights,
        'rowFeatures': rowFeatures,
        'colFeatures': colFeatures,
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

#Extract digits and Labels
digitsDataTrain = read_digits_from_file('./data/digitdata/trainingimages.txt')
labels = read_labels("./data/digitdata/traininglabels.txt")
digitsDataValidation = read_digits_from_file('./data/digitdata/validationimages.txt')
validlabels = read_labels("./data/digitdata/validationlabels.txt")

#train from scratch if user requests
test_percents = input("\nWould you like to train from scratch? (y/n): ")
if(test_percents == "y"):

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

    #set up arrays to record averages
    average_accuracies = []
    average_times = []
    average_stds = []
    all_trained_weights = []

    #Percentage Loop
    for i in range(percent_loops):
        total_accuracy = []
        total_time = []

        #Trial Loop
        for j in range(Trials):
            #Randomly shuffle the training data
            combined_data = list(zip(digitsDataTrain, labels))
            random.shuffle(combined_data)
            shuffled_digitsDataTrain, shuffled_labels = zip(*combined_data)

            #Reinitialize weights
            all_weights = initialize_weights(rowFeatures * colFeatures)
            start_train_time = time.time() #training starts

            #Epoch Loop
            for k in range(epochs):
                all_weights = train_once(all_weights, shuffled_digitsDataTrain, shuffled_labels, i * choice)
            train_time = time.time() - start_train_time #training ends
            if(show_indidivual_results):
                print("Completed training in: " + str(round(train_time,round_digits)))

            #Append accuracy and time for trial
            total_accuracy.append(test_once(all_weights, digitsDataValidation, validlabels))
            total_time.append(train_time)

        #Calculate average and standard deviation of all trials
        average_accuracy = mean(total_accuracy)
        average_time = mean(total_time)
        std_accuracy = std(total_accuracy)
        std_time = std(total_time)
        total_time = sum(total_time)

        #Display all results in terminal
        print("\n-------------------------------------------------------------------")
        print(str(Trials) + " Trials, " + str(epochs) + " Epochs, " + str(round(i * choice * 100,3)) 
            + " percent of data, " + str(rowFeatures) + " row features, "  + str(colFeatures) 
            +  " column features") 
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
        all_trained_weights.append(all_weights)
        training_done = True
        full_training_done = True

    #store new weights
    store = input("\nWould you like to save these weights? (y/n): ")
    if(store == 'y'):
        name = input("\nWhat would you like the name them?: ")
        store_weights(all_trained_weights[-1], name, rowFeatures, colFeatures, Trials, epochs)
else:
    #load saved weights
    data = select_and_load_weights(directory)
    
    if data is not None:
        all_weights = data['weights']
        rowFeatures = data['rowFeatures']
        colFeatures = data['colFeatures']
        Trials = data['Trials']
        epochs = data['epochs']
        print("Loaded weights.\n")
        training_done = True

        #display weight map of loaded weights
        show_weight = input("\nWould you like to see the weight map? (y/n): ")
        if(show_weight == 'y'):
             #Weights visualization heatmap
            for i in range(10):
                plt.figure()
                numpy_trained_weights = numpy.array(all_weights[i])
                numpy_trained_weights = numpy_trained_weights[:-1].reshape(rowFeatures,colFeatures)     
                sns.heatmap(numpy_trained_weights, annot=False, cmap='Greys', cbar=True)
                plt.title('Perceptron Weights Visualization for Digit ' + str(i) + '\n'   + str(epochs) + " Epochs, " + str(rowFeatures) + " row features, "  + str(colFeatures) +  " column features")
                plt.show()

#Show plots if user requests
if(full_training_done): #only plot if they trained through every percentage
    show_plot = input("\nWould you like to plot results? (y/n): ")
    if(show_plot == 'y'):

        #Average Accuracy plot
        plt.figure()
        plt.plot(percentages, average_accuracies, label= 'Average Accuracy (Percentage)')
        plt.plot(percentages, average_stds, label = 'Standard Deviation of Accuracy')
        plt.title('Accuracy Results: \n' + str(Trials) + " Trials, " + str(epochs) + " Epochs, " + str(rowFeatures) + " row features, "  + str(colFeatures) +  " column features")
        plt.xlabel('Percentage of Training Data')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([i * 0.1 for i in range(11)]) 
        plt.yticks([i * 0.1 for i in range(11)])
        plt.legend()
        plt.show()

        #Average Train time and standard deviation plot
        plt.figure()
        plt.title('Average Train Time Results: \n' + str(Trials) + " Trials, " + str(epochs) + " Epochs, " + str(rowFeatures) + " row features, "  + str(colFeatures) +  " column features")
        plt.plot(percentages, average_times, label= 'Average Train Time (Seconds)') 
        plt.xlim(0, 1)
        plt.xticks([i * 0.1 for i in range(11)]) 
        plt.legend()
        plt.show()

        #Weights visualization heatmap
        for i in range(10):
            plt.figure()
            numpy_trained_weights = numpy.array(all_trained_weights[-1][i])
            numpy_trained_weights = numpy_trained_weights[:-1].reshape(rowFeatures,colFeatures)     
            sns.heatmap(numpy_trained_weights, annot=False, cmap='Greys', cbar=True)
            plt.title('Perceptron Weights Visualization for Digit ' + str(i) + ":\n" + str(epochs) + " Epochs, " + str(rowFeatures) + " row features, "  + str(colFeatures) +  " column features")
            plt.show()

#Tests trained perceptron on specified index
def test_one(index):
    digit = digitsDataValidation[index]
    features = feature_extract(digit,rowFeatures,colFeatures)
    result = whichDigit(all_weights, features)
    correct = validlabels[index] 
    print("Perceptron Guess: " + str(result) + "   Correct Label: " + str(correct))
    
#Promt user for request
if(training_done):
    cont = True
    while(cont):
        index = input("\nWhich image to test? (0-999, s to exit): ")
        if(index == "s"):
            cont = False
        else:
            if(abs(int(index)) < len(digitsDataValidation)):
                print_digit(digitsDataValidation, int(index))
                test_one(int(index))
            else:
                print("Index out of range.\n")

print('\n\nHave a great day!\n\n')