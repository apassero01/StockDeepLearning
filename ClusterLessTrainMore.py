# Build models
#
# Use papermill to feed parameters to build_keras_models.ipynb, thus allowing multiple models to be run
#
# Author:
# Prof. Brian King
# Dept. of Computer Science
# Bucknell University


from datetime import datetime
import sys
import os
import multiprocessing as mp
import time
from multiprocessing import Pool
import papermill as pm

MAX_PARALLEL_PROCESSES = 4

feature_lists = [['PctChgClCl','sumPctChgClCl_2','sumPctChgClCl_3','sumPctChgClCl_4','sumPctChgClCl_5','sumPctChgClCl_6',
                'PctChgVol','sumPctChgVol_2','sumPctChgVol_3','sumPctChgVol_4','sumPctChgVol_5','sumPctChgVol_6',
                'PctChgVix','sumPctChgVix_2','sumPctChgVix_3','sumPctChgVix_4','sumPctChgVix_5','sumPctChgVix_6','rsi','Close','Volume']]
n_steps = [10,15,30,50,75] 

# cluster_features_lists = [['PctChgClCl','sumPctChgClCl_2','sumPctChgClCl_3','PctChgVol','sumPctChgVol_2','sumPctChgVol_3'],
#                           ['PctChgVix','PctChgVol','PctChgClCl'],['PctChgClCl','rsi'],
#                           ['PctChgClCl','sumPctChgClCl_2','sumPctChgClCl_3','sumPctChgClCl_4','PctChgVix','sumPctChgVix_2','sumPctChgVix_3','sumPctChgVix_4','PctChgVol','sumPctChgVol_2','sumPctChgVol_3','sumPctChgVol_4']]
cluster_features_lists = [['PctChgVix','PctChgVol','PctChgClCl'],['sumPctChgVix_3','sumPctChgVol_3','sumPctChgClCl_3'],['sumPctChgVix_6','sumPctChgVol_6','sumPctChgClCl_6'],['PctChgVix','PctChgVol','PctChgClCl','sumPctChgVix_6','sumPctChgVol_6','sumPctChgClCl_6']]


# Run names - use more than 1 in list to repeat all tickers with different names
run_name = datetime.today().strftime('%Y-%m-%d_%H-%M')

# tickerCombinations = [['spy','qqq'],['aapl']]
tickerCombinations = [['spy','qqq','nvda','aapl']]

# Set end_date to None to use up to the current time (which may screw up last data point if in middle of the day)
# Also, remember that datetime with no hour will use midnight 00:00:00 for that day
tstart = 2015
tend = 2020


def execute_notebook(args):
    tickerCombo = args[0]
    feature_list = args[1]
    n_steps = args[2]
    cluster_features_list = args[3]
    run_number = args[4]
    

    try:
        output_notebook = 'temp_notebooks/cluster_more_{}_{}.ipynb'.format(run_name,run_number)
        print(f"*** STARTING {output_notebook} ***")
        print("*** START TIME: {}".format(datetime.now()))

        pm.execute_notebook(
            'ClusterLessTrainMore.ipynb',
            output_notebook,
            parameters = dict(tickerList=tickerCombo,
                              run_name=run_name,
                              n_steps=n_steps,
                              tstart=tstart,
                              tend=tend,
                              feature_list=feature_list,
                              cluster_features = cluster_features_list,
                              run_number = run_number)
        )
    except pm.PapermillExecutionError as e:
        print("*** PapermillExecutionError encountered! ***")
        print(e)  # Print the error message

    except KeyboardInterrupt:
        print("*** KeyboardInterrupt...  ***")

def update_processes(processes: list):

    for p in processes:
        if not p.is_alive():
            processes.remove(p)

if __name__ == "__main__":

    # Set up a list of the arguments
    run_number = 0
    input_args = []
    for feature_list in feature_lists:
        for tickerCombo in tickerCombinations:
            for steps in n_steps:
                for cluster_list in cluster_features_lists:
                    input_args.append([tickerCombo, feature_list,steps,cluster_list,run_number ])
                    print(input_args[-1])
                    run_number+=1

    processes = []

    for args in input_args:
        try:
            # Wait for space to open up
            while len(processes) >= MAX_PARALLEL_PROCESSES:
                time.sleep(2)
                update_processes(processes)

            # create the next processes
            process = mp.Process(target=execute_notebook, args=[args])
            processes.append(process)
            process.start()
            time.sleep(5)   # Wait a bit before starting the next one

            # process.join()
        except pm.PapermillExecutionError:
            print("*** Guessing this is a KeyboardInterrupt error? Moving to next run... ***")
        except:
            break  # All other errors

