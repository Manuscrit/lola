import os
import json
import pandas as pd
from matplotlib import pyplot as plot

BASE_LOCATION = os.getcwd()
if __name__ == "__main__":

    # Read data from all subdir and create df for each subdir
    all_df = []
    all_data = {}
    all_child_dir = os.listdir(BASE_LOCATION)
    for child_dir in all_child_dir:
        child_dir_path = os.path.join(BASE_LOCATION, child_dir)
        if os.path.isdir(child_dir_path):
            child_data_file_path = os.path.join(child_dir_path, "progress.json")
            if os.path.exists(child_data_file_path):
                print("reading", child_data_file_path)
                with open(child_data_file_path) as json_file:
                    data = json.load(json_file)
                    df_data=[]
                    for data_line in data:
                        if "ret1" in data_line.keys():
                            df_data.append({
                                "return_agent_0": data_line["ret1"],
                                "return_agent_1": data_line["ret2"],
                            })
                        else:
                            df_data.append({
                                "return_agent_0": data_line["reward_agent0"]/20,
                                "return_agent_1": data_line["reward_agent1"]/20,
                                "C_agent_0": data_line["agent0_C"],
                                "C_agent_1": data_line["agent1_C"],
                            })
                    all_data[str(child_dir)]=df_data
                df = pd.DataFrame(df_data)
                all_df.append(df)

    # Merge data in one new df and plot it
    stop = False
    i = 0
    merge_df_data = []
    while not stop:
        row_data = {}
        stop = True
        for key, data in all_data.items():
            if len(data) > i + 1:
                if "C_agent_0" in data[i].keys():
                    row_data[f"{key}_return_agent_0"] = data[i]["return_agent_0"]
                    row_data[f"{key}_return_agent_1"] = data[i]["return_agent_1"]
                    # row_data[f"{key}_agent0_C"] = data[i]["C_agent_0"]
                    # row_data[f"{key}_agent1_C"] = data[i]["C_agent_1"]
                    stop = False
        merge_df_data.append(row_data)
        i+=1
    df = pd.DataFrame(merge_df_data)
    df.plot.line()
    plot.show()

    # Plot the df for each subdir
    for df in all_df:
        df.plot.line()
        plot.show()