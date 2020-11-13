import json
import os

import pandas as pd
from matplotlib import pyplot as plot


def for_every_child_dir(path, function):
    all_child_dir = os.listdir(path)
    for child_dir in all_child_dir:
        child_dir_path = os.path.join(path, child_dir)
        if os.path.isdir(child_dir_path):
            function(child_dir_path)


def create_df_from_one_dir(child_dir_path):
    global all_df, all_df

    child_data_file_path = os.path.join(child_dir_path, "progress.json")
    if os.path.exists(child_data_file_path):
        print("reading", child_data_file_path)
        df_data = []
        try:
            with open(child_data_file_path) as json_file:
                data = json.load(json_file)
        except json.decoder.JSONDecodeError:
            with open(child_data_file_path) as json_file:
                data = json_file.readlines()
                data = [json.loads(line) for line in data]
        for data_line in data:
            if "AsymCoinGame" in child_data_file_path:
                df_data.append(extract_asym_coin_game_data(data_line))
            elif "CoinGame" in child_data_file_path:
                df_data.append(extract_coin_game_data(data_line))
            elif "IPD" in child_data_file_path:
                df_data.append(extract_ipd_data(data_line))
            else:
                df_data.append(extract_asym_coin_game_data(data_line))

        head, tail = os.path.split(child_dir_path)
        h_head, h_tail = os.path.split(head)
        h_h_head, h_h_tail = os.path.split(h_head)
        key = os.path.join(h_h_tail, h_tail, tail)
        all_data[key] = df_data
        all_df.append(pd.DataFrame(df_data))
    else:
        # Recursive call to extract from all subdirs
        for_every_child_dir(path=child_dir_path, function=create_df_from_one_dir)


def extract_asym_coin_game_data(data_line):
    n_episodes = 4000
    reward_ag_0 = 2 * data_line["rList[0]"] + data_line["rList[2]"] - 1 * data_line["rList[3]"]
    reward_ag_1 = data_line["rList[1]"] + data_line["rList[3]"] - 2 * data_line["rList[2]"]
    pick_own = (data_line["rList[0]"] + data_line["rList[1]"] +
                data_line["rList[4]"] + data_line["rList[5]"])
    picks = (data_line["rList[0]"] + data_line["rList[1]"] +
             data_line["rList[2]"] + data_line["rList[3]"] +
             2 * data_line["rList[4]"] + 2 * data_line["rList[5]"])
    pick_own = pick_own / picks

    return {"return_agent_0": reward_ag_0 / n_episodes,
            "return_agent_1": reward_ag_1 / n_episodes,
            "pick_own": pick_own}


def extract_coin_game_data(data_line):
    n_episodes = 4000
    reward_ag_0 = data_line["rList[0]"] + data_line["rList[2]"] - 2 * data_line["rList[3]"]
    reward_ag_1 = data_line["rList[1]"] + data_line["rList[3]"] - 2 * data_line["rList[2]"]
    pick_own = (data_line["rList[0]"] + data_line["rList[1]"] +
                data_line["rList[4]"] + data_line["rList[5]"])
    picks = (data_line["rList[0]"] + data_line["rList[1]"] +
             data_line["rList[2]"] + data_line["rList[3]"] +
             2 * data_line["rList[4]"] + 2 * data_line["rList[5]"])
    pick_own = pick_own / picks

    return {"return_agent_0": reward_ag_0 / n_episodes,
            "return_agent_1": reward_ag_1 / n_episodes,
            "pick_own": pick_own}


def extract_ipd_data(data_line):
    if "ret1" in data_line.keys():
        return {"return_agent_0": data_line["ret1"],
                "return_agent_1": data_line["ret2"]}
    else:
        return {"return_agent_0": data_line["reward_agent0"] / 20,
                "return_agent_1": data_line["reward_agent1"] / 20,
                "C_agent_0": data_line["agent0_C"],
                "C_agent_1": data_line["agent1_C"]}


def merge_in_env_df_and_plot():
    stop = False
    i = 0
    merge_df_data = {}
    while not stop:
        row_data = {}
        stop = True
        for key, data in all_data.items():
            if "CoinGame" in key:
                merged_df_key = "CoinGame"
            elif "IPD" in key:
                merged_df_key = "IPD"
            else:
                raise ValueError(f"key: {key}")
            if merged_df_key not in row_data.keys():
                row_data[merged_df_key] = {}

            if len(data) > i + 1:
                # if "C_agent_0" in data[i].keys():
                row_data[merged_df_key][f"{key}_return_agent_0"] = data[i]["return_agent_0"]
                row_data[merged_df_key][f"{key}_return_agent_1"] = data[i]["return_agent_1"]
                if "pick_own" in data[i].keys():
                    row_data[merged_df_key][f"{key}_pick_own"] = data[i]["pick_own"]
                stop = False

        for key, value in row_data.items():
            if key not in merge_df_data.keys():
                merge_df_data[key] = []
            merge_df_data[key].append(value)
        # merge_df_data.append(row_data)
        i += 1

    for key, value in merge_df_data.items():
        print("Plotting", key, value)
        df = pd.DataFrame(value)
        df.plot.line()
        plot.show()


BASE_LOCATION = os.getcwd()
if __name__ == "__main__":

    # Read data from all subdir and create df for each subdir
    all_df = []
    all_data = {}
    for_every_child_dir(path=BASE_LOCATION, function=create_df_from_one_dir)

    merge_in_env_df_and_plot()

    # Plot the df for each subdir
    for df in all_df:
        df.plot.line()
        plot.show()
