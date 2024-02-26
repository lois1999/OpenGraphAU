import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compute_mean_activation(data, participant, turn):
    df_au_activation = pd.read_csv(data)
    df_participant_turn = df_au_activation.loc[(df_au_activation["participant_id"] == participant) & (df_au_activation["turn"] == turn)]
    df_mean_activation = df_participant_turn.iloc[:,4:].mean(axis=0)
    df_std_activation = df_participant_turn.iloc[:,4:].std(axis=0)

    return df_mean_activation, df_std_activation

if __name__ == "__main__":
    mean_activations, std_activations = compute_mean_activation("predictions_AU.csv", "m", "n")
    
    f = plt.figure()
    f.set_figwidth(50)
    plt.rcParams.update({'font.size': 6})
    plt.bar(list(mean_activations.index), mean_activations)
    plt.axhline(y=50, color='r', linestyle='-')
    plt.ylabel("Intensity")
    plt.xlabel("Action unit")
    plt.ylim(0, 100)
    plt.show()