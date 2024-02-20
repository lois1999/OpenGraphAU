from feat.plotting import plot_face
import numpy as np
import pandas as pd

present_in_pyfeat = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"]

def compute_mean_activation(data, participant, turn, present_in_pyfeat=present_in_pyfeat):
    df_au_activation = pd.read_csv(data)
    df_participant_turn = df_au_activation.loc["participant_id" == participant & "turn_id" == turn]
    df_mean_activation = df_participant_turn.mean(axis=0)

    activations_faceplot = []
    for au in present_in_pyfeat:
        activations_faceplot.append(df_mean_activation[au])

    return df_mean_activation, activations_faceplot

if __name__ == "__main__":
    all_activations, activations_pyfeat = compute_mean_activation("predictions_AU.csv")
    print(all_activations)
    
