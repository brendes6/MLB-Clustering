import pandas as pd

allstars = pd.read_csv("Data/AllstarFull.csv")
batting = pd.read_csv("Data/MasterNonPitching.csv")

batting["Allstar"] = 0
allstar_players = allstars["playerID"].unique()
batting.loc[batting["playerID"].isin(allstar_players), "Allstar"] = 1

batting.to_csv("Data/MasterNonPitching.csv", index=False)