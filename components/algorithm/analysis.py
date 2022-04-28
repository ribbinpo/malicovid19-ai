import requests
import pandas as pd

def getInformation():
    data = requests.get("https://covid19.ddc.moph.go.th/api/Cases/timeline-cases-by-provinces")
    df = pd.read_json(data.text)
    result = {}
    date_today = df["txn_date"].unique()[-1]
    #Phuket Information
    data_phuket_today = df[(df["txn_date"]==date_today)&(df["province"]=="ภูเก็ต")].reset_index()
    today_case = int(data_phuket_today["new_case"].values[0])
    today_death = int(data_phuket_today["new_death"].values[0])
    total_death = int(data_phuket_today["total_death"].values[0])
    total_case = df[df["province"]=="ภูเก็ต"][-1:]["total_case"].values - df[df["province"]=="ภูเก็ต"][-14:-13]["total_case"].values
    total_case = int(total_case[0])
    #Ranking
    ranking = df[(df["txn_date"]==date_today)][["province","new_case"]].sort_values(by="new_case",ascending=False).reset_index()
    ranking = {
        "rank": ranking.index.tolist(),
        "province": ranking["province"].values.tolist(),
        "cases": ranking["new_case"].values.tolist()
    }
    #Dictionary Format Bundle
    result["date"] = date_today
    result["ranking"] = ranking
    result["province"] = {"Phuket":{"new_case":today_case,"new_death":today_death,"total_death":total_death,"total_case":total_case}}
    # date province->phuket->today_case,today_death,total_cases,cases ranking
    return result