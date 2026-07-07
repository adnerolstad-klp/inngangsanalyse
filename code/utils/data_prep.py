import pandas as pd

def data_prep(df_inngang, df_info, df_weather, cols_to_remove=None):
    
    df_inngang.columns = df_inngang.columns.str.lower()
    df_info.columns = df_info.columns.str.lower()
    df_weather.columns = df_weather.columns.str.lower()


    df_inngang['ankomst_dato'] = pd.to_datetime(df_inngang['ankomst_dato'], format="%d.%m.%Y %H:%M:%S").dt.date
    # Respons (behandlingstid) som sekunder
    df_inngang['behandlingstid'] = (
        df_inngang['behandlingstid']
        .str.split(':')
        .apply(lambda x: int(x[0]) * 60 + int(x[1]))
    )
    # Også etterbehandlingstid som sekunder:
    df_inngang['etterbehandlingstid'] = (
        df_inngang['etterbehandlingstid']
        .str.split(':')
        .apply(lambda x: int(x[0]) * 60 + int(x[1]))
    )
    df_inngang["behandlet"]=df_inngang["behandlet"].eq("Behandlet")
    # Aggregering til dag-nivå
    df_dag = (
        df_inngang.groupby("ankomst_dato", as_index=False)
        .agg(
            antall_samtaler=("unique_id", "size"),
            behandlet_andel=("behandlet", "mean"),
            tid_i_ko_snitt=("tid_i_ko", "mean"),
            behandlingstid_snitt=("behandlingstid", "mean"),
            total_behandlingstid=("behandlingstid", "sum"),
            etterbehandligstid_snitt=("etterbehandlingstid", "mean"), 
            total_etterbehandligstid=("etterbehandlingstid", "sum")
        ).
        sort_values("ankomst_dato")
        .reset_index(drop=True)
    )

    df_info['hf_dato'] = pd.to_datetime(df_info['hf_dato'], format="%Y-%m-%d").dt.date

    # Fiks tid-format
    df_weather["tid(norsk normaltid)"]= pd.to_datetime(df_weather["tid(norsk normaltid)"], format="%d.%m.%Y").dt.date
    # Fiks datatype
    df_weather["nedbør (døgn)"] = df_weather["nedbør (døgn)"].str.replace(",", ".", regex=True).astype(float)
    df_weather["middeltemperatur (døgn)"] = df_weather["middeltemperatur (døgn)"].str.replace(",", ".", regex=True).astype(float)
    df_weather = df_weather.rename(columns={"nedbør (døgn)": "nedbør", "middeltemperatur (døgn)": "middeltemperatur"})

    df_temp = pd.merge(df_dag, df_info, left_on="ankomst_dato", right_on="hf_dato", how="left")
    df = pd.merge(df_temp, df_weather, left_on="ankomst_dato", right_on="tid(norsk normaltid)", how="left")

    # Temperaturavvik fra månedssnitt (per kalender-måned)
    month_key = pd.to_datetime(df["ankomst_dato"]).dt.to_period("M")
    df["middeltemp_mndsnitt"] = df.groupby(month_key)["middeltemperatur"].transform("mean")
    df["tempavvik_fra_mndsnitt"] = df["middeltemperatur"] - df["middeltemp_mndsnitt"]

    df["antall_nye_kunder_b30_ny"] = df["antall_nye_kunder_b30_mpb01_ny"] + df["antall_nye_kunder_b30_eph01_ny"] + df["antall_nye_kunder_b30_epf01_ny"] + df["antall_nye_kunder_b30_upr01_ny"]
    df["antall_nye_kunder_b30_for"] = df["antall_nye_kunder_b30_mpb01_for"] + df["antall_nye_kunder_b30_eph01_for"] + df["antall_nye_kunder_b30_epf01_for"] + df["antall_nye_kunder_b30_upr01_for"]
    df["antall_hf_b30_ny"] = df["antall_hf_b30_mpb01_ny"] + df["antall_hf_b30_eph01_ny"] + df["antall_hf_b30_epf01_ny"] + df["antall_hf_b30_upr01_ny"]
    df["antall_hf_b30_for"] = df["antall_hf_b30_mpb01_for"] + df["antall_hf_b30_eph01_for"] + df["antall_hf_b30_epf01_for"]  + df["antall_hf_b30_upr01_for"]
    df["stddev_premieendring_b30_for"] = (df["stddev_premieendring_b30_eph01_for"] + df["stddev_premieendring_b30_epf01_for"] + df["stddev_premieendring_b30_mpb01_for"] + df["stddev_premieendring_b30_upr01_for"]) / 4
    df["snitt_premieendring_b30_for"] = (df["snitt_premieendring_b30_eph01_for"] + df["snitt_premieendring_b30_epf01_for"] + df["snitt_premieendring_b30_mpb01_for"] + df["snitt_premieendring_b30_upr01_for"]) / 4

    ### Tar totalen av ny og for også:

    df["antall_nye_kunder_b30_tot"] = df["antall_nye_kunder_b30_ny"] + df["antall_nye_kunder_b30_for"]
    df["antall_hf_b30_tot"] = df["antall_hf_b30_ny"] + df["antall_hf_b30_for"]

    if cols_to_remove != None: 
        df = df.drop(columns=cols_to_remove)
    
    return df