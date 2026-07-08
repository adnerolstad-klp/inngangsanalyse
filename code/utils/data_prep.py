import pandas as pd


def _to_minutes(series: pd.Series) -> pd.Series:
    return series.str.split(":").apply(lambda value: int(value[0]) * 60 + int(value[1]))


def _lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    return df


def _prepare_inngang(df_inngang: pd.DataFrame) -> pd.DataFrame:
    df_inngang = _lowercase_columns(df_inngang)
    df_inngang["ankomst_dato"] = pd.to_datetime(
        df_inngang["ankomst_dato"], format="%d.%m.%Y %H:%M:%S"
    ).dt.date
    df_inngang["behandlingstid"] = _to_minutes(df_inngang["behandlingstid"])
    df_inngang["etterbehandlingstid"] = _to_minutes(df_inngang["etterbehandlingstid"])
    df_inngang["behandlet"] = df_inngang["behandlet"].eq("Behandlet")
    return df_inngang


def _aggregate_inngang(df_inngang: pd.DataFrame) -> pd.DataFrame:
    return (
        df_inngang.groupby("ankomst_dato", as_index=False)
        .agg(
            antall_samtaler=("unique_id", "size"),
            behandlet_andel=("behandlet", "mean"),
            tid_i_ko_snitt=("tid_i_ko", "mean"),
            behandlingstid_snitt=("behandlingstid", "mean"),
            total_behandlingstid=("behandlingstid", "sum"),
            etterbehandligstid_snitt=("etterbehandlingstid", "mean"),
            total_etterbehandligstid=("etterbehandlingstid", "sum"),
        )
        .sort_values("ankomst_dato")
        .reset_index(drop=True)
    )


def _prepare_info(df_info: pd.DataFrame) -> pd.DataFrame:
    df_info = _lowercase_columns(df_info)
    df_info["hf_dato"] = pd.to_datetime(df_info["hf_dato"], format="%Y-%m-%d").dt.date
    return df_info


def _prepare_weather(df_weather: pd.DataFrame) -> pd.DataFrame:
    df_weather = _lowercase_columns(df_weather)
    df_weather["tid(norsk normaltid)"] = pd.to_datetime(
        df_weather["tid(norsk normaltid)"], format="%d.%m.%Y"
    ).dt.date
    df_weather["nedbør (døgn)"] = (
        df_weather["nedbør (døgn)"].str.replace(",", ".", regex=True).astype(float)
    )
    df_weather["middeltemperatur (døgn)"] = (
        df_weather["middeltemperatur (døgn)"].str.replace(",", ".", regex=True).astype(float)
    )
    return df_weather.rename(
        columns={"nedbør (døgn)": "nedbør", "middeltemperatur (døgn)": "middeltemperatur"}
    )


def _merge_data(
    df_dag: pd.DataFrame,
    df_info: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    df_temp = pd.merge(df_dag, df_info, left_on="ankomst_dato", right_on="hf_dato", how="left")
    return pd.merge(
        df_temp,
        df_weather,
        left_on="ankomst_dato",
        right_on="tid(norsk normaltid)",
        how="left",
    )


def _add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    month_key = pd.to_datetime(df["ankomst_dato"]).dt.to_period("M")
    df["middeltemp_mndsnitt"] = df.groupby(month_key)["middeltemperatur"].transform("mean")
    df["tempavvik_fra_mndsnitt"] = df["middeltemperatur"] - df["middeltemp_mndsnitt"]
    return df


def _add_b30_features(df: pd.DataFrame) -> pd.DataFrame:
    df["antall_nye_kunder_b30_ny"] = (
        df["antall_nye_kunder_b30_mpb01_ny"]
        + df["antall_nye_kunder_b30_eph01_ny"]
        + df["antall_nye_kunder_b30_epf01_ny"]
        + df["antall_nye_kunder_b30_upr01_ny"]
    )
    df["antall_nye_kunder_b30_for"] = (
        df["antall_nye_kunder_b30_mpb01_for"]
        + df["antall_nye_kunder_b30_eph01_for"]
        + df["antall_nye_kunder_b30_epf01_for"]
        + df["antall_nye_kunder_b30_upr01_for"]
    )
    df["antall_hf_b30_ny"] = (
        df["antall_hf_b30_mpb01_ny"]
        + df["antall_hf_b30_eph01_ny"]
        + df["antall_hf_b30_epf01_ny"]
        + df["antall_hf_b30_upr01_ny"]
    )
    df["antall_hf_b30_for"] = (
        df["antall_hf_b30_mpb01_for"]
        + df["antall_hf_b30_eph01_for"]
        + df["antall_hf_b30_epf01_for"]
        + df["antall_hf_b30_upr01_for"]
    )
    df["stddev_premieendring_b30_for"] = (
        df["stddev_premieendring_b30_eph01_for"]
        + df["stddev_premieendring_b30_epf01_for"]
        + df["stddev_premieendring_b30_mpb01_for"]
        + df["stddev_premieendring_b30_upr01_for"]
    ) / 4
    df["snitt_premieendring_b30_for"] = (
        df["snitt_premieendring_b30_eph01_for"]
        + df["snitt_premieendring_b30_epf01_for"]
        + df["snitt_premieendring_b30_mpb01_for"]
        + df["snitt_premieendring_b30_upr01_for"]
    ) / 4
    df["antall_nye_kunder_b30_tot"] = df["antall_nye_kunder_b30_ny"] + df["antall_nye_kunder_b30_for"]
    df["antall_hf_b30_tot"] = df["antall_hf_b30_ny"] + df["antall_hf_b30_for"]
    return df


def load_newsletter_dates(csv_path: str) -> pd.Series:
    df_dates = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        usecols=[0],
        names=["utsendelsesdato"],
        dtype={"utsendelsesdato": "string"},
    )
    dates = pd.to_datetime(df_dates["utsendelsesdato"], format="%d.%m.%Y", errors="coerce").dt.date
    dates = dates.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    return dates


def add_newsletter_window_feature(
    df: pd.DataFrame,
    newsletter_dates: pd.Series,
    window_days: int = 7,
    feature_name: str = "nyhetsbrev_b7",
) -> pd.DataFrame:
    if "ankomst_dato" not in df.columns:
        raise ValueError("Mangler kolonnen 'ankomst_dato' i df.")

    if window_days < 1:
        raise ValueError("window_days ma vaere >= 1")

    out = df.copy()
    arrival = pd.to_datetime(out["ankomst_dato"])

    send_dates = pd.to_datetime(newsletter_dates.dropna().drop_duplicates().sort_values())
    if send_dates.empty:
        out[feature_name] = 0
        return out

    # Finn siste utsendelse som er strengt for aktuell dato (samme dag teller ikke).
    idx = send_dates.searchsorted(arrival, side="left") - 1
    has_prev = idx >= 0

    last_send = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    if has_prev.any():
        last_send.loc[has_prev] = send_dates.to_numpy()[idx[has_prev]]

    days_since = (arrival - last_send).dt.days
    out[feature_name] = ((days_since >= 1) & (days_since <= window_days)).astype(int)
    return out


def data_prep(
    df_inngang,
    df_info,
    df_weather,
    cols_to_remove=None,
    newsletter_dates_csv: str | None = None,
):
    df_inngang = _prepare_inngang(df_inngang)
    df_dag = _aggregate_inngang(df_inngang)
    df_info = _prepare_info(df_info)
    df_weather = _prepare_weather(df_weather)

    df = _merge_data(df_dag, df_info, df_weather)
    df = _add_temperature_features(df)
    df = _add_b30_features(df)

    if newsletter_dates_csv is not None:
        newsletter_dates = load_newsletter_dates(newsletter_dates_csv)
        df = add_newsletter_window_feature(
            df=df,
            newsletter_dates=newsletter_dates,
            window_days=7,
            feature_name="nyhetsbrev_b7",
        )

    if "ukedag" in df.columns:
        df = df[df["ukedag"] != 0].copy()

    if cols_to_remove is not None:
        df = df.drop(columns=cols_to_remove)

    return df