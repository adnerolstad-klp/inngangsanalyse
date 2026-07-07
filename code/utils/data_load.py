import os
from pathlib import Path

import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session


def get_or_create_session(schema: str = "PRODUKT_WRITE_DEV") -> Session:
	if "POSIT_PRODUCT" in os.environ:
		session = Session.builder.getOrCreate()
		session.sql("USE DATABASE PROD_FOR_SKADE_PRODUKT_ADHOC").collect()
		session.sql(f"USE SCHEMA {schema}").collect()
		session.sql("USE WAREHOUSE SKADE_VWH").collect()
		return session

	try:
		return get_active_session()
	except Exception:
		import win32api

		connection_parameters = {
			"server": "km28161.west-europe.azure.snowflakecomputing.com",
			"warehouse": "SKADE_VWH",
			"account": "VK82539-KLP",
			"database": "PROD_FOR_SKADE_PRODUKT_ADHOC",
			"schema": schema,
			"user": win32api.GetUserNameEx(win32api.NameUserPrincipal),
			"authenticator": "externalbrowser",
		}
		return Session.builder.configs(connection_parameters).create()


def load_raw_data(
	session: Session | None = None,
	weather_path: str | Path = "../../data/oslo_weather.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if session is None:
		session = get_or_create_session()

	df_inngang = session.table("elh_write.inngangsdata").to_pandas()
	df_info = session.table("inngangsdata_info").to_pandas()
	df_weather = pd.read_csv(weather_path, delimiter=";")

	df_inngang.columns = df_inngang.columns.str.lower()
	df_info.columns = df_info.columns.str.lower()
	df_weather.columns = df_weather.columns.str.lower()

	return df_inngang, df_info, df_weather