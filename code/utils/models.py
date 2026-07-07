import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import (
	mean_absolute_error,
	mean_poisson_deviance,
	mean_squared_error,
	r2_score,
	make_scorer,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CATEGORICAL_FEATURES = [
	"aar",
	"kvartal",
	"maaned",
	"ukenummer",
	"ukedag",
	"dag_i_maaned",
	"er_helg",
	"helligdag",
	"er_helligdag",
	"er_dag_foer_helligdag",
	"er_dag_etter_helligdag",
]


def _resolve_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
	missing = [col for col in feature_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Mangler feature-kolonner i df: {missing}")
	return feature_cols


def _prepare_xy(
	df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
	if target_col not in df.columns:
		raise ValueError(f"Mangler target-kolonne i df: {target_col}")

	feature_cols = _resolve_feature_columns(df, feature_cols)
	frame = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
	X = frame[feature_cols]
	y = frame[target_col].astype(float)
	return X, y


def _split_features(
	X: pd.DataFrame,
	categorical_features: list[str] | None = None,
) -> tuple[list[str], list[str]]:
	forced_cat = set(categorical_features or []) & set(X.columns)
	inferred_cat = {
		col
		for col in X.columns
		if pd.api.types.is_object_dtype(X[col])
		or pd.api.types.is_bool_dtype(X[col])
		or isinstance(X[col].dtype, pd.CategoricalDtype)
	}
	cat_cols = sorted(forced_cat | inferred_cat)
	num_cols = [col for col in X.columns if col not in cat_cols]
	return num_cols, cat_cols


def _build_preprocessor(
	X: pd.DataFrame,
	categorical_features: list[str] | None = None,
	scale_numeric: bool = False,
) -> ColumnTransformer:
	num_cols, cat_cols = _split_features(X, categorical_features=categorical_features)

	num_steps = [("imputer", SimpleImputer(strategy="median"))]
	if scale_numeric:
		num_steps.append(("scaler", StandardScaler(with_mean=False)))

	return ColumnTransformer(
		transformers=[
			(
				"num",
				Pipeline(num_steps),
				num_cols,
			),
			(
				"cat",
				Pipeline(
					[
						("imputer", SimpleImputer(strategy="most_frequent")),
						("onehot", OneHotEncoder(handle_unknown="ignore")),
					]
				),
				cat_cols,
			),
		],
		remainder="drop",
	)


def build_count_model(
	X: pd.DataFrame,
	categorical_features: list[str] | None = None,
	alpha: float = 0.0,
	max_iter: int = 3000,
) -> Pipeline:
	preprocessor = _build_preprocessor(
		X,
		categorical_features=categorical_features,
		scale_numeric=True,
	)
	model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
	return Pipeline([("prep", preprocessor), ("model", model)])


def build_mean_duration_model(
	X: pd.DataFrame,
	categorical_features: list[str] | None = None,
	alpha: float = 1.0,
) -> Pipeline:
	preprocessor = _build_preprocessor(
		X,
		categorical_features=categorical_features,
		scale_numeric=True,
	)
	model = Ridge(alpha=alpha, random_state=42)
	return Pipeline([("prep", preprocessor), ("model", model)])


def fit_count_model(
	df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str = "antall_samtaler",
	categorical_features: list[str] | None = None,
) -> Pipeline:
	X, y = _prepare_xy(df=df, feature_cols=feature_cols, target_col=target_col)
	if (y < 0).any():
		raise ValueError("Poisson-modell krever ikke-negative target-verdier.")

	model = build_count_model(X, categorical_features=categorical_features)
	model.fit(X, y)
	return model


def fit_mean_duration_model(
	df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str = "behandlingstid_snitt",
	categorical_features: list[str] | None = None,
) -> Pipeline:
	X, y = _prepare_xy(df=df, feature_cols=feature_cols, target_col=target_col)
	model = build_mean_duration_model(X, categorical_features=categorical_features)
	model.fit(X, y)
	return model


def predict_total_behandlingstid(
	count_model: Pipeline,
	duration_model: Pipeline,
	X: pd.DataFrame,
	min_pred: float = 1e-9,
) -> np.ndarray:
	pred_count = np.clip(count_model.predict(X), min_pred, None)
	pred_duration = np.clip(duration_model.predict(X), min_pred, None)
	return pred_count * pred_duration


def _summarize_cv(cv_result: dict[str, np.ndarray]) -> dict[str, float]:
	summary = {}
	for key, values in cv_result.items():
		if not key.startswith("test_"):
			continue
		metric = key.replace("test_", "")
		mean_value = float(np.mean(values))
		std_value = float(np.std(values))

		if metric.startswith("neg_"):
			metric = metric.replace("neg_", "")
			mean_value = -mean_value
			std_value = float(np.std(-values))

		summary[f"{metric}_mean"] = mean_value
		summary[f"{metric}_std"] = std_value

	return summary


def cross_validate_count_model(
	df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str = "antall_samtaler",
	categorical_features: list[str] | None = None,
	k: int = 10,
	random_state: int = 42,
) -> dict[str, float]:
	X, y = _prepare_xy(df=df, feature_cols=feature_cols, target_col=target_col)
	if (y < 0).any():
		raise ValueError("Poisson-modell krever ikke-negative target-verdier.")

	model = build_count_model(X, categorical_features=categorical_features)
	cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
	scoring = {
		"poisson_dev": make_scorer(mean_poisson_deviance, greater_is_better=False),
		"neg_mae": "neg_mean_absolute_error",
		"neg_rmse": "neg_root_mean_squared_error",
		"r2": "r2",
	}
	result = cross_validate(
		model,
		X,
		y,
		cv=cv,
		scoring=scoring,
		n_jobs=-1,
		error_score="raise",
	)
	summary = _summarize_cv(result)
	summary["poisson_dev_mean"] = -summary["poisson_dev_mean"]
	return summary


def cross_validate_mean_duration_model(
	df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str = "behandlingstid_snitt",
	categorical_features: list[str] | None = None,
	k: int = 10,
	random_state: int = 42,
) -> dict[str, float]:
	X, y = _prepare_xy(df=df, feature_cols=feature_cols, target_col=target_col)
	model = build_mean_duration_model(X, categorical_features=categorical_features)
	cv = KFold(n_splits=k, shuffle=True, random_state=random_state)

	result = cross_validate(
		model,
		X,
		y,
		cv=cv,
		scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
		n_jobs=-1,
		error_score="raise",
	)
	return _summarize_cv(result)


def cross_validate_total_product_model(
	df: pd.DataFrame,
	feature_cols: list[str],
	count_target_col: str = "antall_samtaler",
	duration_target_col: str = "behandlingstid_snitt",
	total_target_col: str = "total_behandlingstid",
	categorical_features: list[str] | None = None,
	k: int = 10,
	random_state: int = 42,
) -> dict[str, float]:
	if total_target_col not in df.columns:
		raise ValueError(f"Mangler target-kolonne i df: {total_target_col}")

	feature_cols = _resolve_feature_columns(df, feature_cols)
	frame = df[feature_cols + [count_target_col, duration_target_col, total_target_col]].dropna().copy()

	X = frame[feature_cols]
	y_count = frame[count_target_col].astype(float)
	y_duration = frame[duration_target_col].astype(float)
	y_total = frame[total_target_col].astype(float)

	if (y_count < 0).any():
		raise ValueError("Poisson-modell krever ikke-negative target-verdier.")

	cv = KFold(n_splits=k, shuffle=True, random_state=random_state)
	fold_rows = []

	for train_idx, test_idx in cv.split(X):
		X_train = X.iloc[train_idx]
		X_test = X.iloc[test_idx]

		y_count_train = y_count.iloc[train_idx]
		y_duration_train = y_duration.iloc[train_idx]
		y_total_test = y_total.iloc[test_idx]

		count_model = build_count_model(X_train, categorical_features=categorical_features)
		duration_model = build_mean_duration_model(X_train, categorical_features=categorical_features)

		count_model.fit(X_train, y_count_train)
		duration_model.fit(X_train, y_duration_train)

		pred_total = predict_total_behandlingstid(
			count_model=count_model,
			duration_model=duration_model,
			X=X_test,
		)

		fold_rows.append(
			{
				"mae": mean_absolute_error(y_total_test, pred_total),
				"rmse": np.sqrt(mean_squared_error(y_total_test, pred_total)),
				"r2": r2_score(y_total_test, pred_total),
			}
		)

	fold_df = pd.DataFrame(fold_rows)
	return {
		"mae_mean": float(fold_df["mae"].mean()),
		"mae_std": float(fold_df["mae"].std()),
		"rmse_mean": float(fold_df["rmse"].mean()),
		"rmse_std": float(fold_df["rmse"].std()),
		"r2_mean": float(fold_df["r2"].mean()),
		"r2_std": float(fold_df["r2"].std()),
	}


def fit_total_product_models(
	df: pd.DataFrame,
	feature_cols: list[str],
	count_target_col: str = "antall_samtaler",
	duration_target_col: str = "behandlingstid_snitt",
	categorical_features: list[str] | None = None,
) -> tuple[Pipeline, Pipeline]:
	feature_cols = _resolve_feature_columns(df, feature_cols)
	frame = df[feature_cols + [count_target_col, duration_target_col]].dropna().copy()

	X = frame[feature_cols]
	y_count = frame[count_target_col].astype(float)
	y_duration = frame[duration_target_col].astype(float)

	if (y_count < 0).any():
		raise ValueError("Poisson-modell krever ikke-negative target-verdier.")

	count_model = build_count_model(X, categorical_features=categorical_features)
	duration_model = build_mean_duration_model(X, categorical_features=categorical_features)

	count_model.fit(X, y_count)
	duration_model.fit(X, y_duration)
	return count_model, duration_model


def predict_total_from_df(
	df: pd.DataFrame,
	feature_cols: list[str],
	count_model: Pipeline,
	duration_model: Pipeline,
) -> pd.Series:
	feature_cols = _resolve_feature_columns(df, feature_cols)
	X = df[feature_cols].copy()
	pred = predict_total_behandlingstid(
		count_model=count_model,
		duration_model=duration_model,
		X=X,
	)
	return pd.Series(pred, index=df.index, name="pred_total_behandlingstid")
