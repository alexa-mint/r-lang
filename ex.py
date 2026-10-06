import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

# 1. Фиксируем списки дат
train_dates = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05', 
               '2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09'] # 9 дат для КВ
test_dates = ['2026-01-10', '2026-01-11']                             # 2 даты для финального теста

# 2. Формируем Train и Holdout-Test выборки
train_df = df[df['DATE_DATE'].isin(train_dates)].sort_values('DATE_DATE')
test_df = df[df['DATE_DATE'].isin(test_dates)].sort_values('DATE_DATE')

X_train_full = train_df.drop(columns=['target', 'DATE_DATE'])
y_train_full = train_df['target']

X_test = test_df.drop(columns=['target', 'DATE_DATE'])
y_test = test_df['target']

# 3. Подготовка фолдов КВ по уникальным датам (Expanding Window)
unique_train_dates = sorted(train_df['DATE_DATE'].unique())
tscv = TimeSeriesSplit(n_splits=4)  # Разбьет 9 дат на 4 последовательных фолда

cv_splits = []
for train_date_idx, val_date_idx in tscv.split(unique_train_dates):
    dates_tr = [unique_train_dates[i] for i in train_date_idx]
    dates_va = [unique_train_dates[i] for i in val_date_idx]
    
    tr_idx = train_df[train_df['DATE_DATE'].isin(dates_tr)].index
    va_idx = train_df[train_df['DATE_DATE'].isin(dates_va)].index
    cv_splits.append((tr_idx, va_idx))

# 4. Optuna objective с кросс-валидацией внутри 9 дат
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    scores = []
    
    # Кросс-валидация внутри 9 дат
    for tr_idx, va_idx in cv_splits:
        X_tr, y_tr = train_df.loc[tr_idx, X_train_full.columns], y_train_full.loc[tr_idx]
        X_va, y_va = train_df.loc[va_idx, X_train_full.columns], y_train_full.loc[va_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        preds = model.predict(X_va)
        scores.append(f1_score(y_va, preds, average='binary'))

    return np.mean(scores)  # Оптимизируем средний F1 по фолдам

# 5. Поиск лучших параметров
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# 6. Финальное обучение на ВСЕХ 9 датах и проверка на 2 датах теста
best_params = study.best_params
best_params.update({'objective': 'binary', 'verbosity': -1, 'random_state': 42})

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_full, y_train_full)

test_preds = final_model.predict(X_test)
print(f"Финальный F1-score на 2 тестовых датах: {f1_score(y_test, test_preds, average='binary'):.4f}")
