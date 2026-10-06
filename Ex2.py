import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import f1_score

# 1. Фиксируем списки дат (хронологическое разделение)
optuna_train_dates = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', 
                      '2026-01-05', '2026-01-06', '2026-01-07']  # 7 дат для обучения Optuna
val_dates = ['2026-01-08', '2026-01-09']                         # 2 даты для валидации Optuna

test_dates = ['2026-01-10', '2026-01-11']                        # 2 даты для финального теста

full_train_dates = optuna_train_dates + val_dates              # Все 9 дат вместе

# 2. Формируем выборки для Optuna и финального теста
df_optuna_train = df[df['DATE_DATE'].isin(optuna_train_dates)].sort_values('DATE_DATE')
df_val = df[df['DATE_DATE'].isin(val_dates)].sort_values('DATE_DATE')

df_full_train = df[df['DATE_DATE'].isin(full_train_dates)].sort_values('DATE_DATE')
df_test = df[df['DATE_DATE'].isin(test_dates)].sort_values('DATE_DATE')

# Подготовка фичей и таргетов
X_tr_opt = df_optuna_train.drop(columns=['target', 'DATE_DATE'])
y_tr_opt = df_optuna_train['target']

X_val = df_val.drop(columns=['target', 'DATE_DATE'])
y_val = df_val['target']

X_train_full = df_full_train.drop(columns=['target', 'DATE_DATE'])
y_train_full = df_full_train['target']

X_test = df_test.drop(columns=['target', 'DATE_DATE'])
y_test = df_test['target']

# 3. Optuna objective с фиксированным Train / Validation сплитом
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

    model = lgb.LGBMClassifier(**params)
    
    # Обучаем на 7 датах, валидируем на 2 датах для Optuna
    model.fit(
        X_tr_opt, y_tr_opt,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='binary')

# 4. Поиск лучших параметров
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# 5. Финальное обучение на ВСЕХ 9 датах и проверка на 2 датах теста
best_params = study.best_params
best_params.update({'objective': 'binary', 'verbosity': -1, 'random_state': 42})

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_full, y_train_full)

test_preds = final_model.predict(X_test)
print(f"Финальный F1-score на 2 тестовых датах: {f1_score(y_test, test_preds, average='binary'):.4f}")
