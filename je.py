import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import optuna

# Отключаем лишние предупреждения Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Фиксируем списки дат (хронологическое разделение)
optuna_train_dates = ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', 
                      '2026-01-05', '2026-01-06', '2026-01-07']  # 7 дат для обучения Optuna
val_dates = ['2026-01-08', '2026-01-09']                         # 2 даты для валидации Optuna
test_dates = ['2026-01-10', '2026-01-11']                        # 2 даты для финального теста

full_train_dates = optuna_train_dates + val_dates              # Все 9 дат вместе

# 2. Формируем выборки
df_optuna_train = df[df['DATE_DATE'].isin(optuna_train_dates)].sort_values('DATE_DATE')
df_val = df[df['DATE_DATE'].isin(val_dates)].sort_values('DATE_DATE')

df_full_train = df[df['DATE_DATE'].isin(full_train_dates)].sort_values('DATE_DATE')
df_test = df[df['DATE_DATE'].isin(test_dates)].sort_values('DATE_DATE')

X_tr_opt = df_optuna_train.drop(columns=['target', 'DATE_DATE'])
y_tr_opt = df_optuna_train['target']

X_val = df_val.drop(columns=['target', 'DATE_DATE'])
y_val = df_val['target']

X_train_full = df_full_train.drop(columns=['target', 'DATE_DATE'])
y_train_full = df_full_train['target']

X_test = df_test.drop(columns=['target', 'DATE_DATE'])
y_test = df_test['target']

# Важно для логистической регрессии: масштабирование признаков
scaler = StandardScaler()
X_tr_opt_scaled = scaler.fit_transform(X_tr_opt)
X_val_scaled = scaler.transform(X_val)
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# 3. Optuna objective для Логистической регрессии
def objective(trial):
    params = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'saga',  # поддерживает и l1, и l2 регуляризацию
        'max_iter': 1000,
        'random_state': 42
    }

    model = LogisticRegression(**params)
    model.fit(X_tr_opt_scaled, y_tr_opt)

    preds = model.predict(X_val_scaled)
    return f1_score(y_val, preds, average='binary')

# 4. Поиск лучших параметров
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# 5. Финальное обучение на ВСЕХ 9 датах и проверка на 2 датах теста
best_params = study.best_params
best_params.update({'solver': 'saga', 'max_iter': 1000, 'random_state': 42})

final_model = LogisticRegression(**best_params)
final_model.fit(X_train_full_scaled, y_train_full)

test_preds = final_model.predict(X_test_scaled)
print(f"Финальный F1-score логистической регрессии на 2 тестовых датах: {f1_score(y_test, test_preds, average='binary'):.4f}")

