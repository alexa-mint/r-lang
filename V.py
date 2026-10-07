import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(12, 5))

# Рисуем первый ряд на основной оси
line1 = ax1.plot(df['feature_1'], color='tab:blue', label='Признак 1', alpha=0.8)

# Создаем вторую ось Y с тем же X и рисуем второй ряд
ax2 = ax1.twinx()
line2 = ax2.plot(df['feature_2'], color='tab:orange', label='Признак 2', alpha=0.8)

# Полностью скрываем оси Y и их подписи/засечки
ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

# Настройка легенды для двух разных осей
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Сравнение формы рядов без предварительной нормализации")
plt.tight_layout()
plt.show()
