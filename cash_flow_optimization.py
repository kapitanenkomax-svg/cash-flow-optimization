import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Анализ клиентов ----------
clients = pd.read_csv('clients_data.csv')

# Определяем критерии проблемности
clients['late_more_than_50_days'] = clients['avg_days_to_pay'] > 50
clients['less_than_8_on_time'] = clients['last_12_payments_on_time'] < 8
clients['is_problematic'] = clients['late_more_than_50_days'] & clients['less_than_8_on_time']

# Ранжируем по средней задержке (от самых медленных)
clients_sorted = clients.sort_values(by='avg_days_to_pay', ascending=False).reset_index(drop=True)

# Берём топ 20% проблемных (минимум 1 клиент)
n_problem = max(1, int(np.ceil(len(clients_sorted) * 0.2)))
problematic_df = clients_sorted.head(n_problem).copy()

# Формируем обоснование
def generate_reason(row):
    reasons = []
    if row['late_more_than_50_days']:
        reasons.append(f"ср. просрочка {int(row['avg_days_to_pay'])} дн.")
    if row['less_than_8_on_time']:
        reasons.append(f"только {int(row['last_12_payments_on_time'])} из 12 платежей вовремя")
    return "; ".join(reasons)

problematic_df['reason'] = problematic_df.apply(generate_reason, axis=1)
problematic_df['recommendation'] = 'Ввести 100% предоплату'

# Сохраняем в отдельный CSV
output_cols = [
    'client_id', 'client_name', 'annual_revenue',
    'avg_days_to_pay', 'last_12_payments_on_time',
    'reason', 'recommendation'
]
problematic_df[output_cols].to_csv('problematic_clients.csv', index=False, encoding='utf-8-sig')

print("✅ Сохранён файл problematic_clients.csv с обоснованием для работы с клиентами.")

# ---------- 2. Моделирование кассового потока ----------
weeks = 13
np.random.seed(42)

weekly_revenue = np.linspace(12_000_000, 18_000_000, weeks)
weekly_expenses = np.linspace(10_000_000, 13_000_000, weeks)

# До оптимизации: случайные поступления 60–80% от выручки
cash_in_before = weekly_revenue * np.random.uniform(0.6, 0.8, weeks)

# После: 20% выручки — мгновенно (предоплата), остальное — лучше сбор (+10%)
problematic_share = 0.20
cash_in_after = (
    weekly_revenue * problematic_share * 1.0 +
    weekly_revenue * (1 - problematic_share) * (np.random.uniform(0.7, 0.9, weeks))
)

cash_out = weekly_expenses

df = pd.DataFrame({
    'week': range(1, weeks + 1),
    'cash_in_before': cash_in_before,
    'cash_in_after': cash_in_after,
    'cash_out': cash_out
})
df['gap_before'] = df['cash_in_before'] - df['cash_out']
df['gap_after'] = df['cash_in_after'] - df['cash_out']

# ---------- 3. Расчёт эффекта ----------
avg_gap_before = df[df['gap_before'] < 0]['gap_before'].mean()
avg_gap_after = df[df['gap_after'] < 0]['gap_after'].mean()

print(f"\n📉 Средний кассовый разрыв ДО: {avg_gap_before:,.0f} ₽")
print(f"📈 Средний кассовый разрыв ПОСЛЕ: {avg_gap_after:,.0f} ₽")
reduction = 100 * (1 - abs(avg_gap_after) / abs(avg_gap_before))
print(f"📉 Снижение разрыва: {reduction:.0f}%")

# Оценка экономии
annual_weeks = 52
overdraft_reduction = abs(avg_gap_before - avg_gap_after) * annual_weeks
interest_rate = 0.067
interest_saving = overdraft_reduction * interest_rate
print(f"💰 Годовая экономия на процентах: ~{interest_saving:,.0f} ₽")

# ---------- 4. Визуализация ----------
plt.figure(figsize=(12, 6))
plt.plot(df['week'], df['gap_before'], label='До оптимизации', marker='o')
plt.plot(df['week'], df['gap_after'], label='После оптимизации', marker='s')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Оптимизация денежного потока: эффект от работы с проблемными клиентами')
plt.xlabel('Неделя')
plt.ylabel('Кассовый разрыв (₽)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cash_flow_optimized.png', dpi=150)
plt.show()