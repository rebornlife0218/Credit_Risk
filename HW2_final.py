import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.stats import norm


df = pd.read_excel("C:\python-NTHU\Financial Risk Management\data_figure.xls")

def equations(Va, sigma_a, Rf, Ve, sigma_e, K, debug = False):      # Va: 代表資產的當前價格    Ve: 代表資產的預期未來價格
    T = 1
    d1 = (np.log(Va/K) + (Rf + 0.5 * sigma_a ** 2) * T) / (sigma_a * np.sqrt(T))
    d2 = d1 - sigma_a * np.sqrt(T)
    y1 = (Ve - (Va * norm.cdf(d1) - np.exp(-Rf * T) * K * norm.cdf(d2))) ** 2
    y2 = (sigma_e - Va/Ve * norm.cdf(d1) * sigma_a) ** 2
    return y1 + y2

def solve_for_asset_value(df):
    Debt = df['DLC'].values + 0.5 * df['DLTT'].values
    Va = df['me'].values + Debt
    Ve_ret = np.log(df['RET'].values + 1)
    td = len(Va)
    sigma_e = np.nanstd(Ve_ret) * np.sqrt(td)
    sigma_a = sigma_e

    # 對先前的 Va 值進行迭代
    for i in range(30):     # 開始一個循環，最多迭代 30 次。
        # 計算 daily Va
        for t in range(len(Va)):
            Ve = df['me'].values[t]
            K = Debt[t]
            Rf = df['ir'].values[t]
            sol = sco.minimize(equations, Va[t], args=(sigma_a, Rf, Ve, sigma_e, K))    # 最小化 equations 函數 (即 Black-Scholes-Merton 模型的誤差函數)，求解當前時間的資產價值 (Va)。
            Va[t] = sol['x'][0]
        
        # 根據新的 Va_ret 值更新 sigma_a
        last_sigma_a = sigma_a
        Va_ret = np.log(Va[1:] / Va[:-1])
        sigma_a = np.nanstd(Va_ret) * np.sqrt(td)
        mean_a = np.nanmean(Va_ret) * td

        if abs(last_sigma_a - sigma_a) < 1e-3:
            # print("Converge!!!\n")
            break
    return Va, mean_a, sigma_a

# 將 ym 轉換成日期格式
df['ym'] = pd.to_datetime(df['ym'], format='%Y%m')
window_size = 12

# 初始化起始年月
start_year = 1998
start_month = 1

# 初始化結束年月
end_year = 1998
end_month = 12

result = {}
for i in range(1, 43):
    start_date = pd.Timestamp(start_year, start_month, 1)
    end_date = pd.Timestamp(end_year, end_month, 1) + pd.DateOffset(months=1) - pd.DateOffset(days=1)       # 要當月的最後一天
    group_key = start_date.strftime('%Y%m') + '-' + end_date.strftime('%Y%m')
    result[group_key] = df[(df['ym'] >= start_date) & (df['ym'] <= end_date)]       # 篩選出 df 中日期 (ym) 屬於當前月份區間的資料
    
    # 更新起始年月和結束年月
    start_month += 1
    end_month += 1

    if start_month > 12:
        start_month = 1
        start_year += 1

    if end_month > 12:
        end_month = 1
        end_year += 1


result_processed = {}
for key, group_df in result.items():        # 遍歷 result 字典中的每個元素
    # 執行函數
    Va, mean_a, sigma_a = solve_for_asset_value(group_df)
    
    # 計算 d1, d2, edf(expected default probability)
    t = len(Va) - 1
    Va = Va[t]
    Ve = group_df['me'].values[t]
    K = group_df['DLC'].values[t] + 0.5 * group_df['DLTT'].values[t]
    Rf = group_df['ir'].values[t]
    T = 1
    d1 = (np.log(Va/K) + (Rf + 0.5 * sigma_a ** 2) * T) / (sigma_a * np.sqrt(T))
    d2 = d1 - sigma_a * np.sqrt(T)
    neg_d2 = d2 * (-1)
    edf = norm.cdf(neg_d2)
    
    # 將結果存到字典中
    result_processed[key] = {
        'Va': Va,
        'mean_a': mean_a,
        'sigma_a': sigma_a,
        'd1': d1,
        'd2': d2,
        'edf': edf
    }

# 將字典轉成 DataFrame
result_df = pd.DataFrame(result_processed)
print(result_df)

# 提取每個時間點的 edf 值
edf_values = result_df.loc['edf']

# 設置 x 軸和 y 軸的數據
start_date = pd.Timestamp(1998, 12, 1)  
end_date = pd.Timestamp(2002, 6, 1)
x_values = pd.date_range(start=start_date, end=end_date, freq='M')
y_values = edf_values.values

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.title('EDF over time')
plt.xlabel('Time')
plt.ylabel('EDF')
plt.grid(True)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
