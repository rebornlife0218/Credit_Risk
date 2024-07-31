import pandas as pd
import numpy as np
import scipy.optimize as sco
from scipy.stats import norm


# df = pd.read_csv(r"C:\python-NTHU\Financial Risk Management\data_20011001.csv")
df = pd.read_csv(r"C:\python-NTHU\Financial Risk Management\data_20020701.csv")

def equations(Va, sigma_a, Rf, Ve, sigma_e, K, debug = False):      # Va: 代表資產的當前價格    Ve: 代表資產的預期未來價格  
    T = 1
    d1 = (np.log(Va / K) + (Rf + 0.5 * sigma_a ** 2) * T) / (sigma_a * np.sqrt(T))
    d2 = d1 - sigma_a * np.sqrt(T)
    y1 = (Ve - (Va * norm.cdf(d1) - np.exp(-Rf * T) * K * norm.cdf(d2))) ** 2
    y2 = (sigma_e - Va / Ve * norm.cdf(d1) * sigma_a) ** 2
    
    if debug:
        print("d1" + str(d1))
        print("d2" + str(d2))
        print("Error" + str(y1))
        
    return y1 + y2

def solve_for_asset_value(df):
    Debt = df['DLC'].values + 0.5 * df['DLTT'].values
    Va = df['me'].values + Debt
    Ve_ret = np.log(df['RET'].values + 1)       # RET 是股票報酬率的百分比
    td = len(Va)        # total number of trading days = td
    sigma_e = np.nanstd(Ve_ret) * np.sqrt(td)
    sigma_a = sigma_e
    
    # 對先前的 Va 值進行迭代
    for i in range(30):
        # 計算 daily Va
        for t in range(len(Va)):
            Ve = df['me'].values[t]
            K = Debt[t]
            Rf = df['ir'].values[t]
            sol = sco.minimize(equations, Va[t], args=(sigma_a, Rf, Ve, sigma_e, K))
            Va[t] = sol['x'][0]
            
        # 根據新的 Va_ret 值更新 sigma_a
        last_sigma_a = sigma_a
        Va_ret = np.log(Va[1 : ] / Va[ : -1])
        sigma_a = np.nanstd(Va_ret) * np.sqrt(td)
        mean_a = np.nanmean(Va_ret) * td
        
        print("No. " + str(i) + " iterate")
        print("last_sigma_a: " + str(last_sigma_a))
        print("sigma_a: " + str(sigma_a) + "\n")
        
        if abs(last_sigma_a - sigma_a) < 1e-3:      # 迭代過程將持續進行，直到資產波動率的變化很小（小於1e-3）為止，表示收斂到穩定的解。
            print("Converge!!!\n")
            break
            
    return Va, mean_a, sigma_a


# main function
Va, mean_a, sigma_a = solve_for_asset_value(df)

t = len(Va) - 1
Va = Va[t]
Ve = df['me'].values[t]
K = df['DLC'].values[t] + 0.5 * df['DLTT'].values[t]
Rf = df['ir'].values[t]

T = 1
d1 = (np.log(Va / K) + (Rf + 0.5 * sigma_a ** 2) * T) / (sigma_a * np.sqrt(T))
d2 = d1 - sigma_a * np.sqrt(T)

neg_d2 = d2 * (-1)
edf = norm.cdf(neg_d2)

print("negative d2: " + str(neg_d2))
print("edf: " + str(edf))
print("d1: " + str(d1))
print("d2: " + str(d2) + "\n" + "-" * 50)

print("Ve at the last day: " + str(Ve))
print("K: " + str(K))
print("Rf: " + str(Rf) + "\n" + "-" * 50)

print("Va at the last day: " + str(Va))
print("mean_a: " + str(mean_a))
print("sigma_a: " + str(sigma_a))
