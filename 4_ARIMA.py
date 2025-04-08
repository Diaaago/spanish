"""
差分自回归移动平均模型（ARIMA）
"""

# %% 0. 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# %% 1. 生成示例时间序列数据
np.random.seed(42)
n = 200  # 数据点数量
x = np.linspace(0, 20, n)  # 生成从 0 到 20 的等间距数列，共 n 个点
y = 0.5 * x + np.random.normal(size=n)  # 带噪声的时间序列数据

# 转换为 DataFrame
df = pd.DataFrame({'y': y})

# 绘制原始数据
plt.figure(figsize=(8, 3))
plt.plot(df['y'], label='Original Data')
plt.title('Generated Time Series Data')
plt.legend()
plt.tight_layout()
plt.show()

# %% 2. 确定差分次数 d

# 定义一个辅助函数来执行 ADF 和 KPSS 检验
def check_stationarity(data):
    """
    使用 ADF 和 KPSS 检验数据的平稳性
    """
    # 删除 NaN 值
    data = data.dropna()

    # ADF 检验
    adf_result = adfuller(data)
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print(f'Critical Values: {adf_result[4]}')

    if adf_result[1] <= 0.05:
        print("ADF 检验结果: 数据平稳（拒绝原假设）。")
    else:
        print("ADF 检验结果: 数据非平稳（无法拒绝原假设）。")

    # KPSS 检验，捕获 InterpolationWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        kpss_result = kpss(data, regression='c')
        print(f'KPSS Statistic: {kpss_result[0]:.4f}')
        print(f'p-value: {kpss_result[1]:.4f}')
        print(f'Critical Values: {kpss_result[3]}')

    if kpss_result[1] > 0.05:
        print("KPSS 检验结果: 数据平稳（无法拒绝原假设）。")
    else:
        print("KPSS 检验结果: 数据非平稳（拒绝原假设）。")

# 初始数据的平稳性检验
print("原始数据的平稳性检验结果：")
check_stationarity(df['y'])

# 根据平稳性检验结果确定差分次数 d
d = 0
current_data = df['y']

while True:
    d += 1
    current_data = current_data.diff().dropna()  # 计算差分并删除 NaN 值

    # 执行 ADF 和 KPSS 检验
    adf_result = adfuller(current_data)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        kpss_result = kpss(current_data, regression='c')

    if adf_result[1] <= 0.05 and kpss_result[1] > 0.05:
        print(f"数据在差分 {d} 次后平稳。")
        break
    else:
        print(f"数据在差分 {d} 次后不平稳，继续差分...")


print(f"确定的差分次数 d 为: {d}")

# %% 3. 绘制 ACF 和 PACF 图以选择最佳阶数 p 和 q

# 绘制自相关函数（ACF）图
plot_acf(df['y'].diff(d).dropna(), lags=10)  # 使用差分后的数据绘制 ACF，降低滞后阶数到10
plt.title('Autocorrelation Function (ACF) After Differencing')
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()

# 绘制偏自相关函数（PACF）图，使用不同的方法避免奇异矩阵
plot_pacf(df['y'].diff(d).dropna(), lags=10, method='ols')  # 使用差分后的数据绘制 PACF，使用 'ols' 方法
plt.title('Partial Autocorrelation Function (PACF) After Differencing')
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()

# 根据 ACF 和 PACF 图自动选择最佳阶数 p 和 q
acf_values = acf(df['y'].diff(d).dropna(), nlags=10)
threshold_acf = 1.96 / np.sqrt(len(df['y'].diff(d).dropna()))  # 95% 置信区间

# 选择第一个 ACF 值低于阈值的位置作为最佳 q 阶数
acf_indices = np.where(np.abs(acf_values) < threshold_acf)[0]
if len(acf_indices) > 0:
    best_q_acf = acf_indices[0]  # 最佳q阶数
else:
    best_q_acf = len(acf_values) - 1  # 如果没有低于阈值的点，则选择最大滞后
print(f"Best MA Order by ACF: {best_q_acf}")

pacf_values = pacf(df['y'].diff(d).dropna(), nlags=10)
threshold_pacf = 1.96 / np.sqrt(len(df['y'].diff(d).dropna()))  # 95% 置信区间

# 选择第一个 PACF 值低于阈值的位置作为最佳 p 阶数
pacf_indices = np.where(np.abs(pacf_values) < threshold_pacf)[0]
if len(pacf_indices) > 0:
    best_p_pacf = pacf_indices[0]  # 最佳p阶数
else:
    best_p_pacf = len(pacf_values) - 1  # 如果没有低于阈值的点，则选择最大滞后
print(f"Best AR Order by PACF: {best_p_pacf}")

# %% 4. 使用 AIC 和 BIC 信息准则来选择最佳 p 和 q


# 计算不同 (p, q) 组合的 AIC 和 BIC，并选择同时最小的组合
def grid_search_arima_aic_bic_min_combined(df, max_p, max_q, d):
    """
    通过网格搜索计算不同 (p, q) 组合的 ARIMA 模型的 AIC 和 BIC 值，并选择同时最小的组合。
    """
    # 用于存储结果的列表
    results = []

    # 遍历所有可能的 (p, q) 组合
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                # 创建并拟合 ARIMA 模型
                model = ARIMA(df['y'], order=(p, d, q), enforce_stationarity=True, enforce_invertibility=True)
                model_fit = model.fit(
                    method_kwargs={'maxiter': 1000, 'disp': False}  # 优化参数
                )

                # 获取 AIC 和 BIC 值
                aic = model_fit.aic
                bic = model_fit.bic

                # 保存结果
                results.append((p, q, aic, bic))
                print(f"ARIMA(p={p}, d={d}, q={q}) - AIC: {aic:.2f}, BIC: {bic:.2f}")

            except Exception as e:
                print(f"ARIMA(p={p}, d={d}, q={q}) 训练失败: {e}")

    # 转换为 DataFrame 便于分析
    results_df = pd.DataFrame(results, columns=['p', 'q', 'AIC', 'BIC'])

    # 找到 AIC 和 BIC 同时最小的组合
    min_aic = results_df['AIC'].min()
    min_bic = results_df['BIC'].min()

    # 找出同时满足 AIC 和 BIC 最小的组合
    combined_min = results_df[(results_df['AIC'] == min_aic) & (results_df['BIC'] == min_bic)]

    if not combined_min.empty:
        # 如果存在同时满足 AIC 和 BIC 最小的组合
        best_combined_order = (combined_min.iloc[0]['p'], d, combined_min.iloc[0]['q'])
        print(
            f"\nBest ARIMA Order by AIC and BIC: (p={best_combined_order[0]}, d={best_combined_order[1]}, q={best_combined_order[2]})")
    else:
        # 如果不存在同时满足条件的组合，选择 AIC 和 BIC 平均值最小的组合
        results_df['AIC_BIC_Mean'] = results_df[['AIC', 'BIC']].mean(axis=1)
        best_index = results_df['AIC_BIC_Mean'].idxmin()
        best_combined_order = (results_df.loc[best_index, 'p'], d, results_df.loc[best_index, 'q'])
        print(
            f"\nBest ARIMA Order by Minimized Average of AIC and BIC: (p={best_combined_order[0]}, d={best_combined_order[1]}, q={best_combined_order[2]})")

    return best_combined_order

# %% 5. 使用 ARIMA 模型拟合并预测

# 使用 ARIMA 模型拟合并预测
def fit_predict_arima_model(p, d, q, df, steps=20):
    # 创建并拟合 ARIMA 模型
    try:
        # 设置模型，增加 enforce_stationarity 和 enforce_invertibility 参数
        model = ARIMA(df['y'], order=(p, d, q), enforce_stationarity=True, enforce_invertibility=True)

        # 设置 fit 方法中的优化参数，通过 method_kwargs 指定最大迭代次数
        model_fit = model.fit(
            method_kwargs={
                'maxiter': 1000,  # 增加最大迭代次数
                'disp': False  # 禁止优化输出
            }
        )

        # 检查拟合结果的收敛状态
        if not model_fit.mle_retvals['converged']:
            print(f"ARIMA 模型 (p={p}, d={d}, q={q}) 的最大似然优化未能收敛。请检查模型参数和数据。")
            return [np.nan] * steps

        # 预测未来的值
        predictions = model_fit.forecast(steps=steps)
        return predictions

    except Exception as e:
        print(f"ARIMA 模型 (p={p}, d={d}, q={q}) 训练失败: {e}")
        return [np.nan] * steps


# 选择合适的 p, d, q 阶数
best_p = best_p_pacf  # 根据 PACF 选择最佳 p
best_q = best_q_acf  # 根据 ACF 选择最佳 q
# 使用 ARIMA 进行预测
predictions_arima_acf_pacf = fit_predict_arima_model(best_p_pacf, d, best_q, df)


max_p = 3  # 最大 p 阶数
max_q = 3  # 最大 q 阶数

# 执行网格搜索
best_aic_bic_order = grid_search_arima_aic_bic_min_combined(df, max_p, max_q, d)

# 使用网格搜索得到的最佳 (p, d, q) 组合进行预测
predictions_arima_aic_bic = fit_predict_arima_model(best_aic_bic_order[0], best_aic_bic_order[1], best_aic_bic_order[2],
                                                    df)

# %% 6. 绘制不同方法下的预测结果对比图

plt.figure(figsize=(8, 4))
plt.plot(df['y'], label='Original Data')
plt.plot(np.arange(len(df), len(df) + len(predictions_arima_acf_pacf)), predictions_arima_acf_pacf,
         label=f'ARIMA Predictions (p={best_p}, d={d}, q={best_q})', linestyle='--')
plt.plot(np.arange(len(df), len(df) + len(predictions_arima_aic_bic)), predictions_arima_aic_bic,
         label=f'ARIMA Predictions (p={best_aic_bic_order[0]}, d={d}, q={best_aic_bic_order[1]})', linestyle='-.')

plt.title('ARIMA Model Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# %% 7. 计算并打印各个模型的预测性能（MSE）

def compute_mse(true_values, predictions, label):
    """
    计算MSE并检查输入数据是否为空的辅助函数。
    """
    if len(true_values) == 0 or len(predictions) == 0:
        print(f"No valid samples available to compute MSE for {label}.")
        return np.nan  # 返回 NaN 表示无法计算
    return mean_squared_error(true_values, predictions)

# 清理预测结果中的 NaN 值
def clean_nan(predictions, true_values):
    valid_idx = ~np.isnan(predictions) & ~np.isnan(true_values)
    return predictions[valid_idx], true_values[valid_idx]

true_values_arima_acf_pacf = df['y'].iloc[-len(predictions_arima_acf_pacf):].values
true_values_arima_aic_bic = df['y'].iloc[-len(predictions_arima_aic_bic):].values


predictions_arima_acf_pacf, true_values_arima_acf_pacf = clean_nan(np.array(predictions_arima_acf_pacf), true_values_arima_acf_pacf)
predictions_arima_aic, true_values_arima_aic = clean_nan(np.array(predictions_arima_aic_bic), true_values_arima_aic_bic)


mse_arima_acf_pacf = compute_mse(true_values_arima_acf_pacf, predictions_arima_acf_pacf, label='ARIMA model with ACF order and PACF order')
mse_arima_aic_bic = compute_mse(true_values_arima_aic, predictions_arima_aic, label='ARIMA model with AIC order')

print(f'MSE for ARIMA model with ACF order and PACF order: {mse_arima_acf_pacf:.2f}')
print(f'MSE for ARIMA model with AIC order and BIC order: {mse_arima_aic_bic:.2f}')
