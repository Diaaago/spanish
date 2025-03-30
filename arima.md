```python
# -*- coding: utf-8 -*-
"""
ARIMA Modeling Example
- English & Chinese Comments

步骤 / Steps:
1) 从CSV读取数据并可视化折线图
2) ADF单位根检验，若不平稳则进行差分
3) 查看ACF和PACF图来辅助选择p,q(最大3)
4) 构建ARIMA模型并评估结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")  # 忽略一些模型警告 / Ignore some model warnings

# ---------- Helper Functions 辅助函数 ----------

def adf_test(series, signif=0.05, series_name=""):
    """
    Conduct ADF test to check stationarity.
    执行ADF单位根检验来检测序列的平稳性。

    :param series: 时间序列数据 / The time series data
    :param signif: 显著性水平 / Significance level
    :param series_name: 序列名称，用于打印 / Name of the series for print
    :return: (is_stationary, p_value, used_diff)
    """
    r = adfuller(series.dropna(), autolag='AIC')
    p_value = r[1]
    print(f"\n[ADF Test for {series_name}]")
    print("ADF Statistic:", r[0])
    print("p-value:", p_value)
    for key, val in r[4].items():
        print("Critical Values:")
        print(f"   {key}, {val}")

    # 如果p值小于signif，就认为是平稳的
    # If p-value < signif, we consider it stationary
    is_stationary = p_value < signif
    return is_stationary, p_value


def difference_if_needed(series, max_diff=2):
    """
    Check stationarity, if not stationary, do differencing up to max_diff times.
    检查平稳性，如果不平稳，则最多进行 max_diff 次差分。

    :param series: 原始序列 / The original time series
    :param max_diff: 最大差分次数 / Maximum number of differences
    :return: (final_series, d) 返回差分后的序列及实际差分次数
    """
    d = 0
    temp_series = series.copy()
    for i in range(max_diff+1):
        stationary, _ = adf_test(temp_series, series_name=f"d={i}")
        if stationary:
            return temp_series, i
        else:
            # 如果还不平稳，就差分一次
            temp_series = temp_series.diff().dropna()
            d += 1

    # 如果到达max_diff还不平稳，就返回最后的差分结果
    return temp_series, d


def find_best_arima_order(series, d, p_range=3, q_range=3):
    """
    Find the best ARIMA(p, d, q) by AIC within given range.
    在给定范围内，通过AIC选择最优ARIMA(p,d,q)。

    :param series: 时间序列（已差分或原序列） / Time series data (differenced or original)
    :param d: 差分阶数 / Differencing order
    :param p_range: p的最大值 / Max p
    :param q_range: q的最大值 / Max q
    :return: (best_order, best_model)
    """
    best_aic = np.inf
    best_order = (0, d, 0)
    best_model = None

    # 在 0~p_range, 0~q_range 上搜索
    # Search p in [0..p_range], q in [0..q_range]
    for p in range(p_range+1):
        for q in range(q_range+1):
            try:
                model = ARIMA(series, order=(p, d, q)).fit(method='least_squares')
                aic = model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = model
            except:
                continue
    return best_order, best_model


def plot_series(df, columns):
    """
    Plot line charts for the given columns.
    对给定列画折线图。
    """
    df[columns].plot(subplots=True, figsize=(10, 6), layout=(len(columns), 1), sharex=True)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series, lags=20, title_prefix=""):
    """
    Plot ACF and PACF for a series.
    为序列绘制 ACF 和 PACF。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=axes[0], lags=lags)
    plot_pacf(series.dropna(), ax=axes[1], lags=lags)
    axes[0].set_title(f"{title_prefix} - ACF")
    axes[1].set_title(f"{title_prefix} - PACF")
    plt.show()


# ---------- Main Logic 主流程 ----------

def main():
    # 1) 读取CSV / Read CSV
    df = pd.read_csv("E:/TFM/data/temp_hum_solar_mean_table.csv", parse_dates=["datetime"])
    # 将 datetime 设置为索引 / Set datetime as index
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)  # 确保按照时间排序 / Ensure sorting by time

    # 只选择我们关心的三列 / Select only the three columns of interest
    columns_of_interest = ["temperature(oC)", "humidity(%)", "solar_radiation(W/m2)"]
    # 如果列名不同，请自行修改 / If column names differ, adjust accordingly

    # 2) 折线图可视化 / Plot the line chart
    plot_series(df, columns_of_interest)

    # 针对每个变量进行ARIMA拟合 / Fit ARIMA for each column
    results = {}
    for col in columns_of_interest:
        print(f"\n{'='*30}\nProcessing Column: {col}")

        # ADF检验，如果不平稳则进行差分 / ADF test, difference if needed
        series = df[col].dropna()
        if series.empty:
            print(f"[WARNING] Column {col} has no data. Skip.")
            continue

        # 2) ADF + 差分 / ADF + differencing
        diffed_series, d = difference_if_needed(series)

        # 3) 绘制差分后或原序列的ACF和PACF，p、q 最大为3
        #    Plot ACF/PACF for differenced or original series
        plot_acf_pacf(diffed_series, lags=20, title_prefix=col)

        # 4) 搜索最佳(p,d,q)，并进行ARIMA拟合 / Search best (p,d,q), fit ARIMA
        best_order, best_model = find_best_arima_order(series, d=d, p_range=3, q_range=3)
        print(f"[INFO] Best ARIMA order for {col}: {best_order} with AIC={best_model.aic:.2f}")

        # 输出模型摘要 / Print model summary
        print(best_model.summary())

        results[col] = {
            "best_order": best_order,
            "model": best_model
        }

    # 如果需要进一步的预测或可视化，可以在这里进行
    # If you need further forecasting or visualization, you can proceed here.

    print("\nAll done!")


if __name__ == "__main__":
    main()

```

以下是对给定代码的完整、详细解释：

------

## 1. 概述

这段代码的核心目标是对时间序列数据使用ARIMA模型进行建模预测分析，具体过程包括：

- **读取数据并可视化**
- **进行单位根检验（ADF）以检测序列平稳性**
- **根据需要进行差分以达到平稳性**
- **绘制自相关函数 (ACF) 和偏自相关函数 (PACF) 图**
- **自动确定最佳的ARIMA(p,d,q)参数**
- **拟合ARIMA模型并展示结果**

------

## 2. 导入库及初始设置

```
python复制编辑import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")  
# 忽略统计模型的警告信息，以避免大量输出干扰。
```

------

## 3. 辅助函数解析

### 3.1 `adf_test`

- **功能：** 使用ADF单位根检验检测序列是否平稳。
- **输入参数：**
  - `series`: 待检测的序列。
  - `signif`: 显著性水平，默认为0.05。
  - `series_name`: 序列名，仅用于打印。
- **返回值：**
  - `is_stationary`: 是否平稳（布尔值）。
  - `p_value`: 检验得到的p值。

> ADF检验（Augmented Dickey-Fuller Test）用于判断序列是否存在单位根，若序列存在单位根则非平稳，需要进行差分。

------

### 3.2 `difference_if_needed`

- **功能：** 检测序列平稳性并根据需要差分（最多2次）。
- **流程：**
  1. 初始对原始序列做ADF检验。
  2. 若不平稳，进行一次差分后再进行ADF检验。
  3. 若仍不平稳，可再差分一次（最多两次差分）。
- **返回值：**
  - `final_series`: 最终差分后的序列。
  - `d`: 实际差分次数。

------

### 3.3 `find_best_arima_order`

- **功能：** 自动在给定范围内，通过AIC值来确定最佳的ARIMA(p,d,q)模型。
- **输入参数：**
  - `series`: 时间序列数据。
  - `d`: 差分次数。
  - `p_range` 和 `q_range`: p 和 q 参数搜索的最大范围（默认3）。
- **流程：**
  - 使用双重循环遍历所有可能的 `(p, d, q)` 组合。
  - 拟合模型并计算每个模型的AIC值，记录AIC最低的模型为最佳模型。
- **返回值：**
  - 最佳模型的 `(p,d,q)` 参数组合及拟合的模型对象。

------

### 3.4 `plot_series`

- **功能：** 绘制序列的折线图用于初步的视觉分析。
- **输入参数：**
  - `df`: 数据框。
  - `columns`: 要绘制的列名列表。

------

### 3.5 `plot_acf_pacf`

- **功能：** 绘制ACF（自相关）和PACF（偏自相关）图，用于确定模型阶数 p 和 q。
- **输入参数：**
  - `series`: 时间序列数据。
  - `lags`: 展示延迟期数（默认20）。
  - `title_prefix`: 图标题前缀，用于区分不同变量。

------

## 4. 主程序逻辑分析 (`main`函数)

### 4.1 读取并预处理数据

```
python复制编辑df = pd.read_csv("...csv", parse_dates=["datetime"])
df.set_index("datetime", inplace=True)
df.sort_index(inplace=True)
```

- 从CSV文件读取时间序列数据，设置日期为索引，并进行排序。

### 4.2 可视化数据

```
python


复制编辑
plot_series(df, columns_of_interest)
```

- 画出各序列的折线图，初步观察趋势、季节性、周期性和波动情况。

### 4.3 针对每个变量逐一进行ARIMA建模：

针对每一个感兴趣的列：

#### （a）检查平稳性及差分

```
python


复制编辑
diffed_series, d = difference_if_needed(series)
```

- 检测数据是否平稳，如非平稳则差分（最多2次）。

#### （b）绘制ACF和PACF图

```
python


复制编辑
plot_acf_pacf(diffed_series, lags=20, title_prefix=col)
```

- ACF图帮助确定参数q（移动平均阶数），PACF图帮助确定参数p（自回归阶数）。

#### （c）自动搜索并拟合最佳ARIMA模型

```
python


复制编辑
best_order, best_model = find_best_arima_order(series, d=d, p_range=3, q_range=3)
```

- 根据AIC值自动确定最佳的(p,d,q)。

#### （d）模型评估与展示结果

```
python


复制编辑
print(best_model.summary())
```

- 输出ARIMA模型的详细摘要信息（参数、显著性、AIC、BIC等）。

------

## 5. 代码实际用途

此代码通常应用于：

- 气象数据（如示例中使用的温度、湿度、太阳辐射数据）
- 金融市场分析（股票价格、汇率预测）
- 其他任何涉及历史数据预测的领域（能源消耗、需求预测等）

------

## 6. 补充概念说明

- **ARIMA(p,d,q)**：
  - **p** (AR项)：序列历史值的回归项。
  - **d** (差分阶数)：使序列平稳的差分次数。
  - **q** (MA项)：移动平均项，即过去误差的影响。
- **AIC (赤池信息准则)**：
  - 衡量模型拟合质量与复杂度的指标，越低模型越好。
- **ADF检验**：
  - 检测单位根的存在性，即检测序列是否平稳。

------

## 7. 后续扩展与预测

该代码可以轻松扩展以进行：

- 未来预测（如`model.forecast()`或`model.predict()`）。
- 模型评估（如交叉验证或滚动窗口验证）。
- 将预测结果与实际结果对比并可视化展示。





```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: 读取 CSV 文件并绘制折线图 / Read CSV file and plot line charts
file_path = 'E:/TFM/data/temp_hum_solar_mean_table.csv'  # 请替换为你的 CSV 文件路径 / Please replace with your CSV file path
data = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# 绘制折线图展示数据趋势 / Plot line charts to show data trends
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(data['temperature(oC)'], label='Temperature (oC)')
plt.title('温度 / Temperature')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(data['humidity(%)'], label='Humidity (%)')
plt.title('湿度 / Humidity')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(data['solar_radiation(W/m2)'], label='Solar Radiation (W/m2)')
plt.title('太阳辐射 / Solar Radiation')
plt.legend()
plt.tight_layout()
plt.show()

# Step 2: ADF 稳定性检测 / ADF test for stationarity
def adf_test(series):
    """
    检查时间序列是否平稳 / Check if the time series is stationary
    返回值 / Return: True (平稳) 或 False (不平稳) / True (stationary) or False (non-stationary)
    """
    result = adfuller(series.dropna())  # 去掉缺失值 / Drop missing values
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("数据不平稳 / Data is not stationary")
        return False
    else:
        print("数据平稳 / Data is stationary")
        return True

# Step 3: ACF 和 PACF 分析 / ACF and PACF analysis
def plot_acf_pacf(series, lags=20):
    """
    绘制 ACF 和 PACF 图以确定 p 和 q / Plot ACF and PACF to determine p and q
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    acf_vals = acf(series, nlags=lags)
    plt.bar(range(len(acf_vals)), acf_vals)
    plt.title('ACF')
    plt.subplot(1, 2, 2)
    pacf_vals = pacf(series, nlags=lags)
    plt.bar(range(len(pacf_vals)), pacf_vals)
    plt.title('PACF')
    plt.show()

# Step 4: ARIMA 模型拟合 / ARIMA model fitting
def fit_arima(series, p, d, q):
    """
    使用 ARIMA 模型拟合数据 / Fit ARIMA model to the data
    使用最小二乘法 (OLS) / Use Ordinary Least Squares (OLS)
    """
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()  # 默认使用 OLS / Default uses OLS
    print(model_fit.summary())
    return model_fit

# Step 5: 模型评估 / Model evaluation
def evaluate_model(series, model_fit):
    """
    使用均方误差 (MSE) 评估模型性能 / Evaluate model performance using Mean Squared Error (MSE)
    """
    predictions = model_fit.predict(start=0, end=len(series)-1)
    mse = mean_squared_error(series, predictions)
    print(f'均方误差 (Mean Squared Error): {mse}')

# 主函数 / Main function
def process_series(series, name):
    """
    处理每列数据的完整流程 / Process the full workflow for each column
    """
    print(f"\n处理 {name} / Processing {name}")
    
    # ADF 检测 / ADF test
    if not adf_test(series):
        print("进行一阶差分 / Performing first-order differencing")
        series_diff = series.diff().dropna()
        if not adf_test(series_diff):
            print("数据仍不平稳，尝试二阶差分 / Data still not stationary, trying second-order differencing")
            series_diff = series.diff().diff().dropna()
            d = 2
        else:
            d = 1
            series_diff = series_diff
    else:
        series_diff = series
        d = 0
    
    # ACF 和 PACF 分析 / ACF and PACF analysis
    plot_acf_pacf(series_diff)
    
    # 用户输入 p 和 q，最大为 3 / User inputs p and q, maximum 3
    p = int(input(f"请输入 {name} 的 p 值 (根据 PACF 图，最大为 3) / Enter p for {name} (based on PACF, max 3): "))
    q = int(input(f"请输入 {name} 的 q 值 (根据 ACF 图，最大为 3) / Enter q for {name} (based on ACF, max 3): "))
    p = min(p, 3)  # 确保 p 不超过 3 / Ensure p does not exceed 3
    q = min(q, 3)  # 确保 q 不超过 3 / Ensure q does not exceed 3
    
    # ARIMA 模型拟合 / Fit ARIMA model
    model_fit = fit_arima(series, p, d, q)
    
    # 模型评估 / Evaluate model
    evaluate_model(series, model_fit)

# 对每列数据进行处理 / Process each column
columns = ['temperature(oC)', 'humidity(%)', 'solar_radiation(W/m2)']
for column in columns:
    process_series(data[column], column)
```

##  一、代码整体功能说明

该代码完整展示了使用 Python 对时间序列数据进行 **ARIMA 模型分析** 的流程：

- **读取 CSV 数据并绘制折线图**，观察数据趋势。
- 使用 **ADF 检验**检查数据的**平稳性**。
- 根据需要进行差分（最多两次）达到平稳。
- 绘制 **ACF（自相关函数）** 和 **PACF（偏自相关函数）** 图，确定 ARIMA 模型参数（`p` 和 `q`）。
- 用户根据图形输入 ARIMA 参数。
- 使用 ARIMA 模型进行拟合。
- 评估模型拟合效果（MSE：均方误差）。

------

## 📚 二、各部分详细解析

### 📌 Step 1: 导入库与数据可视化

```
python复制编辑import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
```

- 导入必要库：数据处理、时间序列分析、绘图及模型评估。

```
python


复制编辑
data = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
```

- 从指定路径读取 CSV 数据，并将日期列解析为日期索引。

#### 📊 数据趋势可视化

```
python复制编辑plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(data['temperature(oC)'], label='Temperature (oC)')
plt.title('温度 / Temperature')
plt.legend()
```

- 为三个变量分别绘制折线图，以便直观查看数据趋势。

------

### 📌 Step 2: 平稳性检测（ADF检验）

函数定义：

```
python复制编辑def adf_test(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("数据不平稳 / Data is not stationary")
        return False
    else:
        print("数据平稳 / Data is stationary")
        return True
```

- **ADF单位根检验**用于判断序列是否平稳：
  - `p-value ≤ 0.05` 平稳；
  - `p-value > 0.05` 非平稳，需要差分。

------

### 📌 Step 3: 绘制 ACF 和 PACF 图

```
python复制编辑def plot_acf_pacf(series, lags=20):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    acf_vals = acf(series, nlags=lags)
    plt.bar(range(len(acf_vals)), acf_vals)
    plt.title('ACF')
    
    plt.subplot(1, 2, 2)
    pacf_vals = pacf(series, nlags=lags)
    plt.bar(range(len(pacf_vals)), pacf_vals)
    plt.title('PACF')
    plt.show()
```

- **ACF图**用于确定 MA 阶数 q。
- **PACF图**用于确定 AR 阶数 p。

------

### 📌 Step 4: ARIMA 模型拟合函数

```
python复制编辑def fit_arima(series, p, d, q):
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit
```

- 使用 ARIMA(p,d,q) 模型拟合数据，默认使用**最小二乘法(OLS)**进行参数估计。

------

### 📌 Step 5: 模型性能评估函数（MSE）

```
python复制编辑def evaluate_model(series, model_fit):
    predictions = model_fit.predict(start=0, end=len(series)-1)
    mse = mean_squared_error(series, predictions)
    print(f'均方误差 (Mean Squared Error): {mse}')
```

- 使用**均方误差(MSE)**评估模型预测的准确性。

------

## 📚 三、主流程函数 `process_series` 的逻辑

该函数处理每一个时间序列变量的全流程：

### 1️⃣ 平稳性检验与差分处理

```
python复制编辑if not adf_test(series):
    print("进行一阶差分 / Performing first-order differencing")
    series_diff = series.diff().dropna()
    if not adf_test(series_diff):
        print("数据仍不平稳，尝试二阶差分 / Data still not stationary, trying second-order differencing")
        series_diff = series.diff().diff().dropna()
        d = 2
    else:
        d = 1
else:
    series_diff = series
    d = 0
```

- 依次进行ADF检测、一阶差分、二阶差分，直至数据平稳为止。
- 记录差分阶数 d。

### 2️⃣ 确定 ARIMA 参数（用户交互式）

```
python复制编辑plot_acf_pacf(series_diff)

p = int(input(f"请输入 {name} 的 p 值 (根据 PACF 图，最大为 3): "))
q = int(input(f"请输入 {name} 的 q 值 (根据 ACF 图，最大为 3): "))
p = min(p, 3)
q = min(q, 3)
```

- 根据绘制的ACF与PACF图，由用户判断并输入最优的 p、q（最大为3）。

### 3️⃣ 模型拟合与评估

```
python复制编辑model_fit = fit_arima(series, p, d, q)
evaluate_model(series, model_fit)
```

- 拟合模型并计算模型的MSE，评估预测效果。

------

## 🚩 四、执行主程序逻辑

```
python复制编辑columns = ['temperature(oC)', 'humidity(%)', 'solar_radiation(W/m2)']
for column in columns:
    process_series(data[column], column)
```

- 依次对数据中的三个变量（温度、湿度、太阳辐射）执行上述流程。

------

## 🌈 五、代码的适用范围与扩展

该代码的主要应用领域：

- 气象、环境监测数据分析。
- 能源管理（如光伏发电预测）。
- 金融领域（股票价格、汇率、市场波动预测）。

后续可以扩展实现：

- 自动选择最佳参数（通过AIC/BIC优化）。
- 未来数据预测与可视化。
- 更高级模型（如SARIMA）和交叉验证评估。

------

## 🧑‍💻 总结

这段代码提供了一个清晰、易于理解的模板化流程，适合：

- 时间序列分析的初学者。
- 实践数据分析项目的模板代码。
- 交互式教学或实验环境中的示范。

它体现了从数据分析、参数确定到模型拟合评估的完整 ARIMA 分析步骤，清晰且易于扩展。