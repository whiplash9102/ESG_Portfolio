import refinitiv.data as rd
import pandas as pd
import logging

# Thiết lập logging để theo dõi tiến trình thay vì dùng print thông thường
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

import pandas as pd
import numpy as np

import logging
import pandas as pd
import numpy as np
import refinitiv.data as rd

# Thiết lập logging để theo dõi tiến trình
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
def scan_safe_momentum(df_returns, lookback=20, z_ceiling=2.0):
    """Quét Momentum an toàn trên bảng Daily Returns (Đã bọc giáp chống lỗi String)."""
    
    # BƯỚC ÉP KIỂU SINH TỬ: Ép toàn bộ bảng về định dạng số (float). 
    # Bất kỳ ký tự chữ, rác, hay khoảng trắng nào sẽ bị biến thành NaN (Not a Number)
    df_returns = df_returns.apply(pd.to_numeric, errors='coerce')
    
    # Cắt lấy số ngày lookback và lấp đầy các khoảng trống (NaN) bằng 0 
    # (Tương đương với việc ngày đó cổ phiếu không giao dịch, tỷ suất = 0%)
    df_recent = df_returns.tail(lookback).fillna(0)
    
    # 1. Tái tạo Chuỗi giá chuẩn hóa
    normalized_prices = 100 * (1 + df_recent).cumprod()
    
    # 2. Tính Momentum
    momentum_scores = (1 + df_recent).prod() - 1
    
    # 3. Tính toán thông số
    sma_normalized = normalized_prices.mean()
    std_normalized = normalized_prices.std().replace(0, 1e-9)
    current_normalized = normalized_prices.iloc[-1]
    
    z_scores = (current_normalized - sma_normalized) / std_normalized
    
    results = []
    for ticker in momentum_scores.index:
        mom_val = momentum_scores[ticker]
        z_val = z_scores[ticker]
        current_p = current_normalized[ticker]
def scan_zscore_from_returns(df_returns, lookback=20, z_threshold=-2.0):
    """Quét tìm các mã bị bán tháo cực đoan (Z-score < ngưỡng)."""
    df_recent = df_returns.tail(lookback)
    
    normalized_prices = 100 * (1 + df_recent).cumprod()
    
    rolling_mean = normalized_prices.rolling(window=lookback, min_periods=1).mean()
    # Thêm bảo vệ chia cho 0
    rolling_std = normalized_prices.rolling(window=lookback, min_periods=1).std().replace(0, 1e-9)
    
    z_scores = (normalized_prices - rolling_mean) / rolling_std
    current_zscores = z_scores.iloc[-1]
    
    results = []
    for ticker in current_zscores.index:
        z = current_zscores[ticker]
        if pd.isna(z):
            continue
            
        if z <= z_threshold:
            results.append({
                'Ticker': ticker,
                'Z-Score': round(z, 2),
                'Tín hiệu': 'Cực đoan (Oversold)'
            })
            
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return "Không có mã nào bị bán tháo tới mức cực đoan hôm nay."
    return df_results.sort_values(by='Z-Score', ascending=True).reset_index(drop=True)

def scan_smart_mean_reversion(df_close, lookback_z=20, sma_period=50, z_threshold=-2.0):
    """Quét bắt đáy thông minh: Z-score < -2.0 VÀ Giá > SMA (mặc định 50 do gọi tham số)."""
    if len(df_close) < sma_period:
        return f"Lỗi: Dữ liệu không đủ {sma_period} phiên giao dịch để tính SMA."
        
    sma_long = df_close.rolling(window=sma_period).mean()
    
    rolling_mean_20 = df_close.rolling(window=lookback_z).mean()
    rolling_std_20 = df_close.rolling(window=lookback_z).std().replace(0, 1e-9)
    z_scores = (df_close - rolling_mean_20) / rolling_std_20
    
    current_price = df_close.iloc[-1]
    current_sma_long = sma_long.iloc[-1]
    current_zscore = z_scores.iloc[-1]
    
    results = []
    for ticker in df_close.columns:
        p = current_price[ticker]
        sma = current_sma_long[ticker]
        z = current_zscore[ticker]
        
        # Bỏ qua nếu dữ liệu không đủ (bị NaN)
        if pd.isna(p) or pd.isna(sma) or pd.isna(z):
            continue
            
        is_oversold = z <= z_threshold
        is_uptrend_longterm = p > sma
        
        if is_oversold and is_uptrend_longterm:
            results.append({
                'Ticker': ticker,
                'Giá Hiện Tại': round(p, 2),
                'Z-Score (20D)': round(z, 2),
                f'SMA {sma_period}': round(sma, 2),
                'Tín hiệu': 'Bắt đáy An toàn'
            })
            
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return "Hệ thống: Không có mã xịn nào bị bán tháo."
    return df_results.sort_values(by='Z-Score (20D)', ascending=True).reset_index(drop=True)

def scan_safe_momentum(df_returns, lookback=20, z_ceiling=2.0):
    """Quét Momentum an toàn trên bảng Daily Returns."""
    df_recent = df_returns.tail(lookback)
    
    normalized_prices = 100 * (1 + df_recent).cumprod()
    momentum_scores = (1 + df_recent).prod() - 1
    
    sma_normalized = normalized_prices.mean()
    std_normalized = normalized_prices.std().replace(0, 1e-9)
    current_normalized = normalized_prices.iloc[-1]
    
    z_scores = (current_normalized - sma_normalized) / std_normalized
    
    results = []
    for ticker in momentum_scores.index:
        mom_val = momentum_scores[ticker]
        z_val = z_scores[ticker]
        current_p = current_normalized[ticker]
        sma_p = sma_normalized[ticker]
        
        if pd.isna(mom_val) or pd.isna(z_val):
            continue
            
        is_uptrend = current_p > sma_p
        is_safe = z_val < z_ceiling
        
        if not is_uptrend:
            status = "Gãy Trend (Loại)"
        elif not is_safe:
            status = f"Đu đỉnh FOMO / Z={z_val:.2f} (Loại)"
        else:
            status = "An Toàn (Pass)"
            
        results.append({
            'Ticker': ticker,
            'Momentum (20D %)': round(mom_val * 100, 2),
            'Z-Score': round(z_val, 2),
            'Trạng thái': status
        })
        
    df_results = pd.DataFrame(results)
    safe_leaders = df_results[df_results['Trạng thái'] == 'An Toàn (Pass)'].sort_values(by='Momentum (20D %)', ascending=False)
    return safe_leaders.reset_index(drop=True), df_results

def calculate_capital_allocation(df_returns, target_tickers, total_capital=90_000_000, lookback=20):
    """Chia tiền Risk Parity dựa trên Độ lệch chuẩn."""
    # Chỉ giữ lại các ticker thực sự có trong DataFrame để tránh lỗi KeyError
    valid_tickers = [t for t in target_tickers if t in df_returns.columns]
    
    if not valid_tickers:
        return pd.DataFrame()

    df_target_returns = df_returns[valid_tickers].tail(lookback)
    volatility = df_target_returns.std().replace(0, 1e-9) # Bảo vệ chia cho 0
    
    inv_vol_sum = sum(1 / vol for vol in volatility)
    results = []
    
    for ticker in valid_tickers:
        vol = volatility[ticker]
        weight = (1 / vol) / inv_vol_sum
        allocated_money = weight * total_capital
        
        results.append({
            'Ticker': ticker,
            'Độ giật rủi ro (Std Dev)': round(vol, 4),
            'Tỷ trọng vốn (%)': round(weight * 100, 2),
            'Tiền giải ngân (VND/EUR)': round(allocated_money, 0)
        })
        
    return pd.DataFrame(results)