import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq

# --- 1. 頁面基礎設定 ---
st.set_page_config(page_title="2026 指數連結商品策略分析", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Microsoft JhengHei', Arial, sans-serif;
    }
    .main-title {
        font-size: 28px;
        color: #2C3E50;
        border-bottom: 2px solid #2C3E50;
        padding-bottom: 10px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .sub-title {
        font-size: 16px;
        color: #7F8C8D;
        margin-bottom: 25px;
    }
    h2 { font-size: 20px; color: #2C3E50; margin-top: 30px; margin-bottom: 15px; font-weight: 600; border-left: 4px solid #2C3E50; padding-left: 10px; }
    .stApp { background-color: #FFFFFF; } 
    
    .analysis-note {
        background-color: #F2F4F6;
        padding: 15px;
        border-radius: 2px;
        color: #2C3E50;
        font-size: 14px;
        margin-top: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #95A5A6;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">2026 指數連結型商品避險策略分析 (IUL / FIA Hedging Solutions)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">富邦期貨法人部 / 日期：2026-01-15</div>', unsafe_allow_html=True)

# --- 2. 側邊欄：全域參數設定 ---
st.sidebar.title("參數控制台")
st.sidebar.caption("v16.7 - Institutional Edition")

st.sidebar.markdown("### 快速情境設定")
scenario_mode = st.sidebar.selectbox("選擇產品模式", 
                                     ["自訂 (Custom)", 
                                      "短年期躉繳 (如指享盈)", 
                                      "長年期壽險 (IUL)"])

# 根據情境設定預設參數
if scenario_mode == "短年期躉繳 (如指享盈)":
    def_tenor = 6
    def_load = 0.80
    def_yield = 4.80
elif scenario_mode == "長年期壽險 (IUL)":
    def_tenor = 20
    def_load = 1.20
    def_yield = 5.50
else:
    def_tenor = 5
    def_load = 0.50
    def_yield = 5.20

page = st.sidebar.radio("選擇分析頁面", 
    ["0. 策略架構說明 (Intro)",
     "1. 方案一：純參與率型 (Uncapped PR)", 
     "2. 方案二：價差避險型 (Bull Call Spread)", 
     "3. 兩案比較與利潤分析 (Comparison)",
     "4. 交易執行計算機 (Trader Execution)"]) 

st.sidebar.markdown("---")

# A. 產品結構
st.sidebar.subheader("1. 產品結構")
tenor_simulation = st.sidebar.number_input("模擬持有年期 (Years)", value=def_tenor, step=1)
sales_expense_amortized = st.sidebar.number_input("年度攤提費用 (Amortized Load) %", value=def_load, step=0.10) / 100

if scenario_mode == "短年期躉繳 (如指享盈)" and sales_expense_amortized < 0.007:
    st.sidebar.warning("提示：短年期佣金攤提壓力較大，建議費用設為 0.7% 以上以符實務。")

# B. 市場環境
st.sidebar.subheader("2. 市場環境")
r_rf = st.sidebar.number_input("無風險利率 (Risk-Free) %", value=4.20, step=0.01) / 100
div_q = st.sidebar.number_input("標的股利率 (Dividend Yield) %", value=1.50, step=0.01) / 100
sigma_atm = st.sidebar.slider("1Y ATM Implied Vol %", 10.0, 30.0, 16.0, step=0.5) / 100
skew_slope = st.sidebar.slider("波動率偏斜係數 (Skew Slope)", -0.5, 0.0, -0.2, step=0.05)

with st.sidebar.expander("進階波動率限制 (Vol Limits)"):
    vol_floor = st.slider("波動率下限 (Vol Floor) %", 2.0, 12.0, 5.0, step=0.5) / 100
    vol_cap = st.slider("波動率上限 (Vol Cap) %", 30.0, 150.0, 80.0, step=5.0) / 100

# C. 資金與成本
st.sidebar.subheader("3. 資金與成本")
bond_yield = st.sidebar.number_input("債券收益率 (Funding Yield) %", value=def_yield, step=0.10) / 100
issuer_spread = st.sidebar.number_input("公司目標利差 (Issuer Spread) %", value=1.50, step=0.10) / 100

is_usd_policy = st.sidebar.checkbox("美金保單 (無 FX 避險成本)", value=True)

if is_usd_policy:
    fx_hedge_cost = 0.0
    st.sidebar.caption("已排除 FX 避險損耗")
else:
    fx_hedge_cost = st.sidebar.number_input("FX 避險成本 (FX Cost) %", value=1.50, step=0.05) / 100

opt_spread_cost = st.sidebar.number_input("期權總合成本係數 (Spread Cost) %", value=0.80, step=0.10) / 100

# --- 3. 核心函數 ---

def get_vol_at_strike(K, S, sigma_atm, slope):
    if K <= 0: return sigma_atm
    moneyness = np.log(K / S)
    raw_vol = sigma_atm + (slope * moneyness)
    clamped_vol = max(vol_floor, min(raw_vol, vol_cap))
    return clamped_vol

def bs_price(S, K, T, r, q, vol, option_type='call'):
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0: return np.nan
    try:
        d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return max(0.0, price)
    except: return np.nan

def solve_strike_dynamic(target_price, S0, T, r, q, atm_vol, slope, spread_cost):
    def obj_func(K):
        v_k = get_vol_at_strike(K, S0, atm_vol, slope)
        price = bs_price(S0, K, T, r, q, v_k, 'call')
        return price - target_price
    brackets = [3.0, 5.0, 8.0, 15.0]
    for b_mult in brackets:
        try:
            val_a = obj_func(S0)
            val_b = obj_func(S0 * b_mult)
            if np.isnan(val_a) or np.isnan(val_b): continue
            if val_a * val_b < 0:
                k_res = brentq(obj_func, S0, S0 * b_mult)
                return k_res, get_vol_at_strike(k_res, S0, atm_vol, slope)
        except: continue
    return 0.0, 0.0

# [P0] 年度預算核心計算
S0 = 100
annual_option_budget_pct = bond_yield - issuer_spread - sales_expense_amortized - fx_hedge_cost
annual_option_budget_amt = S0 * annual_option_budget_pct

# --- 4. 共用繪圖函數 ---
def plot_multi_year_simulation(pr, cap, tenor):
    years = np.arange(0, tenor + 1)
    
    path_bull_idx = [100 * (1.06 ** t) for t in years]
    path_bull_val = [100]
    for t in range(1, tenor + 1):
        ret = (path_bull_idx[t] / path_bull_idx[t-1]) - 1
        credit = min(cap, max(0, ret * pr))
        path_bull_val.append(path_bull_val[-1] * (1 + credit))
        
    path_chop_idx = [100]
    for t in range(1, tenor + 1):
        change = 1.10 if t % 2 != 0 else 0.90
        path_chop_idx.append(path_chop_idx[-1] * change)
    path_chop_val = [100]
    for t in range(1, tenor + 1):
        ret = (path_chop_idx[t] / path_chop_idx[t-1]) - 1
        credit = min(cap, max(0, ret * pr))
        path_chop_val.append(path_chop_val[-1] * (1 + credit))

    path_v_idx = [100, 90, 85, 95, 105, 115, 125, 135]
    if tenor > 7: path_v_idx += [path_v_idx[-1]*1.05 for _ in range(tenor-7)]
    path_v_idx = path_v_idx[:tenor+1]
    
    path_v_val = [100]
    for t in range(1, tenor + 1):
        ret = (path_v_idx[t] / path_v_idx[t-1]) - 1
        credit = min(cap, max(0, ret * pr))
        path_v_val.append(path_v_val[-1] * (1 + credit))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=path_bull_val, name="情境A: 穩健上漲 (CAGR~6%)", line=dict(color='#27AE60', width=3)))
    fig.add_trace(go.Scatter(x=years, y=path_chop_val, name="情境B: 區間震盪 (+10%/-10%)", line=dict(color='#F39C12', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=years, y=path_v_val, name="情境C: 先跌後漲 (保本發揮)", line=dict(color='#E74C3C', width=3, dash='dot')))
    
    fig.update_layout(
        title=f"長期持有情境模擬 ({tenor} Years) - 累積帳戶價值",
        xaxis_title="保單年度", yaxis_title="帳戶價值 (本金100)",
        template="plotly_white", height=450, hovermode="x unified"
    )
    return fig

# --- 程式本體 ---

if annual_option_budget_amt <= 0:
    st.error(f"預算不足 (Budget <= 0)！目前淨預算: {annual_option_budget_pct:.2%}。請檢查債券收益或費用設定。")
    st.stop()

# ==========================================
# PAGE 0: 策略架構說明 (Intro)
# ==========================================
if page == "0. 策略架構說明 (Intro)":
    st.markdown("### 指數連結保單 (FIA/IUL) 產品策略架構")
    st.caption("從底層資產配置到頂層行銷功能的完整解析")

    # --- 1. 底層：保本架構 ---
    st.subheader("1. 底層：保本架構 (The Floor & Guarantee)")
    st.markdown("這是產品的基石，確保客戶資金的安全性與流動性鎖定。")
    
    col_desc_1, col_chart_1 = st.columns([1, 1.5])
    
    with col_desc_1:
        st.markdown("""
        **資金池 (General Account)**
        這是壽險公司的資產負債表核心。當收到客戶 100 萬元保費後：
        
        1.  **投資配置 (Investment)**：
            * **95%~96%**：配置於高評級債券，利用利息收益支撐保本。
            * **4%~5%**：衍生性商品預算 (Option Budget)，即期貨商的服務範疇。
            
        2.  **關鍵機制：Surrender Charge (解約費用)**
            * 透過遞減式的解約費用（如右圖），鎖定資金存續期間。
            * 這讓投資部門得以買入長天期債券，獲取較高的 Duration Premium。
        """)

    with col_chart_1:
        # 圖 1: 資產配置
        fig_alloc = go.Figure(data=[go.Pie(
            labels=['高評級債券 (保本)', '期權預算 (交易室戰場)'], 
            values=[96, 4], 
            hole=.4,
            marker_colors=['#2E86C1', '#E74C3C']
        )])
        fig_alloc.update_layout(title_text="資產配置模型", height=250, margin=dict(t=30, b=0, l=0, r=0))

        # 圖 2: 解約費用曲線
        years = [1, 2, 3, 4, 5, 6, 7, 8]
        charges = [7, 6, 5, 4, 3, 2, 1, 0]
        
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=years, y=charges, mode='lines+markers', name='解約費用 %', line=dict(color='#D35400', width=3)))
        fig_sc.update_layout(
            title_text="Surrender Charge Schedule (鎖利機制)",
            xaxis_title="保單年度",
            yaxis_title="解約費用 (%)",
            height=250,
            margin=dict(t=30, b=0, l=0, r=0)
        )

        tab1, tab2 = st.tabs(["資產配置 (Allocation)", "解約費用 (Surrender Charge)"])
        with tab1:
            st.plotly_chart(fig_alloc, use_container_width=True)
        with tab2:
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # --- 2. 中層：獲利引擎 ---
    st.subheader("2. 中層：獲利引擎 (The Crediting Method)")
    st.markdown("決定客戶利息的核心機制。資金進入後依客戶選擇分為固定或指數連結。")

    col_fix, col_idx = st.columns([1, 1.5])
    
    with col_fix:
        with st.container():
            st.markdown("#### 選項 A：固定利率帳戶 (Fixed)")
            st.markdown("性質：**類定存 (Bank CD)**")
            st.metric("宣告利率", "3.50%", delta="無風險")
            st.markdown("""
            * **功能**：保守資金的避風港，提供確定性收益。
            * **機制**：每年由保險公司宣告固定利率。
            """)

    with col_idx:
        with st.container():
            st.markdown("#### 選項 B：指數連結帳戶 (Indexed)")
            st.markdown("性質：**買權策略 (Call Option Strategy)**")
            st.markdown("**連結標的**：S&P 500 / NASDAQ / 波動率控制指數")
            
            st.markdown("##### 關鍵參數 (Levers)")
            l1, l2, l3 = st.columns(3)
            l1.metric("Cap (上限)", "9.00%", help="最大獲利限制")
            l2.metric("PR (參與率)", "100%", help="指數漲幅之計算倍數")
            l3.metric("Spread (扣點)", "0.00%", help="指數漲幅之內扣費用")
            
            st.markdown("""
            * **計息公式**：`1-Year Point-to-Point` (主流規格)
            * **重設機制 (Annual Reset)**：每年獲利鎖定 (Ratchet)，確保過往收益不被市場下跌侵蝕。
            """)

    st.markdown("---")

    # --- 3. 頂層：加值功能 ---
    st.subheader("3. 頂層：保單加值功能 (The Bells & Whistles)")
    st.markdown("用於提升產品行銷吸引力的選配功能 (Riders)。")
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Bonus (首年獎勵)**\n\n"
                "- **機制**：客戶進場即獲得額外獎勵金 (例如 5%)。\n"
                "- **代價**：成本反映於較低的 Cap 或較高的費用。\n"
                "- **目的**：行銷亮點，吸引新資金。")
    with c2:
        st.success("**Index Lock (指數鎖利)**\n\n"
                   "- **機制**：允許客戶在保單年度中，手動鎖定至今的指數獲利。\n"
                   "- **特色**：增加客戶對市場的掌控感。\n"
                   "- **代價**：增加避險操作的複雜度與成本。")

# ==========================================
# PAGE 1: 方案一
# ==========================================
elif page == "1. 方案一：純參與率型 (Uncapped PR)":
    st.markdown("### 方案一：純參與率型 (Uncapped PR)")
    st.caption("策略描述：將年度淨預算全數購買 ATM Call，提供客戶無上限的固定參與率。")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("年度資金結構")
        st.metric("債券收益 (Yield)", f"{bond_yield:.2%}")
        st.metric("扣除：公司利潤/費用/FX", f"-{(issuer_spread + sales_expense_amortized + fx_hedge_cost):.2%}")
        st.metric("年度期權預算", f"{annual_option_budget_pct:.2%}")
        
    with col2:
        st.subheader("規格試算 (1-Year Option)")
        T_pricing = 1.0 
        call_atm_raw = bs_price(S0, S0, T_pricing, r_rf, div_q, sigma_atm, 'call')
        
        if np.isnan(call_atm_raw):
            st.error("期權價格計算失敗，請檢查參數")
            st.stop()
            
        call_atm_ask = call_atm_raw * (1 + opt_spread_cost)
        pr_opt1 = annual_option_budget_amt / call_atm_ask
        
        st.metric("1Y ATM Call 成本 (含費)", f"${call_atm_ask:.2f}")
        delta_color = "normal"
        if pr_opt1 > 1.0: delta_color = "inverse"
        st.metric("年度參與率 (PR)", f"{pr_opt1:.2%}", delta="無上限", delta_color=delta_color)
        if pr_opt1 > 1.0:
            st.warning("注意：參與率超過 100%，需確認避險可行性或設定 PR 上限。")
        
    st.markdown("---")
    st.subheader("1. 年度損益模擬 (Annual Payoff)")
    
    market_moves = np.linspace(-0.10, 0.20, 400)
    y_opt1 = [max(0, m * pr_opt1) for m in market_moves]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=market_moves*100, y=market_moves*100, name="Index Return", line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=market_moves*100, y=np.array(y_opt1)*100, name=f"Option 1 (PR={pr_opt1:.0%})", line=dict(color='#2E86C1', width=4)))
    fig.update_layout(
        title="單一年度收益結構", 
        xaxis_title="指數年度漲幅 (%)", 
        yaxis_title="客戶年度收益 (%)", 
        template="plotly_white", 
        height=550, 
        xaxis=dict(range=[-10, 20], dtick=5), 
        yaxis=dict(range=[-5, 25])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="analysis-note">
    <b>富期分析觀點：</b><br>
    此結構適合預期市場有大波段行情的客戶。雖然參與率固定為 {pr_opt1:.1%}，但上方獲利無封頂，能完整捕捉極端正向的市場報酬。
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(f"2. 長期持有情境模擬 ({tenor_simulation} Years)")
    fig_sim = plot_multi_year_simulation(pr_opt1, 999.0, int(tenor_simulation))
    st.plotly_chart(fig_sim, use_container_width=True)
    
    st.markdown(f"""
    <div class="analysis-note">
    <b>富期分析觀點：</b><br>
    在「情境A 穩健上漲」中，由於無上限機制，複利效果最為顯著。若市場進入「情境B 區間震盪」，年度重設機制 (Annual Reset) 仍可鎖定正報酬年份的獲利，發揮保單抗跌特性。
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: 方案二
# ==========================================
elif page == "2. 方案二：價差避險型 (Bull Call Spread)":
    st.markdown("### 方案二：價差避險型 (Bull Call Spread)")
    
    st.caption("策略描述：採用多頭價差策略，即「買入 ATM 買權 + 賣出 OTM 買權 (Cap)」。透過賣出上方極端獲利的權利金收入來降低避險成本，進而在設定的 Cap 區間內提供優於方案一的參與率。")

    with st.expander("量化模型說明 (Model Methodology)", expanded=True):
        st.markdown(f"""
        <div class="audit-box">
        <b>[機制] 年度重設 (Annual Reset):</b> 本模型採用 1 年期 Bull Call Spread 定價。<br>
        <b>[假設] 資產支持 (ALM):</b> 0% Floor 假設由資產端保本配置支持。<br>
        <b>[波動率] 動態偏斜 (Dynamic Skew):</b> 賣出 Cap 時，使用 Strike-Dependent Volatility (Smile) 進行定價。<br>
        <b>[限制] 波動率範圍 (Vol Clamp):</b> {vol_floor:.1%} ~ {vol_cap:.0%} (防止數值發散)。
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 設計模式選擇")
    solve_mode = st.radio("請選擇設計邏輯：", 
                          ["模式 A：固定 Cap (自訂) ➜ 算出 參與率 (PR)", 
                           "模式 B：固定參與率 (100%) ➜ 算出 Cap"])
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    T_pricing = 1.0
    call_atm_raw = bs_price(S0, S0, T_pricing, r_rf, div_q, sigma_atm, 'call')
    if np.isnan(call_atm_raw):
         st.error("ATM Call 計算失敗")
         st.stop()
    call_atm_ask = call_atm_raw * (1 + opt_spread_cost)
    
    final_cap = 0; final_pr = 0; cap_display = ""
    
    if solve_mode == "模式 A：固定 Cap (自訂) ➜ 算出 參與率 (PR)":
        with col1:
            st.subheader("模式 A：鎖定 Cap (競品對標)")
            target_cap_input = st.slider("請設定目標 Cap %", 3.0, 15.0, 8.0, step=0.5) / 100
            
        with col2:
            st.subheader("試算結果")
            k_cap_target = S0 * (1 + target_cap_input)
            
            vol_skewed = get_vol_at_strike(k_cap_target, S0, sigma_atm, skew_slope)
            call_short_raw = bs_price(S0, k_cap_target, T_pricing, r_rf, div_q, vol_skewed, 'call')
            
            eq_rev_short = call_short_raw * (1 - opt_spread_cost)
            unit_spread_cost = call_atm_ask - eq_rev_short
            
            if unit_spread_cost <= 0:
                 st.error("組合單成本異常 (賣比買貴)，請檢查波動率設定")
                 final_pr = 0
            else:
                final_pr = annual_option_budget_amt / unit_spread_cost
            
            final_cap = target_cap_input
            cap_display = f"{final_cap:.2%}"
            
            st.metric("設定年度上限 (Cap)", cap_display)
            
            delta_color = "normal"
            if final_pr > 1.0: delta_color = "inverse"
            st.metric("可提供參與率 (PR)", f"{final_pr:.2%}", delta="考慮 Skew 後", delta_color=delta_color)

    else: # 模式 B：固定參與率 (100%)
        with col1:
            st.subheader("模式 B：鎖定 PR = 100%")
            funding_gap = call_atm_ask - annual_option_budget_amt
            st.metric("年度預算缺口", f"-${funding_gap:.2f}", help="為了達到 100% 參與率，我們還缺多少錢，需透過賣 Call 來補")

        with col2:
            st.subheader("試算結果")
            if funding_gap <= 0:
                final_cap = 9.99; cap_display = "無上限"
                final_pr = 1.0
            else:
                target_short_val = funding_gap / (1 - opt_spread_cost)
                k_cap, vol_at_cap = solve_strike_dynamic(target_short_val, S0, T_pricing, r_rf, div_q, sigma_atm, skew_slope, opt_spread_cost)
                
                if k_cap > 0:
                    final_cap = (k_cap / S0) - 1
                    cap_display = f"{final_cap:.2%}"
                else:
                    final_cap = 0; cap_display = "無法計算"
                
                final_pr = 1.0
            
            st.metric("年度參與率 (PR)", "100%")
            st.metric("推算年度上限 (Cap)", cap_display, delta="考慮 Skew 後")
            
            if final_cap > 0 and final_cap < 9.0:
                st.info(f"Cap Moneyness = {final_cap+1:.2f}x | Implied Vol(Cap) = {vol_at_cap:.2%}")

    st.markdown("---")
    st.subheader("1. 年度損益模擬 (Annual Payoff)")
    
    market_moves = np.linspace(-0.10, 0.30, 400)
    y_opt2 = [max(0, min(m * final_pr, final_cap)) for m in market_moves]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=market_moves*100, y=market_moves*100, name="Index Performance", line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=market_moves*100, y=np.array(y_opt2)*100, name=f"Client Payoff (Cap={cap_display})", line=dict(color='#C0392B', width=4)))
    
    if isinstance(final_cap, (int, float)):
        anno_x = 25 
        fig.add_annotation(
            x=anno_x,  
            y=final_cap*100, 
            text=f"獲利封頂 {cap_display}", 
            showarrow=True, 
            arrowhead=2, 
            ax=0, 
            ay=-40, 
            font=dict(color="#C0392B", size=12)
        )

    fig.update_layout(
        title="單一年度收益結構 (Payoff Diagram)", 
        xaxis_title="指數年度漲幅 (%)", 
        yaxis_title="客戶年度收益 (%)", 
        template="plotly_white", 
        height=550,
        xaxis=dict(range=[-10, 30], dtick=5),
        yaxis=dict(range=[-5, max(final_cap*100 * 1.5, 12)]) 
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="analysis-note">
    <b>富期分析觀點：</b><br>
    透過賣出上方買權 (Sold Call) 換取預算，本方案能在 0% 至 {cap_display} 的區間內提供更高的參與率（如試算所示為 {final_pr:.1%}）。
    此結構在溫和上漲的市場環境下優勢最大，符合壽險資金追求穩健絕對報酬的屬性。
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader(f"2. 長期持有情境模擬 ({tenor_simulation} Years)")
    f_pr = final_pr if isinstance(final_pr, (int, float)) else 0
    f_cap = final_cap if isinstance(final_cap, (int, float)) else 0
    fig_sim = plot_multi_year_simulation(f_pr, f_cap, int(tenor_simulation))
    st.plotly_chart(fig_sim, use_container_width=True)
    
    st.markdown(f"""
    <div class="analysis-note">
    <b>富期分析觀點：</b><br>
    在長期持有情境中，由於 Cap 的存在，雖然會切掉部分大漲年份的極端收益，但換來的高參與率能讓資產在中小幅波動的年份累積更快的淨值。
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 3: 兩案比較
# ==========================================
elif page == "3. 兩案比較與利潤分析 (Comparison)":
    st.markdown("### 兩案比較與利潤結構分析")
    
    T_pricing = 1.0
    call_atm_raw = bs_price(S0, S0, T_pricing, r_rf, div_q, sigma_atm, 'call')
    call_atm_ask = call_atm_raw * (1 + opt_spread_cost)
    
    pr_opt1 = annual_option_budget_amt / call_atm_ask
    
    gap = call_atm_ask - annual_option_budget_amt
    if gap <= 0: 
        final_cap_o2 = 9.99
    else:
        target_short = gap / (1 - opt_spread_cost)
        k_res, _ = solve_strike_dynamic(target_short, S0, T_pricing, r_rf, div_q, sigma_atm, skew_slope, opt_spread_cost)
        if k_res > 0:
            final_cap_o2 = (k_res / S0) - 1
        else:
            final_cap_o2 = 0
    final_pr_o2 = 1.0

    with st.expander("Cap 敏感度分析", expanded=False):
        sens_vols = [sigma_atm - 0.04, sigma_atm - 0.02, sigma_atm, sigma_atm + 0.02, sigma_atm + 0.04]
        sens_vols = [v for v in sens_vols if v > 0]
        sens_yields = [bond_yield - 0.01, bond_yield - 0.005, bond_yield, bond_yield + 0.005, bond_yield + 0.01]
        
        res_matrix = []
        for y in sens_yields:
            row = []
            b_amt = S0 * (y - issuer_spread - sales_expense_amortized - fx_hedge_cost)
            for v in sens_vols:
                if b_amt <= 0:
                    row.append("N/A")
                    continue
                c_atm = bs_price(S0, S0, 1.0, r_rf, div_q, v, 'call')
                if np.isnan(c_atm):
                    row.append("Err")
                    continue
                c_atm_ask = c_atm * (1 + opt_spread_cost)
                g = c_atm_ask - b_amt
                if g <= 0:
                    row.append("Uncapped")
                else:
                    tgt = g / (1 - opt_spread_cost)
                    k_sol, _ = solve_strike_dynamic(tgt, S0, 1.0, r_rf, div_q, v, skew_slope, opt_spread_cost)
                    if k_sol > 0:
                        row.append(f"{(k_sol/S0)-1:.2%}")
                    else:
                        row.append("Low Budg")
            res_matrix.append(row)
        df_sens = pd.DataFrame(res_matrix, columns=[f"Vol {v:.0%}" for v in sens_vols], index=[f"Yield {y:.2%}" for y in sens_yields])
        st.dataframe(df_sens)

    st.markdown("---")
    st.subheader("1. 效益整合分析")
    
    market_moves = np.linspace(-0.10, 0.20, 400)
    y_o1 = [max(0, m * pr_opt1) for m in market_moves]
    y_o2 = [max(0, min(m * final_pr_o2, final_cap_o2)) for m in market_moves]
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=market_moves*100, y=np.array(y_o1)*100, name=f'方案一: 純參與率型', line=dict(color='#2E86C1', width=3)))
    fig_comp.add_trace(go.Scatter(x=market_moves*100, y=np.array(y_o2)*100, name=f'方案二: 價差避險型', line=dict(color='#C0392B', width=4)))
    
    if final_pr_o2 > pr_opt1:
        cross_point = final_cap_o2 / pr_opt1 if pr_opt1 > 0 else 0
        draw_limit = min(cross_point, 0.20)
        
        if draw_limit > 0:
            x_zone = np.linspace(0, draw_limit, 100)
            y_upper = x_zone * final_pr_o2
            y_lower = x_zone * pr_opt1
            
            fig_comp.add_trace(go.Scatter(
                x=np.concatenate([x_zone, x_zone[::-1]]) * 100, 
                y=np.concatenate([y_upper, y_lower[::-1]]) * 100, 
                fill='toself', 
                fillcolor='rgba(46, 204, 113, 0.2)', 
                line=dict(color='rgba(255,255,255,0)'), 
                name='Bull Call Spread 優勢區間'
            ))
            
            if cross_point < 0.20:
                 fig_comp.add_annotation(x=cross_point*100, y=final_cap_o2*100, text=f"黃金交叉: {cross_point:.1%}", showarrow=True, arrowhead=2, ax=40, ay=-40)

    fig_comp.update_layout(
        title="單一年度客戶收益比較 (Comparison)", 
        height=650, 
        template="plotly_white", 
        hovermode="x unified", 
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(title="指數年度漲幅 (%)", range=[-10, 20], dtick=5),
        yaxis=dict(title="客戶年度收益 (%)", range=[-2, max(final_cap_o2*100 * 2.0, 15)])
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("---")
    st.subheader("2. 資本結構拆解 (Capital Structure Breakdown)")
    
    val_margin = issuer_spread
    val_cost = annual_option_budget_pct * (opt_spread_cost / (1 + opt_spread_cost))
    val_client = annual_option_budget_pct - val_cost
    
    fig_ics = go.Figure()
    fig_ics.add_trace(go.Bar(y=['結構'], x=[val_client*100], name='客戶權益 (Net Option Budget)', orientation='h', marker=dict(color='#AED6F1'), text=[f"{val_client:.2%}"], textposition='auto'))
    fig_ics.add_trace(go.Bar(y=['結構'], x=[val_cost*100], name='交易與避險成本', orientation='h', marker=dict(color='#F1948A'), text=[f"{val_cost:.2%}"], textposition='auto'))
    fig_ics.add_trace(go.Bar(y=['結構'], x=[fx_hedge_cost*100], name='FX 避險成本', orientation='h', marker=dict(color='#8E44AD'), text=[f"{fx_hedge_cost:.2%}"], textposition='auto'))
    fig_ics.add_trace(go.Bar(y=['結構'], x=[val_margin*0.4*100], name='風險邊際 (Risk Margin)', orientation='h', marker=dict(color='#F39C12'), text=[f"{val_margin*0.4:.2%}"], textposition='auto'))
    fig_ics.add_trace(go.Bar(y=['結構'], x=[val_margin*0.6*100], name='股東利潤 (Net Profit)', orientation='h', marker=dict(color='#27AE60'), text=[f"{val_margin*0.6:.2%}"], textposition='auto'))
    
    fig_ics.update_layout(barmode='stack', title="Spread 深度拆解 (Post-ICS 2.0)", height=450, margin=dict(l=20, r=20, t=50, b=20), xaxis_title="佔本金百分比 (%)")
    st.plotly_chart(fig_ics, use_container_width=True)

    st.markdown(f"""
    <div class="analysis-note">
    <h4>Post-ICS 2.0 結構說明：</h4>
    <p>隨著台灣接軌 <b>TW-ICS (Insurance Capital Standard)</b>，保險商品的定價邏輯已從傳統的「利差益」轉向「風險調整後資本回報 (RAROC)」。本圖表展示了在新的監理架構下，債券收益率 (Yield) 的真實分配：</p>
    <ul>
        <li><b>風險邊際 (Risk Margin)：</b> 在 ICS 架構下，持有風險資產（如債券）需計提資本風險電荷。此區塊代表為了支撐該資產配置所需的資金成本 (Cost of Capital)，非純利潤。</li>
        <li><b>股東利潤 (Net Profit)：</b> 扣除風險成本後的真實經濟利潤 (Economic Value Added)。</li>
        <li><b>依據：</b> 本模型假設 Issuer Spread 為 {issuer_spread:.2%}，並依照一般壽險業資本模型參數，將其約 40% 歸類為風險邊際，60% 為股東利潤。</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 4: 交易執行計算機 (Trader Execution)
# ==========================================
elif page == "4. 交易執行計算機 (Trader Execution)":
    st.markdown("### 交易執行計算機 (Execution Calculator)")
    st.caption("專為交易室設計：支援雙向求解（鎖定PR求Cap / 鎖定Cap求PR），快速生成精確指令。")

# --- 1. 交易參數輸入 ---
    with st.container():
        st.markdown("#### 1. 資金與標的設定")
        
        # 輸入模式切換
        input_mode = st.radio("預算輸入方式", ["方式 A：依預算比例 (%)", "方式 B：依權利金總額 ($)"], horizontal=True)
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            # [修改技巧] format="%d" 確保顯示整數，並在 help 中提示輸入格式
            notional_amt = st.number_input(
                "名目本金 (Notional) USD", 
                value=10000000, 
                step=1000000, 
                format="%d"
            )
            # [新增] 在下方用藍字顯示千分位，讓使用者確認金額
            st.markdown(f":blue[**確認金額: ${notional_amt:,.0f}**]")
            
            if input_mode == "方式 A：依預算比例 (%)":
                exec_budget_pct = st.number_input("核定預算 %", value=2.80, step=0.05, format="%.2f") / 100
                total_premium_amt = notional_amt * exec_budget_pct
                st.caption(f"換算總權利金: USD {total_premium_amt:,.0f}")
            else:
                total_premium_amt = st.number_input(
                    "可用權利金總額 (Premium) USD", 
                    value=280000, 
                    step=10000, 
                    format="%d"
                )
                # [新增] 同樣加上千分位確認
                st.markdown(f":blue[**確認金額: ${total_premium_amt:,.0f}**]")
                
                exec_budget_pct = total_premium_amt / notional_amt if notional_amt > 0 else 0
                st.caption(f"換算預算比例: {exec_budget_pct:.2%}")

        with c2:
            spot_price = st.number_input("SPY 現價 (Spot)", value=585.0, step=1.0, help="SPY價格約為SPX的1/10")
            contract_mult = st.number_input("合約乘數 (Multiplier)", value=100, help="SPY FLEX Option 乘數通常為100")
        with c3:
            exec_strategy = st.selectbox("避險策略", ["Call Spread (價差單)", "Call Only (單邊買權)"])
            
            target_val_input = 0.0
            calc_target = "N/A"
            
            if exec_strategy == "Call Spread (價差單)":
                calc_target = st.radio("計算目標 (Solver Target)", ["鎖定 PR ➔ 求 Cap", "鎖定 Cap ➔ 求 PR"])
                if calc_target == "鎖定 PR ➔ 求 Cap":
                    target_val_input = st.number_input("目標參與率 (Target PR)", value=1.0, step=0.1, format="%.2f")
                else:
                    target_val_input = st.number_input("目標封頂 (Target Cap) %", value=8.00, step=0.25, format="%.2f") / 100
            else:
                st.info("單邊買權模式：根據預算自動計算最大 PR")

    st.markdown("---")

    # --- 2. 求解與試算 ---
    st.markdown("#### 2. 規格試算 (Solver)")
    
    # 計算 ATM Call 價格 (成本)
    call_atm_price = bs_price(spot_price, spot_price, 1.0, r_rf, div_q, sigma_atm, 'call')
    call_atm_ask_exec = call_atm_price * (1 + opt_spread_cost)
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.info(f"ATM Call 預估成本: ${call_atm_ask_exec:.2f} ({(call_atm_ask_exec/spot_price):.2%})")
        
    specs_output = {} 

    if exec_strategy == "Call Only (單邊買權)":
        cal_pr = exec_budget_pct * spot_price / call_atm_ask_exec
        specs_output['PR'] = cal_pr
        specs_output['Cap'] = None
        specs_output['K_Lower'] = spot_price
        specs_output['K_Upper'] = None
        
        with col_res2:
            st.metric("可執行參與率 (Achievable PR)", f"{cal_pr:.2%}", delta=f"預算 {exec_budget_pct:.2%}")

    else: # Call Spread
        budget_per_unit = spot_price * exec_budget_pct 
        
        if calc_target == "鎖定 PR ➔ 求 Cap":
            target_pr = target_val_input
            target_cost = call_atm_ask_exec - (budget_per_unit / target_pr)
            
            if target_cost <= 0:
                st.error("預算過高或目標 PR 過低，無需賣出 Call")
                specs_output['K_Upper'] = None
            else:
                target_theoretical_short = target_cost / (1 - opt_spread_cost)
                k_cap_exec, _ = solve_strike_dynamic(target_theoretical_short, spot_price, 1.0, r_rf, div_q, sigma_atm, skew_slope, opt_spread_cost)
                cap_rate = (k_cap_exec / spot_price) - 1
                
                specs_output['PR'] = target_pr
                specs_output['Cap'] = cap_rate
                specs_output['K_Lower'] = spot_price
                specs_output['K_Upper'] = k_cap_exec

                with col_res2:
                    st.metric("推算封頂 (Achievable Cap)", f"{cap_rate:.2%}", delta=f"Strike: {k_cap_exec:.0f}")
                    
        else: # 鎖定 Cap ➔ 求 PR
            target_cap = target_val_input
            k_cap_exec = spot_price * (1 + target_cap)
            
            vol_cap = get_vol_at_strike(k_cap_exec, spot_price, sigma_atm, skew_slope)
            c_short = bs_price(spot_price, k_cap_exec, 1.0, r_rf, div_q, vol_cap, 'call')
            c_short_bid = c_short * (1 - opt_spread_cost)
            
            spread_cost = call_atm_ask_exec - c_short_bid
            
            if spread_cost <= 0:
                st.error("組合單成本異常")
                specs_output['K_Upper'] = None
            else:
                cal_pr = budget_per_unit / spread_cost
                
                specs_output['PR'] = cal_pr
                specs_output['Cap'] = target_cap
                specs_output['K_Lower'] = spot_price
                specs_output['K_Upper'] = k_cap_exec
                
                with col_res2:
                    st.metric("可執行參與率 (Achievable PR)", f"{cal_pr:.2%}", delta=f"鎖定 Cap {target_cap:.2%}")

    st.markdown("---")

    # --- 3. 下單指令生成 (Ticket Generator) ---
    st.markdown("#### 3. 交易指令 (Execution Ticket)")
    
    if specs_output.get('K_Upper') or exec_strategy == "Call Only (單邊買權)":
        lots_raw = (notional_amt * specs_output['PR']) / (spot_price * contract_mult)
        lots_round = int(round(lots_raw))
        
        ticket_data = []
        
        ticket_data.append({
            "Direction": "BUY",
            "Type": "Call",
            "Strike": f"{specs_output['K_Lower']:.0f}",
            "Product": "SPY FLEX (European/Cash)",
            "Lots": lots_round,
            "Est. Price": f"{call_atm_ask_exec:.2f}"
        })
        
        if specs_output.get('K_Upper'):
            vol_cap = get_vol_at_strike(specs_output['K_Upper'], spot_price, sigma_atm, skew_slope)
            c_short = bs_price(spot_price, specs_output['K_Upper'], 1.0, r_rf, div_q, vol_cap, 'call')
            c_short_bid = c_short * (1 - opt_spread_cost)
            
            ticket_data.append({
                "Direction": "SELL",
                "Type": "Call",
                "Strike": f"{specs_output['K_Upper']:.0f}",
                "Product": "SPY FLEX (European/Cash)",
                "Lots": lots_round,
                "Est. Price": f"{c_short_bid:.2f}"
            })
        
        df_ticket = pd.DataFrame(ticket_data)
        st.table(df_ticket)
        
        if input_mode == "方式 B：依權利金總額 ($)":
            budget_str = f"USD {total_premium_amt:,.0f} ({exec_budget_pct:.2%})"
        else:
            budget_str = f"{exec_budget_pct:.2%} (USD {total_premium_amt:,.0f})"

        if specs_output.get('K_Upper'):
            trade_text = f"""
[RFQ] SPY FLEX Options (European/Cash)
Notional: USD {notional_amt:,.0f}
Structure: {specs_output['PR']:.0%} PR / {specs_output['Cap']:.2%} Cap
---------------------------
BUY {lots_round}x SPY 1Y {specs_output['K_Lower']:.0f} Call
SELL {lots_round}x SPY 1Y {specs_output['K_Upper']:.0f} Call
---------------------------
Net Budget: {budget_str}
            """
        else:
            trade_text = f"""
[RFQ] SPY FLEX Options (European/Cash)
Notional: USD {notional_amt:,.0f}
Structure: {specs_output['PR']:.2%} PR (Uncapped)
---------------------------
BUY {lots_round}x SPY 1Y {specs_output['K_Lower']:.0f} Call
---------------------------
Net Budget: {budget_str}
            """
            
        st.caption("複製下方指令")
        st.code(trade_text.strip(), language="text")