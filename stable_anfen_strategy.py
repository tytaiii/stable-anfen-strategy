"""
Stable-Anfen Strategy (1x no-leverage edition)
=============================================
Built on the Retrogade-Assassin V5 live-trading base, stripped of all
leverage and secondary components:
1. Keeps the full ML signal system (Expanding Window + time decay)
2. Constant 1x leverage only; no vol limiter / drawdown protection / cooldown
3. No Scout Gun gambling components - single-gun style execution
4. Keeps the 70/30 momentum rotation framework

Designed for conservative, low-risk strategy deployment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

import os
from pathlib import Path

# Data directory: defaults to this repo's data/, overridable via ANFEN_DATA_DIR
DATA_DIR = os.environ.get('ANFEN_DATA_DIR', str(Path(__file__).resolve().parent / 'data'))

HORIZONS_DAYS = [1, 3, 5, 7, 14, 30]
HORIZONS_BARS = [6, 18, 30, 42, 84, 180]

def load_data(coin='btc'):
    price_df = pd.read_csv(f'{DATA_DIR}/{coin}usdt_4h.csv')
    price_df['datetime'] = pd.to_datetime(price_df['datetime'])
    price_df = price_df.sort_values('datetime').set_index('datetime')

    dvol_df = pd.read_csv(f'{DATA_DIR}/deribit_dvol_btc_4h.csv')
    time_col = dvol_df.columns[0]
    dvol_df[time_col] = pd.to_datetime(dvol_df[time_col])
    dvol_df = dvol_df.sort_values(time_col).set_index(time_col)

    if 'close' in dvol_df.columns:
        if 'dvol' in dvol_df.columns:
            dvol_df = dvol_df.drop(columns=['dvol'])
        dvol_df = dvol_df.rename(columns={'close': 'dvol'})

    dvol_series = dvol_df['dvol']
    if isinstance(dvol_series, pd.DataFrame):
        dvol_series = dvol_series.iloc[:, 0]

    df = price_df.join(dvol_series.rename('dvol'), how='inner').dropna(subset=['dvol'])
    return df

import time
def calculate_ml_expanding(df, feature_cols, retrain_freq=42, min_train_bars=1500, critical_pct=95):
    predictions = {H: np.full(len(df), np.nan) for H in HORIZONS_BARS}

    print(f"  [ML] Expanding Window Rolling Training (Frequency={retrain_freq} bars, Warmup={min_train_bars} bars, Anchoring={100-critical_pct}%)...")

    sc = time.time()
    for t_calc in range(min_train_bars, len(df), retrain_freq):
        t_end = min(t_calc + retrain_freq, len(df))
        target_slice = df.iloc[t_calc:t_end]
        valid_target = target_slice.dropna(subset=feature_cols)

        if len(valid_target) == 0:
            continue

        hist_feats = df.iloc[:t_calc][feature_cols].dropna()
        if len(hist_feats) < 100:
            continue

        scaler = StandardScaler()
        scaler.fit(hist_feats)
        X_pred = scaler.transform(valid_target[feature_cols])

        for H in HORIZONS_BARS:
            max_train_idx = t_calc - 1 - H
            if max_train_idx <= 0:
                continue

            train_slice = df.iloc[:max_train_idx].dropna(subset=feature_cols + [f'y_{H}'])
            if len(train_slice) < 500:
                continue

            X_train = scaler.transform(train_slice[feature_cols])
            y_train = train_slice[f'y_{H}'].values

            if len(np.unique(y_train)) < 2:
                continue

            model = LogisticRegression(C=1.0, random_state=42)

            # --- Time decay law (18-month half-life) + Critical Event Anchoring ---
            # Layer 1: gentle exponential decay (half-life = 540 days = 3240 bars)
            N_samples = len(train_slice)
            half_life_bars = 3240
            distances = np.arange(N_samples) - (N_samples - 1)
            decay_weights = np.exp(distances * np.log(2) / half_life_bars)

            # Layer 2: Critical event anchoring
            fwd_ret_col = f'y_{H}'
            fwd_prices = df['close'].values
            fwd_ret_abs = np.zeros(N_samples)
            for si in range(N_samples):
                orig_idx = train_slice.index[si]
                orig_pos = df.index.get_loc(orig_idx)
                target_pos = min(orig_pos + H, len(fwd_prices) - 1)
                if fwd_prices[orig_pos] > 0:
                    fwd_ret_abs[si] = abs(fwd_prices[target_pos] / fwd_prices[orig_pos] - 1)

            if len(fwd_ret_abs) > 20:
                critical_threshold = np.percentile(fwd_ret_abs, critical_pct)
                is_critical = fwd_ret_abs >= critical_threshold
                decay_weights[is_critical] = 1.0

            model.fit(X_train, y_train, sample_weight=decay_weights)

            preds = model.predict_proba(X_pred)[:, 1]

            for i, idx in enumerate(valid_target.index):
                int_pos = df.index.get_loc(idx)
                predictions[H][int_pos] = preds[i]

    print(f"  [ML] Rolling training done in {time.time()-sc:.2f}s")

    for H in HORIZONS_BARS:
        df[f'p_{H}'] = predictions[H]

    p_cols = [df[f'p_{H}'] for H in HORIZONS_BARS]
    df['safety_score'] = np.nanmedian(p_cols, axis=0)

    return df

def feature_engineering(df):
    # Core indicators
    df['r'] = np.log(df['close'] / df['close'].shift(1))
    df['zscore'] = (df['dvol'] - df['dvol'].rolling(60).mean()) / df['dvol'].rolling(60).std()

    # 90-day DVOL Z-score for Scout Greedy Algorithm
    df['z_dvol_90'] = (df['dvol'] - df['dvol'].rolling(540).mean()) / df['dvol'].rolling(540).std()

    # ATR14
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    df['vol'] = df['atr14'] / df['close']

    # Base strategy triggers
    df['dz'] = df['zscore'].diff()
    df['dz_lag1'] = df['dz'].shift(1)
    df['trigger'] = ((df['dz_lag1'] < 0) & (df['dz'] > 0)).astype(int)

    # ML Features (Clean V4: drop collinear redundant features, keep only z and dz)
    df['z'] = df['zscore']

    # Labels y_H
    for H in HORIZONS_BARS:
        fwd_ret = df['close'].shift(-H) / df['open'].shift(-1) - 1
        df[f'y_{H}'] = (fwd_ret > 0).astype(int)
        df.loc[df.index[-H:], f'y_{H}'] = np.nan

    return df

def run_strategy(df, mode='1x', threshold=0.51, U=0.60, L_min=0.5, L_max=5.0, p=2, fee_rate=0.0007, delay_method=None, T_short=0.50, T_long=0.52, max_wait=30):
    # Core settings from User Request
    """
    mode:
      '1x': basic mode, 1x leverage, flat if S<T
      'lev': dynamic leverage
      'lev_vol': dynamic leverage + vol limiter
      'lev_vol_cb': dynamic leverage + vol limiter + circuit breaker (DD penalty + CD)
    """
    cost = fee_rate
    closes = df['close'].values
    opens = df['open'].values
    triggers = df['trigger'].values
    rs = df['r'].values
    vols = df['vol'].values
    times = df.index

    scores = df['safety_score'].values if 'safety_score' in df.columns else np.ones(len(df))

    # Fetch short-horizon score array early for the Queued logic
    if 'p_6' in df.columns:
        s_short_arr = df[['p_6', 'p_18', 'p_30']].mean(axis=1).values
    else:
        s_short_arr = np.zeros(len(df))

    positions = np.zeros(len(df))
    levs = np.zeros(len(df))
    trades = []

    pos = 0
    current_lev = 0.0
    entry_price = 0
    entry_time = None

    queued_trigger_timer = 0

    # Track statistics
    metrics = {'rej_S': 0, 'rej_vol': 0, 'rej_cb': 0, 'rej_delay': 0}

    equity = 1.0
    max_equity = 1.0
    equity_curve = np.ones(len(df))

    for t in range(1, len(df) - 1):
        trigger_active = (triggers[t] == 1)

        # Mark-to-market each bar to track true MDD
        if pos == 1:
            pnl_base_mtm = closes[t] / entry_price - 1
            mtm_equity = equity * (1 + pnl_base_mtm * current_lev)
        else:
            mtm_equity = equity

        max_equity = max(max_equity, mtm_equity)
        current_dd = mtm_equity / max_equity - 1
        equity_curve[t] = mtm_equity

        decision = 0
        score_t = scores[t]

        # Accept static float or dynamic numpy.ndarray / pandas.Series thresholds
        try:
            th_t = threshold[t]
        except:
            th_t = threshold

        gate_open = True
        if np.isnan(score_t) or score_t < th_t:
            gate_open = False

        if not gate_open:
            if pos == 1:
                decision = -1
            else:
                if trigger_active and rs[t] > 0:
                    metrics['rej_S'] += 1
                queued_trigger_timer = 0
                continue

        # Gate is open
        if delay_method == 'queued':
            if pos == 0:
                if trigger_active and rs[t] > 0:
                    if s_short_arr[t] >= T_short:
                        decision = 1
                        queued_trigger_timer = 0
                    else:
                        metrics['rej_delay'] += 1
                        queued_trigger_timer = max_wait
                        decision = 0
                elif queued_trigger_timer > 0:
                    queued_trigger_timer -= 1
                    if s_short_arr[t] >= T_short:
                        decision = 1
                        queued_trigger_timer = 0
            elif pos == 1:
                if trigger_active and rs[t] < 0:
                    decision = -1
                    queued_trigger_timer = 0
        else:
            # Baseline: follow trigger directly
            if trigger_active:
                if rs[t] > 0 and pos == 0:
                    decision = 1
                elif rs[t] < 0 and pos == 1:
                    decision = -1

        if decision == 1 and pos == 0:
            # Determine leverage L_entry
            L_entry = 0.0 # Default to 0, will be set if conditions met
            if mode == '1x':
                L_entry = 1.0
            elif mode in ['lev', 'lev_vol', 'lev_vol_cb']:
                x = np.clip((score_t - th_t) / (U - th_t), 0, 1)
                # Ensure L_entry is extracted cleanly as a float scalar just in case
                L_entry = float(L_min + (L_max - L_min) * (x ** p))
            else:
                L_entry = 1.0 # Fallback for unknown modes

            if mode in ['lev_vol', 'lev_vol_cb']:
                vol_t = vols[t]
                if vol_t > 0.03:
                    L_entry = 0 # Reject trade if vol too high
                    metrics['rej_vol'] += 1
                elif vol_t > 0.02:
                    L_entry = min(L_entry, 2.0) # Cap leverage if vol is moderately high

            if mode == 'lev_vol_cb' and L_entry > 0: # Only apply CB if trade is still on
                # Account-level DD protection: if off-high drawdown > 15%, emergency-cap leverage to 1.0x
                if current_dd < -0.15:
                    if L_entry > 1.0:
                        metrics['rej_cb'] += 1
                    L_entry = min(L_entry, 1.0)

            if L_entry > 0: # Only enter if leverage is positive
                pos = 1
                current_lev = L_entry
                entry_time = times[t + 1]
                entry_price = opens[t + 1] * (1 + cost * current_lev)
            else:
                decision = 0 # If L_entry became 0 due to vol/CB, cancel decision

        elif decision == -1 and pos == 1:
            exit_time = times[t + 1]
            exit_price = opens[t + 1] * (1 - cost * current_lev)
            pnl_base = exit_price / entry_price - 1
            pnl_lev = pnl_base * current_lev

            equity *= (1 + pnl_lev)
            max_equity = max(max_equity, equity)


            trades.append({
                'entry_time': entry_time, 'entry_price': entry_price,
                'exit_time': exit_time, 'exit_price': exit_price,
                'pnl': pnl_lev, 'lev': current_lev
            })
            pos = 0
            current_lev = 0.0

        positions[t] = pos
        levs[t] = current_lev

    # Force-close any open position on the last bar
    if pos == 1:
        exit_price = closes[-1] * (1 - cost * current_lev)
        pnl_base = exit_price / entry_price - 1
        pnl_lev = pnl_base * current_lev
        equity *= (1 + pnl_lev)

        trades.append({
            'entry_time': entry_time, 'entry_price': entry_price,
            'exit_time': times[-1], 'exit_price': exit_price,
            'pnl': pnl_lev, 'lev': current_lev
        })

    equity_curve[-1] = equity

    df_eval = df.copy()
    df_eval['pos'] = pd.Series(positions, index=df_eval.index).shift(1).fillna(0)
    df_eval['lev'] = pd.Series(levs, index=df_eval.index).shift(1).fillna(0)
    df_eval['equity'] = equity_curve

    return df_eval, pd.DataFrame(trades), metrics

def calc_stats_complex(df_eval, trades_df, metrics):
    if len(df_eval) == 0:
        return {'ret':0, 'ann_ret':0, 'mdd':0, 'trades':0, 'exp':0, 'sharpe': 0, 'calmar': 0, 'win_rate': 0, 'pf': 0, 'avg_l': 0, 'l_dist': 'N/A', 'mdd_26': 0, 'rej_S': 0, 'rej_vol': 0, 'rej_cb': 0}
    years = (df_eval.index[-1] - df_eval.index[0]).days / 365.25
    ret = df_eval['equity'].iloc[-1] / df_eval['equity'].iloc[0] - 1
    ann_ret = (1 + ret) ** (1/years) - 1 if years > 0 and ret > -1 else 0
    rebased = df_eval['equity'] / df_eval['equity'].iloc[0]
    mdd = (rebased / rebased.cummax() - 1).min()
    exp = trades_df['pnl'].mean() if len(trades_df) > 0 else 0

    # Sharpe Ratio (daily returns -> annualized)
    daily_ret = df_eval['equity'].resample('D').last().pct_change().dropna()
    sharpe = np.sqrt(365) * daily_ret.mean() / daily_ret.std() if len(daily_ret) > 1 and daily_ret.std() != 0 else 0

    # Calmar Ratio
    calmar = abs(ann_ret / mdd) if mdd != 0 else 0

    # Win Rate & Profit Factor
    if len(trades_df) > 0:
        win_rate = (trades_df['pnl'] > 0).mean()
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss != 0 else np.inf
    else:
        win_rate = 0
        pf = 0

    # monthly mdd
    monthly_ret = df_eval['equity'].resample('ME').last().pct_change()
    mdd_2026 = 0
    if not monthly_ret[monthly_ret.index.year == 2026].empty:
        mdd_2026 = monthly_ret[monthly_ret.index.year == 2026].min()

    avg_l = trades_df['lev'].mean() if len(trades_df) > 0 else 0
    l_bins = [0.5, 1.0, 2.0, 3.0, 4.0, 5.01]
    if len(trades_df) > 0:
        hist, _ = np.histogram(trades_df['lev'], bins=l_bins)
        hist_pct = hist / len(trades_df) * 100
        l_dist_str = ", ".join([f"[{l_bins[i]}-{l_bins[i+1]}): {p:.1f}%" for i, p in enumerate(hist_pct)])
    else:
        l_dist_str = "N/A"

    return {
        'ret': ret,
        'ann_ret': ann_ret,
        'mdd': mdd,
        'trades': len(trades_df),
        'exp': exp,
        'sharpe': sharpe,
        'calmar': calmar,
        'win_rate': win_rate,
        'pf': pf,
        'avg_l': avg_l,
        'l_dist': l_dist_str,
        'mdd_26': mdd_2026,
        'rej_S': metrics['rej_S'],
        'rej_vol': metrics['rej_vol'],
        'rej_cb': metrics['rej_cb']
    }

# ========================================================================
# [Dual-Gun Architecture]: second gun and weight-blending logic
# (Scout backup gun). Remove this block and all its calls in main() if the
# dual-gun architecture is not needed.
# Contents: 1. Adaptive trend MA strategy (MA10/MA20 entry/exit, ATR dynamic leverage)
#           2. Dynamic weight mixer (Sigmoid offset logic using Z_MA200 to
#              automatically balance the two guns)
# ========================================================================
def scout_strategy_greedy_dvol(df, cost=0.0007, risk_target=0.03, max_lev=5.0, min_lev=0.5):
    """Dual-gun architecture scout gun (clean V4: complex if-else patches and
    losing-streak penalties removed)"""
    import numpy as np
    import pandas as pd
    closes = df['close'].values; opens = df['open'].values; highs = df['high'].values; lows = df['low'].values; times = df.index
    tr_all = np.vstack([highs-lows, np.abs(highs-np.concatenate(([0], closes[:-1]))), np.abs(lows-np.concatenate(([0], closes[:-1])))]).max(axis=0)
    atrs = pd.Series(tr_all).rolling(14).mean().shift(1).values
    pos=0; entry_price=0; current_lev=0.0; equity=1.0
    equity_curve=np.ones(len(df)); trades=[]; trade_start_eq=1.0; eq_curr=1.0

    for t in range(120, len(df)-1):
        momentum = (closes[t]-closes[t-20])/closes[t-20]
        hh20, atr = np.max(highs[t-20:t]), atrs[t]
        weight = 1.0  # NO STREAK MULTIPLIER, NO IF-ELSE PATCHES

        if pos==1: mtm_equity = equity*(1+(closes[t]/entry_price-1)*current_lev*weight)
        else: mtm_equity = equity
        equity_curve[t] = mtm_equity

        decision = 0
        if pos==0 and momentum>0 and closes[t]>hh20: decision = 1
        elif pos==1 and closes[t]<np.min(lows[t-10:t]): decision = -1

        if decision==1 and pos==0:
            safe_lev = 1.0  # Stable-Anfen strategy: scout gun fixed at 1x too
            pos=1; current_lev=safe_lev; entry_price=opens[t+1]*(1+cost*current_lev)
            entry_time=times[t+1]; trade_start_eq=eq_curr

        elif decision==-1 and pos==1:
            exit_price=opens[t+1]*(1-cost*current_lev); exit_time=times[t+1]
            pnl_lev=(exit_price/entry_price-1)*current_lev
            trades.append({'entry_time':entry_time, 'entry_price':entry_price, 'exit_time':exit_time, 'exit_price':exit_price, 'pnl':pnl_lev, 'lev':current_lev, 'weight':weight})
            eq_curr*=(1+pnl_lev); equity*=(1+pnl_lev)
            pos=0; current_lev=0.0

    if pos==1:
        pnl_lev=(closes[-1]*(1-cost*current_lev)/entry_price-1)*current_lev
        equity*=(1+pnl_lev)
        trades.append({'entry_time':entry_time, 'exit_time':times[-1], 'pnl':pnl_lev, 'lev':current_lev, 'weight':weight})

    equity_curve[-1] = equity; df_eval=df.copy(); df_eval['equity']=equity_curve
    metrics = {'rej_S': 0, 'rej_vol': 0, 'rej_cb': 0}
    return df_eval, pd.DataFrame(trades), metrics

def run_dual_gun_blend(df, d_full_lev, t_full_lev, fee_rate=0.0007):
    """Blend the main gun (base protection gun) with the backup gun (scout gun),
    returning the final dual-gun equity curve"""
    # 1. Run scout gun independently (using D1 generalized greedy algorithm)
    d_scout, t_scout, m_scout = scout_strategy_greedy_dvol(df, cost=fee_rate)

    # 2. Macro moving-average calculation (for non-linear dual-gun weight allocation)
    # Optimization hint: set MAX_W to 0 if non-linear dynamic weights are not needed.
    df_w = df.copy()
    df_w['MA_200d'] = df_w['close'].rolling(1200).mean().shift(1)  # 200 days = 1200 4H bars
    df_w['Z_MA200'] = (df_w['close'] - df_w['MA_200d']) / df_w['MA_200d'] * 100

    # 3. Extract the two guns' return series
    raw_v4 = d_full_lev['equity'].values
    raw_scout = d_scout['equity'].values

    ret_v4 = np.insert(np.diff(raw_v4) / raw_v4[:-1], 0, 0)
    ret_scout = np.insert(np.diff(raw_scout) / raw_scout[:-1], 0, 0)

    # 4. Dynamic weight setup (centralized tuning)
    MAX_W = 0.40  # Scout gun max weight 40% (main gun at least 60%)
    K = 0.3       # Sigmoid slope
    w_scout_sig = MAX_W / (1 + np.exp(-K * df_w['Z_MA200'].values))
    w_scout_sig = np.nan_to_num(w_scout_sig, nan=0.0)
    w_v4_sig = 1.0 - w_scout_sig # main gun weight

    # 5. Fuse the dual-gun equity
    ret_dual = w_v4_sig * ret_v4 + w_scout_sig * ret_scout
    eq_dual_curve = np.cumprod(1 + ret_dual)

    # 6. Return the dual-gun equity df_eval
    d_dual = df.copy()
    d_dual['equity'] = eq_dual_curve

    # Merge trade records of both guns (display/stats only)
    combined_trades = pd.concat([t_full_lev, t_scout], ignore_index=True) if len(t_scout) > 0 else t_full_lev

    return d_dual, combined_trades, w_scout_sig


# ========================================================================
# [Main program] BTC dual-gun strategy only
# ========================================================================
def evaluate_segment(res_df, start_date, end_date=None):
    mask = (res_df.index >= start_date)
    if end_date:
        mask = mask & (res_df.index < end_date)
    seg_df = res_df[mask]
    if len(seg_df) < 2:
        return 0, 0, 0
    ret = seg_df['equity'].iloc[-1] / seg_df['equity'].iloc[0] - 1
    days = (seg_df.index[-1] - seg_df.index[0]).days
    years = max(days / 365.25, 0.001)
    cagr = (1 + ret) ** (1/years) - 1

    rebased = seg_df['equity'] / seg_df['equity'].iloc[0]
    mdd = (rebased / rebased.cummax() - 1).min()

    daily_eq = seg_df['equity'].resample('1D').last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    vol_ann = daily_ret.std() * np.sqrt(365)
    sharpe = cagr / vol_ann if vol_ann > 0 else 0

    return cagr, mdd, sharpe

def run_single_coin(coin, threshold=0.52, fee_rate=0.0007):
    """Run the full Stable-Anfen strategy for a single coin; returns the
    dual-gun equity DataFrame"""
    df = load_data(coin=coin)
    df = feature_engineering(df)

    feature_cols = ['z', 'dz']
    df = df.dropna(subset=feature_cols + ['r', 'trigger'])

    # Expanding-window ML on the full history
    start_time_idx = df.index.searchsorted(pd.to_datetime('2022-06-01'))
    df = calculate_ml_expanding(df, feature_cols, retrain_freq=42, min_train_bars=start_time_idx)

    import os
    th_path = f'{DATA_DIR}/{coin}_80D_wfa.csv'
    if os.path.exists(th_path):
        th_df = pd.read_csv(th_path, index_col=0, parse_dates=True)
        th_df.index = pd.to_datetime(th_df.index)

        th_df = th_df.rename(columns={'best_th': 'dynamic_th'})
        df_merged = df.join(th_df[['dynamic_th']], how='left')
        df['dynamic_th'] = df_merged['dynamic_th'].ffill().bfill()
        run_threshold = df['dynamic_th'].values
        print(f"  [{coin.upper()}] 80D WFA + 70/30/0 champion defense loaded!")
    else:
        print(f"  [{coin.upper()}] WARNING: 80D WFA matrix file missing, falling back to static threshold {threshold}!!!")
        run_threshold = threshold

    # Stable-Anfen strategy: constant 1x leverage, no vol limiter / DD protection / cooldown
    _T_short = 0.50 if coin == 'bnb' else 0.54
    d_full, t_full, m_full = run_strategy(df, mode='1x', threshold=run_threshold,
                                          delay_method='queued', T_short=_T_short, max_wait=12, fee_rate=fee_rate)
    # Scout gun also locked to 1x leverage
    d_dual, t_dual, _ = run_dual_gun_blend(df, d_full, t_full, fee_rate=fee_rate)
    d_dual.index = pd.to_datetime(d_dual.index)

    print(f"  [{coin.upper()}] Rejected: {m_full['rej_delay']} | Main-gun trades: {len(t_full)}")
    return d_dual, t_dual

def build_momentum_rotation(coin_data_dict, lookback_bars=120, weights_dict={1: 0.7, 2: 0.3, 3: 0.0}):
    """
    Generic momentum rotation engine (core strategy function; do NOT reimplement
    externally to avoid indicator divergence). Uses strict deterministic 20-day
    momentum (lookback_bars=120) to synthesize the portfolio equity and
    pass-through trade records.

    :param coin_data_dict: dict, {'btc': (d_dual, t_dual), 'eth': ...}
    :param lookback_bars: momentum window, 4H: 120 bars = 20 days (final parameter!)
    :param weights_dict: weight assigned to each rank
    :return: (portfolio equity DataFrame, portfolio trade records DataFrame)
    """
    coins = list(coin_data_dict.keys())

    eqs_df = {c: coin_data_dict[c][0] for c in coins}
    trades_df = {c: coin_data_dict[c][1] for c in coins}

    common_idx = eqs_df[coins[0]].index
    for c in coins[1:]:
        common_idx = common_idx.intersection(eqs_df[c].index)

    rets = {}
    for c in coins:
        rets[c] = eqs_df[c].loc[common_idx, 'equity'].pct_change().fillna(0)

    df_returns = pd.DataFrame(rets)
    df_equity = (1 + df_returns).cumprod()

    mom = df_equity.pct_change(lookback_bars)
    ranks = mom.rank(axis=1, ascending=False, method='first')

    w_df = pd.DataFrame(index=df_returns.index, columns=coins)
    for c in coins:
        w_df[c] = sum(np.where(ranks[c] == r, wt, 0) for r, wt in weights_dict.items())

    w_df = w_df.shift(1).fillna(0)

    ret_rot = sum(w_df[c] * df_returns[c] for c in coins)
    eq_rot = (1 + ret_rot).cumprod()
    d_combo = pd.DataFrame({'equity': eq_rot}, index=common_idx)

    rot_trades = []
    for c in coins:
        tdf = trades_df[c].copy()
        if len(tdf) > 0:
            if 'entry_time' not in tdf.columns:
                continue
            weights_at_entry = w_df[c].reindex(pd.to_datetime(tdf['entry_time'])).fillna(0).values
            tdf['port_weight'] = weights_at_entry
            tdf = tdf[tdf['port_weight'] > 0].copy()
            tdf['pnl'] = tdf['pnl'] * tdf['port_weight']
            rot_trades.append(tdf)

    if rot_trades:
        rot_trades_df = pd.concat(rot_trades, ignore_index=True)
        rot_trades_df = rot_trades_df.sort_values('entry_time').reset_index(drop=True)
    else:
        rot_trades_df = pd.DataFrame()

    return d_combo, rot_trades_df



def run_full_strategy():
    """Stable-Anfen Strategy - BTC/BNB momentum rotation 70/30 (1x no-leverage edition)"""
    print("=" * 60)
    print("  Stable-Anfen Strategy - 1x no-leverage momentum rotation (BTC+BNB, 70/30)")
    print("=" * 60)

    THRESHOLD = 0.52
    COINS = ['btc', 'bnb']

    coin_data = {}
    for i, coin in enumerate(COINS):
        print(f"\n[{i+1}/{len(COINS)}] Running {coin.upper()} 80D WFA polar defense independently...")
        d_dual, t_dual = run_single_coin(coin, threshold=THRESHOLD)
        coin_data[coin] = (d_dual, t_dual)

    # Strict dual-core pure rotation weights (edge coins removed to eliminate frictional noise)
    d_combo, t_combo = build_momentum_rotation(coin_data, lookback_bars=120, weights_dict={1: 0.7, 2: 0.3})

    results = {f'{c.upper()} Solo': coin_data[c][0] for c in COINS}
    results['Dual-Core Rotation 70/30'] = d_combo

    def print_stats(seg_results, title):
        print(f"\n{'='*60}\n  {title}\n{'='*60}")
        for name, seg_df in seg_results.items():
            if len(seg_df) < 2: continue
            ret = seg_df['equity'].iloc[-1] / seg_df['equity'].iloc[0] - 1
            days = (seg_df.index[-1] - seg_df.index[0]).days
            years = max(days / 365.25, 0.001)
            cagr = (1 + ret) ** (1/years) - 1
            rebased = seg_df['equity'] / seg_df['equity'].iloc[0]
            mdd = (rebased / rebased.cummax() - 1).min()
            daily_eq = seg_df['equity'].resample('1D').last().dropna()
            daily_ret = daily_eq.pct_change().dropna()
            vol_ann = daily_ret.std() * np.sqrt(365)
            sharpe = cagr / vol_ann if vol_ann > 0 else 0
            print(f"[{name:<14}] CAGR: {cagr*100:>5.1f}% | MDD: {mdd*100:>6.1f}% | Sharpe: {sharpe:>4.2f}")

    print_stats({k: v[v.index < '2024-01-01'] for k, v in results.items()}, "Pre-2024 (2021-2023)")
    print_stats({k: v[v.index >= '2024-01-01'] for k, v in results.items()}, "Post-2024 (2024-present)")

    # Full-period (4-year) absolute drawdown
    print(f"\n{'='*60}\n  4-Year Full-Period Extreme Stress Summary\n{'='*60}")
    for name, seg_df in results.items():
        if len(seg_df) < 2: continue
        rebased = seg_df['equity'] / seg_df['equity'].iloc[0]
        mdd = (rebased / rebased.cummax() - 1).min()
        print(f"[{name:<14}] 4-year absolute deepest drawdown: {mdd*100:>6.1f}%")

    # Chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Stable-Anfen Strategy: [BTC+BNB] 1x No-Leverage Momentum Rotation (all live fees included)', fontsize=20, fontweight='bold', y=0.94)

    coin_colors = ['#f39c12', '#3498db']
    colors = {f'{c.upper()} Solo': coin_colors[i] for i, c in enumerate(COINS)}
    colors['Dual-Core Rotation 70/30'] = '#e74c3c'
    lw = {k: 1.5 for k in colors}; lw['Dual-Core Rotation 70/30'] = 3.0

    for name, r in results.items():
        s = r[r.index < '2024-01-01']
        if len(s) > 0: ax1.plot(s.index, s['equity']/s['equity'].iloc[0], label=name, color=colors[name], linewidth=lw[name])
    ax1.set_title("Pre-2024 Macro Mapping (2021-2023 cross bull/bear foundation)", fontsize=16, fontweight='bold')
    ax1.set_ylabel('Equity (Base=1.0)'); ax1.legend(fontsize=12, loc='upper left'); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')

    for name, r in results.items():
        s = r[r.index >= '2024-01-01']
        if len(s) > 0: ax2.plot(s.index, s['equity']/s['equity'].iloc[0], label=name, color=colors[name], linewidth=lw[name])
    ax2.set_title("Post-2024 High-Frequency Combat (2024-present chaotic meat-grinder)", fontsize=16, fontweight='bold')
    ax2.set_ylabel('Equity (Base=1.0)'); ax2.legend(fontsize=12, loc='upper left'); ax2.grid(True, alpha=0.3); ax2.set_yscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = f"{DATA_DIR}/stable_anfen_strategy_backtest.png"
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Full chart saved to: {out_path}")

if __name__ == '__main__':
    run_full_strategy()

    THRESHOLD = 0.52

