"""

SAP Business Data Cloud — Record-to-Report (R2R)
Month-End / Year-End Financial Close Analytics Pipeline
------------------------------------------------------------
Capstone Project  | Anwesha Raha | Roll No: 23051248
Batch: 2027 SAP (OE) | Specialization: SAP Data Analytics Engineer
Course: SAP Business Data Cloud (C_BDC)
Business Scenario:
  Record-to-Report (R2R) is the financial accounting process that spans
  from capturing journal entries through period-end close to publishing
  financial statements. Finance controllers need real-time visibility into
  GL account balances, cost centre variances, profit & loss trends, and
  DSO (Days Sales Outstanding) — all within SAP BDC.

Pipeline Architecture (mirrors real SAP BDC deployment):
  SAP S/4HANA (Source)
    └─► SAP Datasphere (Ingestion via SDI / SLT)
          ├─► BW/4HANA Bridge  ──► OLAP Financial Queries
          ├─► SAP Databricks   ──► Period-End Aggregations (Spark)
          └─► SAP Analytics Cloud ──► Financial Dashboard (6 panels)
                    └─► Data Products (Datasphere Marketplace CSVs)

"""

import pandas as pd
import numpy as np
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import warnings, os
warnings.filterwarnings('ignore')

# SAP brand palette 
SAP_BLUE   = '#003366'
SAP_GOLD   = '#F0AB00'
SAP_GREEN  = '#107E3E'
SAP_RED    = '#BB0000'
SAP_TEAL   = '#0070F2'
SAP_GRAY   = '#6A6D70'
SAP_LGRAY  = '#F4F6F8'
WATERMARK  = 'SAP BDC | Anwesha Raha | 23051248'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})


# STEP 1 — Simulate SAP S/4HANA Financial Source Data
#
# In real SAP BDC: GL journal entries arrive from S/4HANA via
# SAP Data Integration (SDI / SLT replication) into Datasphere.
# Tables simulated: BKPF (Accounting Doc Header), BSEG (Line Item),
#                   FAGLFLEXT (GL Totals), CSKA (Cost Elements)

np.random.seed(2024)
N = 600

COST_CENTRES  = ['CC_FINANCE', 'CC_SALES', 'CC_IT', 'CC_OPS', 'CC_HR', 'CC_MARKETING']
GL_ACCOUNTS   = {
    '400000': 'Revenue – Product Sales',
    '400100': 'Revenue – Service',
    '500000': 'Cost of Goods Sold',
    '600000': 'Salaries & Wages',
    '610000': 'Travel & Expenses',
    '620000': 'IT Infrastructure',
    '700000': 'Depreciation',
    '800000': 'Interest Expense',
}
PROFIT_CENTRES = ['PC_INDIA', 'PC_EUROPE', 'PC_USA', 'PC_APAC']
COMPANIES      = ['1000']   # Company Code (like SAP BUKRS)
PERIODS        = [f'P{str(m).zfill(2)}/2024' for m in range(1, 13)]  # P01–P12

start = datetime(2024, 1, 1)
dates = [start + timedelta(days=int(x)) for x in np.random.randint(0, 365, N)]

gl_codes  = np.random.choice(list(GL_ACCOUNTS.keys()), N,
             p=[0.18, 0.12, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10])
amounts   = []
for g in gl_codes:
    if g.startswith('4'):   # Revenue — positive
        amounts.append(np.round(np.random.lognormal(11.0, 0.9), 2))
    else:                   # Expense — negative
        amounts.append(-np.round(np.random.lognormal(10.5, 0.8), 2))

df_gl = pd.DataFrame({
    'DOC_NO'        : [f'1{str(5000000 + i)}' for i in range(N)],
    'POST_DATE'     : dates,
    'PERIOD'        : [f'P{str(d.month).zfill(2)}/2024' for d in dates],
    'FISCAL_YEAR'   : 2024,
    'COMPANY_CODE'  : '1000',
    'GL_ACCOUNT'    : gl_codes,
    'GL_DESC'       : [GL_ACCOUNTS[g] for g in gl_codes],
    'COST_CENTRE'   : np.random.choice(COST_CENTRES, N),
    'PROFIT_CENTRE' : np.random.choice(PROFIT_CENTRES, N),
    'AMOUNT_LC'     : amounts,          # Local Currency (INR)
    'CURRENCY'      : 'INR',
    'DOC_TYPE'      : np.random.choice(['SA', 'DR', 'KR', 'AB'], N,
                                        p=[0.40, 0.25, 0.25, 0.10]),
    'CLEARED'       : np.random.choice(['Y', 'N'], N, p=[0.72, 0.28]),
})

df_gl['MONTH']     = df_gl['POST_DATE'].dt.strftime('%b-%Y')
df_gl['MONTH_NUM'] = df_gl['POST_DATE'].dt.month
df_gl['IS_REVENUE']= df_gl['GL_ACCOUNT'].str.startswith('4').astype(int)
df_gl['IS_EXPENSE']= (~df_gl['GL_ACCOUNT'].str.startswith('4')).astype(int)

# AR Ageing simulation (for DSO calculation — SAP FI receivables)
np.random.seed(42)
M = 200
df_ar = pd.DataFrame({
    'INVOICE_NO'  : [f'INV-{90000+i}' for i in range(M)],
    'POST_DATE'   : [start + timedelta(days=int(x)) for x in np.random.randint(0, 330, M)],
    'DUE_DATE'    : [start + timedelta(days=int(x)) for x in np.random.randint(30, 365, M)],
    'PROFIT_CENTRE': np.random.choice(PROFIT_CENTRES, M),
    'INVOICE_AMT' : np.round(np.random.lognormal(11.2, 0.8, M), 2),
    'PAID_AMT'    : 0.0,
    'STATUS'      : np.random.choice(['OPEN', 'PAID', 'OVERDUE'], M, p=[0.35, 0.45, 0.20]),
})
df_ar['OVERDUE_DAYS'] = (datetime(2024, 12, 31) - df_ar['DUE_DATE']).dt.days.clip(lower=0)
df_ar.loc[df_ar['STATUS'] == 'PAID', 'PAID_AMT'] = df_ar.loc[df_ar['STATUS'] == 'PAID', 'INVOICE_AMT']

print("=" * 65)
print("  SAP BDC — Record-to-Report (R2R) Analytics Pipeline")
print("  Anwesha Raha | Roll No: 23051248 | Batch: 2027 SAP (OE)")
print("=" * 65)
print(f"\n[STEP 1] GL source data loaded: {len(df_gl)} journal entries, "
      f"{len(df_ar)} AR invoices.\n")

# STEP 2 — SAP BW/4HANA Bridge: Financial OLAP Queries
#
# In real BDC: these execute as BW Analytical Queries on
# CompositeProviders built over SAP S/4HANA CDS Views or
# migrated BW/4HANA ADSOs via BW Bridge in Datasphere.

con = duckdb.connect()
con.register('GL_FACT', df_gl)
con.register('AR_FACT', df_ar)

#  P&L Summary by Period (Period-End Close View)
pnl_by_period = con.execute("""
    SELECT
        PERIOD,
        MONTH_NUM,
        SUM(CASE WHEN IS_REVENUE=1 THEN AMOUNT_LC ELSE 0 END)  AS TOTAL_REVENUE,
        SUM(CASE WHEN IS_EXPENSE=1 THEN ABS(AMOUNT_LC) ELSE 0 END) AS TOTAL_EXPENSE,
        SUM(AMOUNT_LC) AS NET_PROFIT
    FROM GL_FACT
    GROUP BY PERIOD, MONTH_NUM
    ORDER BY MONTH_NUM
""").df()
pnl_by_period['MARGIN_PCT'] = (
    pnl_by_period['NET_PROFIT'] / pnl_by_period['TOTAL_REVENUE'].replace(0, np.nan) * 100
).round(2)

#  GL Account Balance Summary (Trial Balance)
trial_balance = con.execute("""
    SELECT
        GL_ACCOUNT,
        GL_DESC,
        SUM(AMOUNT_LC)      AS BALANCE,
        COUNT(DOC_NO)       AS POSTING_COUNT,
        AVG(ABS(AMOUNT_LC)) AS AVG_POSTING_AMT
    FROM GL_FACT
    GROUP BY GL_ACCOUNT, GL_DESC
    ORDER BY GL_ACCOUNT
""").df()

#  Cost Centre Expense Breakdown
cc_expenses = con.execute("""
    SELECT
        COST_CENTRE,
        SUM(ABS(AMOUNT_LC)) FILTER (WHERE IS_EXPENSE=1) AS TOTAL_EXPENSE,
        COUNT(DOC_NO)        FILTER (WHERE IS_EXPENSE=1) AS POSTING_COUNT,
        AVG(ABS(AMOUNT_LC))  FILTER (WHERE IS_EXPENSE=1) AS AVG_EXPENSE
    FROM GL_FACT
    GROUP BY COST_CENTRE
    ORDER BY TOTAL_EXPENSE DESC
""").df()

#  Profit Centre Revenue vs Expense
pc_pnl = con.execute("""
    SELECT
        PROFIT_CENTRE,
        SUM(CASE WHEN IS_REVENUE=1 THEN AMOUNT_LC  ELSE 0 END) AS REVENUE,
        SUM(CASE WHEN IS_EXPENSE=1 THEN ABS(AMOUNT_LC) ELSE 0 END) AS EXPENSE,
        SUM(AMOUNT_LC) AS NET
    FROM GL_FACT
    GROUP BY PROFIT_CENTRE
    ORDER BY REVENUE DESC
""").df()

#  AR Ageing Buckets (DSO Analysis)
ar_ageing = con.execute("""
    SELECT
        PROFIT_CENTRE,
        SUM(CASE WHEN OVERDUE_DAYS = 0          THEN INVOICE_AMT ELSE 0 END) AS CURRENT,
        SUM(CASE WHEN OVERDUE_DAYS BETWEEN 1 AND 30  THEN INVOICE_AMT ELSE 0 END) AS DAYS_1_30,
        SUM(CASE WHEN OVERDUE_DAYS BETWEEN 31 AND 60 THEN INVOICE_AMT ELSE 0 END) AS DAYS_31_60,
        SUM(CASE WHEN OVERDUE_DAYS > 60         THEN INVOICE_AMT ELSE 0 END) AS OVER_60,
        COUNT(INVOICE_NO) AS INVOICE_COUNT
    FROM AR_FACT
    GROUP BY PROFIT_CENTRE
    ORDER BY PROFIT_CENTRE
""").df()

#  Document Type Distribution (Audit / Compliance)
doc_type_dist = con.execute("""
    SELECT
        DOC_TYPE,
        CASE DOC_TYPE
            WHEN 'SA' THEN 'GL Transfer'
            WHEN 'DR' THEN 'Customer Invoice'
            WHEN 'KR' THEN 'Vendor Invoice'
            WHEN 'AB' THEN 'Asset Posting'
        END AS DOC_DESC,
        COUNT(DOC_NO)       AS COUNT,
        SUM(ABS(AMOUNT_LC)) AS TOTAL_AMOUNT
    FROM GL_FACT
    GROUP BY DOC_TYPE
    ORDER BY COUNT DESC
""").df()

print("[STEP 2] BW/4HANA Bridge OLAP queries executed (6 Financial InfoQueries).\n")


# STEP 3 — SAP Databricks Layer (Period-End Spark Aggregations)
#
# Simulates two Databricks notebooks:
#   nb_period_close_pnl   → Rolling P&L with MoM variance
#   nb_dso_scorecard      → DSO calculation per Profit Centre


# Month-over-Month P&L Variance
pnl_by_period = pnl_by_period.sort_values('MONTH_NUM').reset_index(drop=True)
pnl_by_period['PREV_REVENUE']  = pnl_by_period['TOTAL_REVENUE'].shift(1)
pnl_by_period['MOM_VARIANCE']  = pnl_by_period['TOTAL_REVENUE'] - pnl_by_period['PREV_REVENUE']
pnl_by_period['MOM_PCT']       = (
    pnl_by_period['MOM_VARIANCE'] / pnl_by_period['PREV_REVENUE'].replace(0, np.nan) * 100
).round(2)
pnl_by_period['ROLLING_3M_REV'] = (
    pnl_by_period['TOTAL_REVENUE'].rolling(3, min_periods=1).mean().round(0)
)

# DSO per Profit Centre
dso_df = df_ar.groupby('PROFIT_CENTRE').apply(
    lambda g: pd.Series({
        'TOTAL_AR'      : g['INVOICE_AMT'].sum(),
        'TOTAL_OVERDUE' : g.loc[g['STATUS']=='OVERDUE', 'INVOICE_AMT'].sum(),
        'AVG_OVERDUE_DAYS': g['OVERDUE_DAYS'].mean().round(1),
        'COLLECTION_RATE': (g.loc[g['STATUS']=='PAID', 'INVOICE_AMT'].sum()
                            / g['INVOICE_AMT'].sum() * 100).round(1),
    })
).reset_index()

print("[STEP 3] SAP Databricks period-end aggregations completed.\n")

# STEP 4 — SAP Analytics Cloud (SAC) Dashboard — 6 Story Panels


os.makedirs('Screenshots', exist_ok=True)

def watermark(ax):
    ax.text(0.99, 0.01, WATERMARK, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=SAP_GRAY, style='italic')

# Fig 1: Monthly P&L — Revenue vs Expense vs Net Profit 
fig, ax = plt.subplots(figsize=(13, 5.5))
x     = np.arange(len(pnl_by_period))
w     = 0.32
months_short = [p.replace('/2024','').replace('P0','P').replace('P','') for p in pnl_by_period['PERIOD']]
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

b1 = ax.bar(x - w, pnl_by_period['TOTAL_REVENUE']/1e6,  width=w, color=SAP_BLUE,  label='Revenue',  edgecolor='white')
b2 = ax.bar(x,     pnl_by_period['TOTAL_EXPENSE']/1e6,  width=w, color=SAP_GOLD,  label='Expense',  edgecolor='white')
b3 = ax.bar(x + w, pnl_by_period['NET_PROFIT']/1e6,     width=w,
            color=[SAP_GREEN if v >= 0 else SAP_RED for v in pnl_by_period['NET_PROFIT']],
            label='Net Profit', edgecolor='white')
ax2 = ax.twinx()
ax2.plot(x, pnl_by_period['MARGIN_PCT'], color=SAP_RED, linewidth=2,
         marker='o', markersize=5, label='Margin %', linestyle='--')
ax2.set_ylabel('Net Margin %', color=SAP_RED, fontsize=10)
ax2.tick_params(axis='y', colors=SAP_RED)
ax2.axhline(0, color=SAP_RED, linewidth=0.5, linestyle=':')

ax.set_xticks(x)
ax.set_xticklabels(month_labels, fontsize=9)
ax.set_title('Fig 1 — Monthly P&L: Revenue vs Expense vs Net Profit\n'
             '(SAP Analytics Cloud — Financial Close Dashboard)',
             fontsize=12, fontweight='bold', color=SAP_BLUE, pad=10)
ax.set_ylabel('Amount (INR Millions)', fontsize=10)
ax.set_facecolor(SAP_LGRAY)
fig.patch.set_facecolor('white')
lines = [mpatches.Patch(color=SAP_BLUE, label='Revenue'),
         mpatches.Patch(color=SAP_GOLD, label='Expense'),
         mpatches.Patch(color=SAP_GREEN, label='Net Profit')]
ax.legend(handles=lines, loc='upper left', fontsize=9)
ax.spines[['top']].set_visible(False)
watermark(ax)
plt.tight_layout()
plt.savefig('Screenshots/Fig1_Monthly_PnL.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Trial Balance — GL Account Balances 
fig, ax = plt.subplots(figsize=(12, 6))
tb = trial_balance.sort_values('BALANCE')
colors_tb = [SAP_GREEN if v >= 0 else SAP_RED for v in tb['BALANCE']]
bars = ax.barh(tb['GL_DESC'], tb['BALANCE']/1e6, color=colors_tb, edgecolor='white', height=0.6)
ax.axvline(0, color=SAP_GRAY, linewidth=1)
for bar, val in zip(bars, tb['BALANCE']):
    xpos = bar.get_width() + (0.2 if val >= 0 else -0.2)
    ha   = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f'₹{val/1e6:.1f}M', va='center', ha=ha, fontsize=9, fontweight='bold')
ax.set_title('Fig 2 — Trial Balance: GL Account Balances\n'
             '(SAP BW/4HANA Bridge — FAGLFLEXT InfoProvider)',
             fontsize=12, fontweight='bold', color=SAP_BLUE, pad=10)
ax.set_xlabel('Balance (INR Millions)', fontsize=10)
ax.set_facecolor(SAP_LGRAY)
fig.patch.set_facecolor('white')
ax.spines[['top', 'right']].set_visible(False)
watermark(ax)
plt.tight_layout()
plt.savefig('Screenshots/Fig2_Trial_Balance.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Cost Centre Expense Heatmap 
fig, ax = plt.subplots(figsize=(12, 5.5))
pivot_cc = df_gl[df_gl['IS_EXPENSE'] == 1].pivot_table(
    values='AMOUNT_LC', index='COST_CENTRE',
    columns=df_gl[df_gl['IS_EXPENSE']==1]['POST_DATE'].dt.strftime('%b'),
    aggfunc=lambda x: abs(x).sum(), fill_value=0
)
# reorder months
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot_cc = pivot_cc.reindex(columns=[m for m in month_order if m in pivot_cc.columns])
pivot_cc_m = pivot_cc / 1e6
sns.heatmap(pivot_cc_m, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white', ax=ax,
            annot_kws={'size': 9},
            cbar_kws={'label': 'Expense (INR Millions)'})
ax.set_title('Fig 3 — Cost Centre × Month Expense Heatmap\n'
             '(SAP Datasphere — Cost Controlling Data Product)',
             fontsize=12, fontweight='bold', color=SAP_BLUE, pad=10)
ax.set_xlabel('Month', fontsize=10)
ax.set_ylabel('Cost Centre', fontsize=10)
fig.text(0.99, 0.01, WATERMARK, ha='right', va='bottom', fontsize=8, color=SAP_GRAY, style='italic')
plt.tight_layout()
plt.savefig('Screenshots/Fig3_CostCentre_Heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

#  Fig 4: Profit Centre P&L Comparison 
fig, ax = plt.subplots(figsize=(11, 5.5))
x  = np.arange(len(pc_pnl))
w  = 0.28
pc_colors = [SAP_BLUE, SAP_GOLD, SAP_TEAL, SAP_GREEN]
b1 = ax.bar(x - w, pc_pnl['REVENUE']/1e6,  width=w, color=SAP_BLUE,  label='Revenue',  edgecolor='white')
b2 = ax.bar(x,     pc_pnl['EXPENSE']/1e6,  width=w, color=SAP_GOLD,  label='Expense',  edgecolor='white')
b3 = ax.bar(x + w, pc_pnl['NET']/1e6,      width=w,
            color=[SAP_GREEN if v >= 0 else SAP_RED for v in pc_pnl['NET']],
            label='Net P&L', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(pc_pnl['PROFIT_CENTRE'], fontsize=10)
ax.set_title('Fig 4 — Profit Centre P&L Comparison\n'
             '(SAP Analytics Cloud — EC-PCA Profit Centre Accounting)',
             fontsize=12, fontweight='bold', color=SAP_BLUE, pad=10)
ax.set_ylabel('Amount (INR Millions)', fontsize=10)
ax.set_facecolor(SAP_LGRAY)
fig.patch.set_facecolor('white')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(0, color=SAP_GRAY, linewidth=0.8, linestyle=':')
watermark(ax)
plt.tight_layout()
plt.savefig('Screenshots/Fig4_ProfitCentre_PnL.png', dpi=150, bbox_inches='tight')
plt.close()

#Fig 5: AR Ageing Buckets (DSO Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
age_cols   = ['CURRENT', 'DAYS_1_30', 'DAYS_31_60', 'OVER_60']
age_labels = ['Current', '1–30 Days', '31–60 Days', '>60 Days']
age_colors = [SAP_GREEN, SAP_GOLD, SAP_TEAL, SAP_RED]
x = np.arange(len(ar_ageing))
w = 0.20
for i, (col, label, color) in enumerate(zip(age_cols, age_labels, age_colors)):
    axes[0].bar(x + i*w - 1.5*w, ar_ageing[col]/1e6, width=w,
                color=color, label=label, edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(ar_ageing['PROFIT_CENTRE'], fontsize=9)
axes[0].set_title('AR Ageing Buckets by Profit Centre', fontweight='bold', color=SAP_BLUE)
axes[0].set_ylabel('Amount (INR Millions)', fontsize=9)
axes[0].set_facecolor(SAP_LGRAY)
axes[0].spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8)

axes[1].bar(dso_df['PROFIT_CENTRE'], dso_df['COLLECTION_RATE'],
            color=[SAP_BLUE, SAP_GOLD, SAP_GREEN, SAP_TEAL], edgecolor='white')
axes[1].set_title('Collection Rate % by Profit Centre', fontweight='bold', color=SAP_BLUE)
axes[1].set_ylabel('Collection Rate (%)', fontsize=9)
axes[1].set_facecolor(SAP_LGRAY)
axes[1].spines[['top','right']].set_visible(False)
axes[1].axhline(80, color=SAP_RED, linewidth=1.5, linestyle='--', label='80% Target')
axes[1].legend(fontsize=8)
for bar, val in zip(axes[1].patches, dso_df['COLLECTION_RATE']):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Fig 5 — AR Ageing & Collection Rate Analysis\n'
             '(SAP Databricks — DSO Scorecard Notebook)',
             fontsize=12, fontweight='bold', color=SAP_BLUE)
fig.patch.set_facecolor('white')
fig.text(0.99, 0.01, WATERMARK, ha='right', va='bottom', fontsize=8, color=SAP_GRAY, style='italic')
plt.tight_layout()
plt.savefig('Screenshots/Fig5_AR_Ageing_DSO.png', dpi=150, bbox_inches='tight')
plt.close()

#  Fig 6: Document Type Audit + Rolling Revenue 
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
wedge_colors = [SAP_BLUE, SAP_GOLD, SAP_GREEN, SAP_TEAL]
axes[0].pie(doc_type_dist['COUNT'], labels=doc_type_dist['DOC_DESC'],
            autopct='%1.1f%%', colors=wedge_colors, startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Document Type Distribution\n(Compliance & Audit View)', fontweight='bold', color=SAP_BLUE)

axes[1].fill_between(range(len(pnl_by_period)), pnl_by_period['TOTAL_REVENUE']/1e6,
                     alpha=0.25, color=SAP_BLUE)
axes[1].plot(range(len(pnl_by_period)), pnl_by_period['TOTAL_REVENUE']/1e6,
             color=SAP_BLUE, linewidth=2.5, marker='o', markersize=5, label='Monthly Revenue')
axes[1].plot(range(len(pnl_by_period)), pnl_by_period['ROLLING_3M_REV']/1e6,
             color=SAP_GOLD, linewidth=2, linestyle='--', label='3M Rolling Avg')
axes[1].set_xticks(range(len(pnl_by_period)))
axes[1].set_xticklabels(month_labels, fontsize=8, rotation=30)
axes[1].set_title('Revenue Trend + 3M Rolling Average\n(Databricks Window Function)', fontweight='bold', color=SAP_BLUE)
axes[1].set_ylabel('Revenue (INR Millions)', fontsize=9)
axes[1].set_facecolor(SAP_LGRAY)
axes[1].spines[['top','right']].set_visible(False)
axes[1].legend(fontsize=9)

fig.suptitle('Fig 6 — Audit Document Distribution & Rolling Revenue Trend\n'
             '(SAP Analytics Cloud — Financial Compliance Story)',
             fontsize=12, fontweight='bold', color=SAP_BLUE)
fig.patch.set_facecolor('white')
fig.text(0.99, 0.01, WATERMARK, ha='right', va='bottom', fontsize=8, color=SAP_GRAY, style='italic')
plt.tight_layout()
plt.savefig('Screenshots/Fig6_Audit_RollingRevenue.png', dpi=150, bbox_inches='tight')
plt.close()

print("[STEP 4] SAC dashboard charts saved → Screenshots/ folder.\n")


# STEP 5 — Export Data Products (SAP Datasphere Marketplace)


os.makedirs('Data_Products', exist_ok=True)
df_gl.to_csv('Data_Products/DP_GL_Journal_Entries.csv', index=False)
pnl_by_period.to_csv('Data_Products/DP_PnL_By_Period.csv', index=False)
trial_balance.to_csv('Data_Products/DP_Trial_Balance.csv', index=False)
cc_expenses.to_csv('Data_Products/DP_CostCentre_Expenses.csv', index=False)
pc_pnl.to_csv('Data_Products/DP_ProfitCentre_PnL.csv', index=False)
dso_df.to_csv('Data_Products/DP_DSO_Scorecard.csv', index=False)
df_ar.to_csv('Data_Products/DP_AR_Ageing.csv', index=False)

print("[STEP 5] Data Products exported → Data_Products/ folder.\n")


# STEP 6 — Executive KPI Console (SAC KPI Tile simulation)


total_rev  = df_gl.loc[df_gl['IS_REVENUE']==1, 'AMOUNT_LC'].sum()
total_exp  = df_gl.loc[df_gl['IS_EXPENSE']==1, 'AMOUNT_LC'].abs().sum()
net_profit = total_rev - total_exp
overdue_ar = df_ar.loc[df_ar['STATUS']=='OVERDUE', 'INVOICE_AMT'].sum()
coll_rate  = (df_ar.loc[df_ar['STATUS']=='PAID','INVOICE_AMT'].sum()
              / df_ar['INVOICE_AMT'].sum() * 100)
top_cc_exp = cc_expenses.iloc[0]['COST_CENTRE']

print("=" * 65)
print("  R2R EXECUTIVE KPI SUMMARY — Financial Close FY 2024")
print("=" * 65)
print(f"  Total Revenue (FY2024)  : ₹{total_rev:>14,.0f}")
print(f"  Total Expense (FY2024)  : ₹{total_exp:>14,.0f}")
print(f"  Net Profit              : ₹{net_profit:>14,.0f}")
print(f"  Overdue AR Balance      : ₹{overdue_ar:>14,.0f}")
print(f"  Collection Rate         : {coll_rate:>13.1f}%")
print(f"  Highest Cost Centre     : {top_cc_exp}")
print(f"  GL Postings Processed   : {len(df_gl):>14,}")
print(f"  AR Invoices Processed   : {len(df_ar):>14,}")
print("=" * 65)
print("\n[DONE] R2R Pipeline complete. All outputs saved.\n")
print("  Screenshots/   → 6 SAC Financial Dashboard panels (PNG)")
print("  Data_Products/ → 7 CSV Data Products (Datasphere exports)")
print("=" * 65)
