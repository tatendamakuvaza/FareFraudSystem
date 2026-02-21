"""
STREAMLIT DASHBOARD
Generate OR Upload - Your Choice
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from core_engine import get_system, FraudDetectionSystem
import pandas as pd
from datetime import datetime
import time

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="🚌 Fraud Detection | Mukumba Brothers",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .upload-btn {
        background-color: #28a745 !important;
        color: white !important;
    }
    .generate-btn {
        background-color: #007bff !important;
        color: white !important;
    }
    .css-1d391kg { padding-top: 0rem; }
</style>
""", unsafe_allow_html=True)

# Initialize
if 'system' not in st.session_state:
    st.session_state.system = get_system()
    st.session_state.data_source = None  # 'generated' or 'uploaded'
    st.session_state.analyzed = False

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/bus.png", width=80)
st.sidebar.title("🎛️ Control Center")

# ==========================================
# STEP 1: DATA INPUT - TWO OPTIONS
# ==========================================
st.sidebar.header("📊 Step 1: Load Data")

# Create two columns in sidebar for buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    st.sidebar.markdown("**Option A: Generate**")
    n_trips = st.sidebar.number_input("Trips to generate", 100, 5000, 1000, 100)
    fraud_rate = st.sidebar.slider("Fraud % in generated data", 1, 20, 5, 1)
    
    if st.sidebar.button("🔄 GENERATE", key="gen_btn", help="Create fake data for testing"):
        with st.spinner("Generating synthetic data..."):
            st.session_state.system.generate_data(n_trips, fraud_rate/100)
            st.session_state.data_source = 'generated'
            st.session_state.analyzed = False
            st.rerun()

with col2:
    st.sidebar.markdown("**Option B: Upload Real Data**")
    st.sidebar.markdown("*Excel or CSV*")
    
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload file",
        type=['xlsx', 'csv', 'xls'],
        key="file_uploader",
        help="Upload your real trip data"
    )

# Handle file upload immediately
if uploaded_file is not None and st.session_state.data_source != 'uploaded':
    with st.spinner("Loading your data..."):
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store raw data for column mapping
            st.session_state.uploaded_df = df
            st.session_state.data_source = 'uploaded'
            st.session_state.analyzed = False
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Show column mapper if data uploaded
if st.session_state.data_source == 'uploaded' and 'uploaded_df' in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Map Your Columns")
    
    df = st.session_state.uploaded_df
    cols = df.columns.tolist()
    cols_with_none = ['None'] + cols
    
    # Required mappings
    st.sidebar.caption("Required columns:")
    trip_id_col = st.sidebar.selectbox("Trip ID", cols, index=cols.index('trip_id') if 'trip_id' in cols else 0)
    conductor_col = st.sidebar.selectbox("Conductor Name", cols, index=cols.index('conductor_name') if 'conductor_name' in cols else 0)
    tickets_col = st.sidebar.selectbox("Tickets Issued", cols, index=cols.index('tickets_issued') if 'tickets_issued' in cols else 0)
    cash_col = st.sidebar.selectbox("Cash Collected", cols, index=cols.index('cash_collected') if 'cash_collected' in cols else 0)
    
    # Expense columns
    st.sidebar.caption("Expense columns:")
    fuel_col = st.sidebar.selectbox("Fuel Expense", cols, index=cols.index('expenses_fuel') if 'expenses_fuel' in cols else 0)
    tolls_col = st.sidebar.selectbox("Tolls", cols, index=cols.index('expenses_tolls') if 'expenses_tolls' in cols else 0)
    other_col = st.sidebar.selectbox("Other Expenses", cols, index=cols.index('expenses_other') if 'expenses_other' in cols else 0)
    
    # Optional
    with st.sidebar.expander("Optional columns"):
        route_col = st.sidebar.selectbox("Route Name", cols_with_none, index=cols_with_none.index('route_name') if 'route_name' in cols else 0)
        date_col = st.sidebar.selectbox("Trip Date", cols_with_none, index=cols_with_none.index('trip_date') if 'trip_date' in cols else 0)
        electronic_col = st.sidebar.selectbox("Electronic Revenue", cols_with_none, index=0)
        distance_col = st.sidebar.selectbox("Distance (km)", cols_with_none, index=0)
        duration_col = st.sidebar.selectbox("Duration (min)", cols_with_none, index=0)
    
    # Confirm button
    if st.sidebar.button("✅ CONFIRM MAPPING & LOAD", type="primary"):
        with st.spinner("Processing your data..."):
            # Rename columns to standard format
            df_renamed = df.copy()
            df_renamed = df_renamed.rename(columns={
                trip_id_col: 'trip_id',
                conductor_col: 'conductor_name',
                tickets_col: 'tickets_issued',
                cash_col: 'cash_collected',
                fuel_col: 'expenses_fuel',
                tolls_col: 'expenses_tolls',
                other_col: 'expenses_other'
            })
            
            # Add optional columns
            df_renamed['electronic_revenue'] = df[electronic_col] if electronic_col != 'None' else df_renamed['tickets_issued'] * 5 * 0.3
            df_renamed['route_name'] = df[route_col] if route_col != 'None' else 'Unknown'
            df_renamed['trip_date'] = pd.to_datetime(df[date_col]) if date_col != 'None' else datetime.now()
            df_renamed['distance_km'] = df[distance_col] if distance_col != 'None' else 25
            df_renamed['trip_duration_min'] = df[duration_col] if duration_col != 'None' else 45
            df_renamed['expenses_other'] = df_renamed.get('expenses_other', 0)
            df_renamed['is_fraud'] = 0
            
            # Fill missing
            for col in ['route_name', 'trip_date', 'distance_km', 'trip_duration_min', 'expenses_other']:
                if col not in df_renamed.columns:
                    df_renamed[col] = 0 if 'expenses' in col else ('Unknown' if col == 'route_name' else datetime.now())
            
            st.session_state.system.current_data = df_renamed
            st.session_state.data_loaded = True
            st.rerun()

# ==========================================
# STEP 2: ANALYSIS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("🔍 Step 2: Analysis")

analyze_disabled = st.session_state.data_source is None

if st.sidebar.button("🚨 RUN FRAUD DETECTION", disabled=analyze_disabled, type="primary"):
    with st.spinner("AI analyzing for fraud patterns..."):
        if st.session_state.data_source == 'generated':
            st.session_state.system.analyze()
        else:
            st.session_state.system.analyze(st.session_state.system.current_data)
        st.session_state.analyzed = True
        st.rerun()

# ==========================================
# STEP 3: EXPORT
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("💾 Step 3: Export")

if st.sidebar.button("📥 Export to Excel", disabled=not st.session_state.analyzed):
    filename = st.session_state.system.export_results()
    with open(filename, 'rb') as f:
        st.sidebar.download_button("⬇️ Download", f, filename.split('/')[-1], 
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Template download
with st.sidebar.expander("Need template?"):
    template_data = {
        'trip_id': ['TRP-001', 'TRP-002'],
        'conductor_name': ['John B', 'Sarah K'],
        'tickets_issued': [45, 50],
        'cash_collected': [315.00, 350.00],
        'expenses_fuel': [85.00, 150.00],
        'expenses_tolls': [20.00, 40.00],
        'expenses_other': [10.00, 25.00],
        'route_name': ['Route A', 'Route B'],
        'trip_date': ['2026-01-15', '2026-01-15']
    }
    template_df = pd.DataFrame(template_data)
    template_df.to_excel('template.xlsx', index=False)
    with open('template.xlsx', 'rb') as f:
        st.sidebar.download_button("📥 Download Template", f, "data_template.xlsx", 
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==========================================
# MAIN CONTENT
# ==========================================
st.markdown('<p class="main-header">🚌 Fare Revenue Protection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Fraud Detection | COSO Internal Control Framework</p>', unsafe_allow_html=True)

# Status banner
if st.session_state.data_source is None:
    st.warning("👈 **Get Started:** Choose GENERATE (fake data) or UPLOAD (your real data) in the sidebar")
elif st.session_state.data_source == 'generated':
    st.info("📊 Using **GENERATED** synthetic data for testing")
elif st.session_state.data_source == 'uploaded':
    st.success(f"📁 Using **UPLOADED** real data: {len(st.session_state.system.current_data)} trips loaded")
    
if not st.session_state.analyzed and st.session_state.data_source:
    st.info("👈 **Next:** Click 'RUN FRAUD DETECTION' to analyze")

# Show data preview if uploaded
if st.session_state.data_source == 'uploaded' and not st.session_state.analyzed and 'uploaded_df' in st.session_state:
    st.subheader("📋 Your Data Preview")
    st.dataframe(st.session_state.uploaded_df.head(10), use_container_width=True)

# TABS (only show if analyzed)
if st.session_state.analyzed:
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔎 Investigations", "🚨 Live Alerts", "📊 Analytics"])
    
    summary = st.session_state.system.get_summary()
    
    with tab1:
        # Metrics
        st.subheader("Real-Time Risk Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trips", summary['total'])
        c2.metric("🔴 Critical", summary['critical'], delta_color="inverse")
        c3.metric("🟠 High Risk", summary['high'], delta_color="inverse")
        c4.metric("🟡 Medium", summary['medium'])
        c5.metric("🟢 Low Risk", summary['low'])
        
        # Pie chart
        risk_df = pd.DataFrame({
            'Category': ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
            'Count': [summary['critical'], summary['high'], summary['medium'], summary['low']]
        })
        fig = px.pie(risk_df, values='Count', names='Category',
                    color='Category',
                    color_discrete_map={
                        'CRITICAL': '#ff0000', 'HIGH': '#ff6600',
                        'MEDIUM': '#ffcc00', 'LOW': '#00cc00'
                    }, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top conductors
        st.subheader("⚠️ Highest Risk Conductors")
        if summary['top_conductors']:
            top_df = pd.DataFrame({
                'Conductor': list(summary['top_conductors'].keys()),
                'Avg Risk Score': [f"{v:.3f}" for v in summary['top_conductors'].values()],
                'Status': ['🔴 Review' if v > 0.6 else '🟠 Monitor' if v > 0.3 else '🟢 OK' 
                          for v in summary['top_conductors'].values()]
            })
            st.dataframe(top_df, hide_index=True)
    
    with tab2:
        st.subheader("Detailed Trip Analysis")
        df = st.session_state.system.current_results
        
        # Filters
        c1, c2, c3 = st.columns(3)
        with c1:
            risk_filter = st.multiselect("Risk Level", ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                                        default=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
        with c2:
            conductors = df['conductor_name'].unique()
            cond_filter = st.multiselect("Conductor", list(conductors), default=list(conductors))
        with c3:
            min_score = st.slider("Min Risk Score", 0.0, 1.0, 0.0, 0.05)
        
        filtered = df[(df['risk_category'].isin(risk_filter)) & 
                     (df['conductor_name'].isin(cond_filter)) &
                     (df['risk_score'] >= min_score)]
        
        # Scatter
        fig = px.scatter(filtered, x='revenue_gap_pct', y='expense_ratio',
                        color='risk_category', size='risk_score',
                        hover_data=['trip_id', 'conductor_name', 'route_name'],
                        color_discrete_map={
                            'CRITICAL': '#ff0000', 'HIGH': '#ff6600',
                            'MEDIUM': '#ffcc00', 'LOW': '#00cc00'
                        },
                        title="Revenue Gap vs Expense Ratio")
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(filtered.sort_values('risk_score', ascending=False), 
                    height=400, use_container_width=True)
    
    with tab3:
        st.subheader("🚨 Active Fraud Alerts")
        
        if st.session_state.system.alerts:
            critical = len([a for a in st.session_state.system.alerts if a['category'] == 'CRITICAL'])
            high = len([a for a in st.session_state.system.alerts if a['category'] == 'HIGH'])
            
            c1, c2, c3 = st.columns(3)
            c1.error(f"🔴 CRITICAL: {critical}")
            c2.warning(f"🟠 HIGH: {high}")
            c3.info(f"📋 Total: {len(st.session_state.system.alerts)}")
            
            for alert in sorted(st.session_state.system.alerts, key=lambda x: x['risk_score'], reverse=True)[:15]:
                if alert['category'] == 'CRITICAL':
                    with st.container():
                        st.error(f"""
                        **🔴 CRITICAL ALERT #{alert['alert_id']}**
                        
                        **Risk Score:** {alert['risk_score']:.3f} | **Time:** {alert['timestamp'][:19]}
                        
                        **Trip:** {alert['trip_id']} | **Conductor:** {alert['conductor']} | **Route:** {alert['route']}
                        
                        **Risk Factors:** {', '.join(alert['factors'])}
                        
                        **⚡ REQUIRED ACTION:** {alert['action']}
                        """)
                        c1, c2, c3 = st.columns(3)
                        c1.button("✅ Acknowledge", key=f"ack_{alert['alert_id']}")
                        c2.button("🔍 Investigate", key=f"inv_{alert['alert_id']}")
                        c3.button("📋 Escalate", key=f"esc_{alert['alert_id']}")
                        
                elif alert['category'] == 'HIGH':
                    st.warning(f"""
                    **🟠 HIGH RISK ALERT**
                    
                    **Trip:** {alert['trip_id']} | **Conductor:** {alert['conductor']}
                    
                    **Score:** {alert['risk_score']:.3f} | **Factors:** {', '.join(alert['factors'])}
                    """)
                else:
                    with st.expander(f"🟡 MEDIUM: {alert['trip_id']} - {alert['conductor']}"):
                        st.write(f"Score: {alert['risk_score']:.3f}")
                        st.write(f"Factors: {', '.join(alert['factors'])}")
        else:
            st.success("✅ No active alerts - system is secure!")
            st.balloons()
    
    with tab4:
        st.subheader("📊 Advanced Analytics")
        df = st.session_state.system.current_results
        
        # Time analysis
        df['hour'] = pd.to_datetime(df['trip_date']).dt.hour
        hourly = df.groupby('hour').agg({
            'actual_fraud': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        
        fig = px.line(hourly, x='hour', y=['actual_fraud', 'risk_score'],
                     labels={'value': 'Rate', 'hour': 'Hour of Day'},
                     title="Fraud Pattern by Hour")
        st.plotly_chart(fig, use_container_width=True)
        
        # Route analysis
        route_stats = df.groupby('route_name').agg({
            'risk_score': 'mean',
            'trip_id': 'count'
        }).reset_index()
        route_stats.columns = ['Route', 'Avg Risk', 'Trip Count']
        
        fig = px.bar(route_stats, x='Route', y='Avg Risk',
                    color='Avg Risk', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

else:
    # Empty state
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://img.icons8.com/clouds/200/upload.png")
        st.info("👆 Load data using the sidebar to begin analysis")

st.markdown("---")
st.caption("🔒 COSO Framework | ML-Powered | © Tatenda Makuvaza")