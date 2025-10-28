import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Singapore Job Market Dashboard", layout="wide", initial_sidebar_state="expanded")

# Constants
Industries = [
    'Accommodation & Food Services',
    'Administrative & Support Services',
    'Community, Social & Personal Services',
    'Construction',
    'Financial & Insurance Services',
    'Information & Communications',
    'Manufacturing',
    'Professional Services',
    'Public Administration & Education Services',
    'Real Estate Services',
    'Transportation & Storage',
    'Wholesale & Retail Trade',
]

Palette = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
    '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#2E91E5'
]

# Sidebar filters
st.sidebar.title("🎛️ Dashboard Filters")
st.sidebar.markdown("---")

# Initialize session state
if 'industries_selection' not in st.session_state:
    st.session_state.industries_selection = Industries
if 'year_range' not in st.session_state:
    st.session_state.year_range = (2015, 2024)

# Year range filter
min_year, max_year = 2015, 2024
st.sidebar.markdown("**Select Year Range**")

# Initialize state
if "year_range" not in st.session_state:
    st.session_state.year_range = (min_year, max_year)

# Reset button
if st.sidebar.button("Reset Years", key="reset_years", use_container_width=True):
    st.session_state.year_range = (min_year, max_year)
    st.session_state.year_slider = (min_year, max_year)
    st.rerun()

# Slider
year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=st.session_state.get("year_slider", st.session_state.year_range),
    step=1,
    label_visibility="collapsed",
    key="year_slider"
)

if year_range != st.session_state.year_range:
    st.session_state.year_range = year_range

year_cols = list(range(st.session_state.year_range[0], st.session_state.year_range[1] + 1))

st.sidebar.markdown("")  # Add spacing

# Industry filter
st.sidebar.markdown("**Select Industries**")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("✓ Select All", key="select_all", use_container_width=True):
        st.session_state.industries_selection = Industries
        st.rerun()
with col2:
    if st.button("✗ Clear All", key="clear_all", use_container_width=True):
        st.session_state.industries_selection = []
        st.rerun()

selected_industries = st.sidebar.multiselect(
    "Industries",
    options=Industries,
    default=st.session_state.industries_selection,
    label_visibility="collapsed"
)

# Update session state
st.session_state.industries_selection = selected_industries

st.sidebar.markdown("---")
st.sidebar.info("📊 This dashboard visualizes Singapore employment data across industries from 2015-2024")

# Main title
st.title("🇸🇬 Singapore Job Market Dashboard")
st.markdown("---")

# Helper function to filter data
def filter_data(df, industries, years):
    df_filtered = df[df['Data Series'].isin(industries)]
    df_filtered = df_filtered[['Data Series'] + [year for year in years if year in df_filtered.columns]]
    return df_filtered

# ==================== KEY FIGURES CARDS SECTION ====================
st.header("📈 Key Figures")

# Add CSS to prevent text cutoff in metric labels
st.markdown("""
<style>
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        line-height: 1.2 !important;
        min-height: 40px !important;
    }
</style>
""", unsafe_allow_html=True)

try:
    import re

    # Helper function to clean year columns
    def clean_year_columns(cols):
        cleaned = []
        for col in cols:
            try:
                # Try to extract just the numeric part (e.g., '2007a' -> 2007)
                year_str = str(col).strip()
                if year_str.replace('.', '').isdigit():
                    cleaned.append(int(float(year_str)))
                else:
                    # Extract first numeric part
                    match = re.match(r'(\d{4})', year_str)
                    if match:
                        cleaned.append(int(match.group(1)))
                    else:
                        cleaned.append(col)
            except:
                cleaned.append(col)
        return cleaned

    # Load data for KPIs
    # 1. Employment data
    df_employment = pd.read_csv('M182081.csv')
    df_employment = df_employment.dropna(thresh=2)
    year_cols_raw = clean_year_columns(df_employment.iloc[0,1:])
    df_employment.columns = ['Data Series'] + year_cols_raw
    df_employment = df_employment[['Data Series'] + list(range(min_year, max_year + 1))]
    df_employment['Data Series'] = df_employment['Data Series'].str.strip()
    df_employment['Data Series'] = df_employment['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
    df_employment = df_employment[df_employment['Data Series'].isin(Industries)].reset_index(drop=True)
    df_employment = df_employment.replace(['-', 'na', np.nan], 0)
    for year in range(min_year, max_year + 1):
        df_employment[year] = df_employment[year].astype(float)
    df_employment = df_employment.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

    # 2. Gender data
    df_male = pd.read_csv('M182141.csv')
    df_male = df_male.dropna(thresh=2)
    year_cols_raw = clean_year_columns(df_male.iloc[0,1:])
    df_male.columns = ['Data Series'] + year_cols_raw
    df_male = df_male[['Data Series'] + list(range(min_year, max_year + 1))]
    df_male['Data Series'] = df_male['Data Series'].str.strip()
    df_male['Data Series'] = df_male['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
    df_male['Data Series'] = df_male['Data Series'].replace('All Industries (Employed Male Residents)', 'Total')
    df_male = df_male[df_male['Data Series'].isin(['Total'] + Industries)].reset_index(drop=True)
    df_male = df_male.replace(['-', 'na', np.nan], 0)
    for year in range(min_year, max_year + 1):
        df_male[year] = df_male[year].astype(float)
    df_male = df_male.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

    df_female = pd.read_csv('M182151.csv')
    df_female = df_female.dropna(thresh=2)
    year_cols_raw = clean_year_columns(df_female.iloc[0,1:])
    df_female.columns = ['Data Series'] + year_cols_raw
    df_female = df_female[['Data Series'] + list(range(min_year, max_year + 1))]
    df_female['Data Series'] = df_female['Data Series'].str.strip()
    df_female['Data Series'] = df_female['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
    df_female['Data Series'] = df_female['Data Series'].replace('All Industries (Employed Female Residents)', 'Total')
    df_female = df_female[df_female['Data Series'].isin(['Total'] + Industries)].reset_index(drop=True)
    df_female = df_female.replace(['-', 'na', np.nan], 0)
    for year in range(min_year, max_year + 1):
        df_female[year] = df_female[year].astype(float)
    df_female = df_female.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

    # 3. Income data
    df_income = pd.read_csv('FT_Res_ind_income.csv')
    # Clean year column (remove letters like '2007a' -> 2007)
    df_income['year'] = df_income['year'].astype(str).str.extract(r'(\d{4})')[0].astype(int)
    df_income['industry'] = df_income['industry'].str.title()
    df_income['industry'] = df_income['industry'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')

    # 4. Job vacancy data
    df_vacancies = pd.read_csv('M184071.csv')
    df_vacancies = df_vacancies.dropna(thresh=2)
    df_vacancies.columns = ['Data Series'] + list(df_vacancies.iloc[0, 1:])
    df_vacancies = df_vacancies.drop(columns=[col for col in df_vacancies.columns[1:] if '4Q' not in str(col)])
    # Clean year columns from quarterly data
    year_cols_vac = []
    for col in df_vacancies.columns[1:]:
        match = re.match(r'(\d{4})', str(col))
        if match:
            year_cols_vac.append(int(match.group(1)))
        else:
            year_cols_vac.append(col)
    df_vacancies.columns = ['Data Series'] + year_cols_vac
    df_vacancies = df_vacancies[['Data Series'] + list(range(min_year, max_year + 1))]
    df_vacancies['Data Series'] = df_vacancies['Data Series'].str.strip()
    df_vacancies['Data Series'] = df_vacancies['Data Series'].str.replace('And', '&', regex=False)
    df_vacancies['Data Series'] = df_vacancies['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
    df_vacancies = df_vacancies[df_vacancies['Data Series'].isin(Industries)].reset_index(drop=True)
    df_vacancies = df_vacancies.replace(['-', 'na', np.nan], 0)
    for year in range(min_year, max_year + 1):
        df_vacancies[year] = df_vacancies[year].astype(float)
    df_vacancies = df_vacancies.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

    # 5. Working hours data
    df_hours = pd.read_csv('M184061.csv')
    df_hours = df_hours.dropna(thresh=2)
    year_cols_raw = clean_year_columns(df_hours.iloc[0,1:])
    df_hours.columns = ['Data Series'] + year_cols_raw
    df_hours = df_hours[['Data Series'] + list(range(min_year, max_year + 1))]
    df_hours['Data Series'] = df_hours['Data Series'].str.strip()
    df_hours['Data Series'] = df_hours['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
    df_hours = df_hours[df_hours['Data Series'].isin(Industries)].reset_index(drop=True)
    df_hours = df_hours.replace(['-', 'na', np.nan], 0)
    for year in range(min_year, max_year + 1):
        df_hours[year] = df_hours[year].astype(float)
    df_hours = df_hours.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

    # 6. Education data
    df_education = pd.read_csv('M182271.csv')
    df_education = df_education.dropna(thresh=2)
    year_cols_raw = clean_year_columns(df_education.iloc[0,1:])
    df_education.columns = ['Data Series'] + year_cols_raw
    df_education = df_education.iloc[2:7][['Data Series'] + list(range(min_year, max_year + 1))]
    df_education['Data Series'] = df_education['Data Series'].str.strip()
    df_education['Data Series'] = df_education['Data Series'].replace('Post-Secondary (Non-Tertiary)', 'Post-Secondary')
    df_education.reset_index(drop=True, inplace=True)

    # Calculate KPI values
    latest_year = year_cols[-1]
    prev_year = year_cols[-2] if len(year_cols) > 1 else latest_year

    # KPI 1: Total Employment
    emp_filtered = df_employment[df_employment['Data Series'].isin(selected_industries)]
    total_employment_current = emp_filtered[latest_year].sum()
    total_employment_prev = emp_filtered[prev_year].sum()
    employment_growth = ((total_employment_current - total_employment_prev) / total_employment_prev * 100) if total_employment_prev > 0 else 0

    # KPI 2: Median Income
    income_filtered = df_income[(df_income['year'] == latest_year) & (df_income['industry'].isin(selected_industries))]
    avg_median_income = income_filtered['median_gross_monthly_income_excluding_employer_cpf'].mean()
    income_prev = df_income[(df_income['year'] == prev_year) & (df_income['industry'].isin(selected_industries))]['median_gross_monthly_income_excluding_employer_cpf'].mean()
    income_change = ((avg_median_income - income_prev) / income_prev * 100) if income_prev > 0 else 0

    # KPI 3: Job Vacancies
    vac_filtered = df_vacancies[df_vacancies['Data Series'].isin(selected_industries)]
    total_vacancies = vac_filtered[latest_year].sum()
    total_vacancies_prev = vac_filtered[prev_year].sum()
    vacancies_change = ((total_vacancies - total_vacancies_prev) / total_vacancies_prev * 100) if total_vacancies_prev > 0 else 0

    # KPI 4: Gender Balance
    male_filtered = df_male[df_male['Data Series'].isin(selected_industries)]
    female_filtered = df_female[df_female['Data Series'].isin(selected_industries)]
    total_male = male_filtered[latest_year].sum()
    total_female = female_filtered[latest_year].sum()
    male_pct = (total_male / (total_male + total_female) * 100) if (total_male + total_female) > 0 else 0
    female_pct = 100 - male_pct

    # Calculate YoY changes for gender
    total_male_prev = male_filtered[prev_year].sum()
    total_female_prev = female_filtered[prev_year].sum()
    male_pct_prev = (total_male_prev / (total_male_prev + total_female_prev) * 100) if (total_male_prev + total_female_prev) > 0 else 0
    female_pct_prev = 100 - male_pct_prev
    male_pct_change = male_pct - male_pct_prev
    female_pct_change = female_pct - female_pct_prev

    # KPI 5: Average Working Hours
    hours_filtered = df_hours[df_hours['Data Series'].isin(selected_industries)]
    avg_hours = hours_filtered[latest_year].mean()
    avg_hours_prev = hours_filtered[prev_year].mean()
    hours_change = avg_hours - avg_hours_prev

    # KPI 6: Higher Education %
    higher_ed = df_education[df_education['Data Series'].isin(['Degree', 'Diploma & Professional Qualification'])]
    total_higher_ed = higher_ed[latest_year].sum()
    total_workforce = df_education[latest_year].sum()
    higher_ed_pct = (total_higher_ed / total_workforce * 100) if total_workforce > 0 else 0

    total_higher_ed_prev = higher_ed[prev_year].sum()
    total_workforce_prev = df_education[prev_year].sum()
    higher_ed_pct_prev = (total_higher_ed_prev / total_workforce_prev * 100) if total_workforce_prev > 0 else 0
    higher_ed_change = higher_ed_pct - higher_ed_pct_prev

    # Display KPI cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6, kpi_col7 = st.columns(7)

    with kpi_col1:
        st.metric(
            label="👥 Total Employment",
            value=f"{total_employment_current:.1f}K",
            delta=f"{employment_growth:+.1f}% YoY"
        )

    with kpi_col2:
        st.metric(
            label="💰 Median Income",
            value=f"${avg_median_income:,.0f}",
            delta=f"{income_change:+.1f}% YoY"
        )

    with kpi_col3:
        st.metric(
            label="📋 Job Vacancies",
            value=f"{total_vacancies:,.0f}",
            delta=f"{vacancies_change:+.1f}% YoY"
        )

    with kpi_col4:
        st.metric(
            label="👨 Male Employment",
            value=f"{male_pct:.1f}%",
            delta=f"{male_pct_change:+.1f}% YoY"
        )

    with kpi_col5:
        st.metric(
            label="👩 Female Employment",
            value=f"{female_pct:.1f}%",
            delta=f"{female_pct_change:+.1f}% YoY"
        )

    with kpi_col6:
        st.metric(
            label="⏰ Avg Work Hours",
            value=f"{avg_hours:.1f} hrs/wk",
            delta=f"{hours_change:+.1f} hrs YoY"
        )

    with kpi_col7:
        st.metric(
            label="🎓 Higher Education",
            value=f"{higher_ed_pct:.1f}%",
            delta=f"{higher_ed_change:+.1f}% YoY",
            help="Degree + Diploma holders"
        )

except Exception as e:
    st.error(f"Error loading KPI data: {e}")

st.markdown("---")

# ==================== TABBED SECTIONS ====================
tab1, tab2 = st.tabs(["🏭 Industry-Based Analysis", "📊 Demographic Analysis"])

# ==================== TAB 1: INDUSTRY-BASED ANALYSIS ====================
with tab1:
    # ==================== OVERALL SECTION ====================
    st.header("Employment Overview")
    col1 = st.columns(1)[0]

    with col1:
        try:
            df = pd.read_csv('M182081.csv')
            df = df.dropna(thresh=2)
            df.columns = ['Data Series'] + list(df.iloc[0,1:].astype(int))
            df = df[['Data Series'] + list(range(min_year, max_year + 1))]
            df['Data Series'] = df['Data Series'].str.strip()
            df['Data Series'] = df['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
            df = df[df['Data Series'].isin(Industries)].reset_index(drop=True)
            df = df.replace(['-', 'na', np.nan], 0)
            for year in range(min_year, max_year + 1):
                df[year] = df[year].astype(float)
            df = df.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

            # Filter data
            df_filtered = filter_data(df, selected_industries, year_cols)
            df_melted = df_filtered.melt(id_vars='Data Series', var_name='Year', value_name='Employed Residents')

            # Plot
            fig = px.line(df_melted, x='Year', y='Employed Residents', color='Data Series',
                          title='Employed Residents By Industry Over Years',
                          labels={'Employed Residents': 'Employed Residents (thousands)', 'Data Series': 'Industry'},
                          color_discrete_sequence=Palette)
            fig.update_traces(mode='lines+markers')  # Add markers for single-year visibility
            fig.update_layout(height=500)
            fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data: {e}")

    st.markdown("---")

    # ==================== INCOME SECTION ====================
    st.header("Income & Bonus Analysis")
    col1, col2 = st.columns(2)

    with col1:
        try:
            df_income = pd.read_csv('FT_Res_ind_income.csv')
            df_income = df_income[df_income['year'].isin(list(map(str, year_cols)))]
            df_income['year'] = df_income['year'].astype(int)
            df_income['industry'] = df_income['industry'].str.title()
            df_income['industry'] = df_income['industry'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
            df_income = df_income[df_income['industry'].isin(selected_industries)]

            fig = px.line(df_income, x='year', y='median_gross_monthly_income_excluding_employer_cpf',
                          color='industry',
                          title='Median Gross Monthly Income by Industry',
                          labels={'year': 'Year', 'industry': 'Industry',
                                  'median_gross_monthly_income_excluding_employer_cpf': 'Median Income (SGD)'},
                          color_discrete_sequence=Palette,
                          category_orders={'industry': Industries})
            fig.update_traces(mode='lines+markers')  # Add markers for single-year visibility
            fig.update_layout(height=500)
            fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data (Chart 1): {e}")

    with col2:
        try:
            df_bonus = pd.read_csv('mrsd_18_Variable_component.csv')
            df_bonus = df_bonus[df_bonus['year'].isin(year_cols)]
            df_bonus['year'] = df_bonus['year'].astype(int)

            mapping = {
                'Wholesale Trade': 'Wholesale & Retail Trade',
                'Retail Trade': 'Wholesale & Retail Trade',
                'Transport And Storage': 'Transportation & Storage',
                'Accommodation': 'Accommodation & Food Services',
                'Food  Beverage Services': 'Accommodation & Food Services',
                'Information And Communications': 'Information & Communications',
                'Financial And Insurance Services': 'Financial & Insurance Services',
                'Real Estate Services': 'Real Estate Services',
                'Professional Services': 'Professional Services',
                'Administrative And Support Services': 'Administrative & Support Services',
                'Community Social And Personal Services': 'Community, Social & Personal Services'
            }
            df_bonus['ind2'] = df_bonus['ind2'].replace(mapping)
            df_bonus = df_bonus.groupby(['year','ind2'])['avc'].sum().reset_index()
            df_bonus = df_bonus[df_bonus['ind2'].isin(selected_industries)]

            fig = px.line(df_bonus, x='year', y='avc', color='ind2',
                          title='Bonus Quantum Paid by Industry',
                          labels={'year': 'Year', 'avc': 'Annual Variable Component (AVC)', 'ind2': 'Industry'},
                          color_discrete_sequence=Palette,
                          category_orders={'ind2': Industries})
            fig.update_traces(mode='lines+markers')  # Add markers for single-year visibility
            fig.update_layout(height=500)
            fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data (Chart 2): {e}")

    st.markdown("---")

    # ==================== VACANCY SECTION ====================
    st.header("Job Vacancy Analysis")
    col1, col2 = st.columns(2)

    with col1:
        try:
            df = pd.read_csv('M184071.csv')
            df = df.dropna(thresh=2)
            df.columns = ['Data Series'] + list(df.iloc[0, 1:])
            df = df.drop(columns=[col for col in df.columns[1:] if '4Q' not in str(col)])
            df.columns = ['Data Series'] + [int(col.split()[0]) for col in df.columns[1:]]
            df = df[['Data Series'] + list(range(min_year, max_year + 1))]
            df['Data Series'] = df['Data Series'].str.strip()
            df['Data Series'] = df['Data Series'].str.replace('And', '&', regex=False)
            df['Data Series'] = df['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
            df = df[df['Data Series'].isin(Industries)].reset_index(drop=True)
            df = df.replace(['-', 'na', np.nan], 0)
            for year in range(min_year, max_year + 1):
                df[year] = df[year].astype(float)
            df = df.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

            df_filtered = filter_data(df, selected_industries, year_cols)
            df_melted = df_filtered.melt(id_vars='Data Series', var_name='Year', value_name='Job Vacancies')

            fig = px.line(df_melted, x='Year', y='Job Vacancies', color='Data Series',
                          title='Job Vacancies By Industry Over Years',
                          color_discrete_sequence=Palette)
            fig.update_traces(mode='lines+markers')  # Add markers for single-year visibility
            fig.update_layout(height=500)
            fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data (Chart 1): {e}")

    with col2:
        try:
            df_vacancies_melted = df.melt(id_vars='Data Series', var_name='year', value_name='Job Vacancy')
            df_income_copy = df_income.rename(columns={'industry': 'Data Series'})
            combined_df = pd.merge(df_vacancies_melted, df_income_copy, on=['Data Series', 'year'])

            df_lastyear = combined_df[combined_df['year'] == combined_df['year'].max()]
            df_lastyear = df_lastyear[df_lastyear['Data Series'].isin(selected_industries)]

            fig = px.scatter(df_lastyear, x='Job Vacancy', y='median_gross_monthly_income_excluding_employer_cpf',
                             color='Data Series', size='Job Vacancy', hover_name='Data Series',
                             title=f'Job Vacancy vs. Median Income by Industry ({df_lastyear["year"].max()})',
                             labels={'Data Series': 'Industry',
                                     'median_gross_monthly_income_excluding_employer_cpf': 'Median Income (SGD)'},
                             color_discrete_sequence=Palette)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data (Chart 2): {e}")

    st.markdown("---")

    # ==================== WORKING HOURS SECTION ====================
    st.header("Working Hours Analysis")
    col1, col2 = st.columns(2)

    with col1:
        try:
            # Load df1
            df1 = pd.read_csv('M184061.csv')
            df1 = df1.dropna(thresh=2)
            df1.columns = ['Data Series'] + list(df1.iloc[0,1:].astype(int))
            df1 = df1[['Data Series'] + list(range(min_year, max_year + 1))]
            df1['Data Series'] = df1['Data Series'].str.strip()
            df1['Data Series'] = df1['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
            df1 = df1[df1['Data Series'].isin(Industries)].reset_index(drop=True)
            df1 = df1.replace(['-', 'na', np.nan], 0)
            for year in range(min_year, max_year + 1):
                df1[year] = df1[year].astype(float)
            df1 = df1.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

            # Load df2
            df2 = pd.read_csv('M184091.csv')
            df2 = df2.dropna(thresh=2)
            df2.columns = ['Data Series'] + list(df2.iloc[0,1:].astype(int))
            df2 = df2.iloc[1:][['Data Series'] + list(range(min_year, max_year + 1))]
            df2['Data Series'] = df2['Data Series'].str.strip()
            df2['Data Series'] = df2['Data Series'].replace('Post-Secondary (Non-Tertiary)', 'Post-Secondary')
            df2 = df2.replace(['-', 'na', np.nan], 0)
            for year in range(min_year, max_year + 1):
                df2[year] = df2[year].astype(float)
            df2.reset_index(drop=True, inplace=True)

            # Filter and combine
            df1_filtered = filter_data(df1, selected_industries, year_cols)
            df1_melted = df1_filtered.melt(id_vars='Data Series', var_name='Year', value_name='Avg paid working hrs per week')
            df2_melted = df2.melt(id_vars='Data Series', var_name='Year', value_name='Value')
            df2_melted = df2_melted[df2_melted['Year'].isin(year_cols)]
            df2_wide = df2_melted.pivot(index='Year', columns='Data Series', values='Value').reset_index()

            df_combined = df1_melted.merge(df2_wide, on='Year', how='left')
            df_combined['Hours_Gap'] = df_combined['Mean Usual Hours Worked Per Week (Hours)'] - df_combined['Avg paid working hrs per week']

            fig = px.line(df_combined, x='Year', y='Hours_Gap', color='Data Series',
                          title='Hours Gap Over the Years by Industry',
                          labels={'Hours_Gap': 'Hours Gap', 'Data Series': 'Industry'},
                          color_discrete_sequence=Palette)
            fig.update_traces(mode='lines+markers')  # Add markers for single-year visibility
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5)
            fig.update_layout(height=500)
            fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data (Chart 1): {e}")

    with col2:
        try:
            heatmap_data = df_combined.pivot_table(
                index='Data Series',
                columns='Year',
                values='Avg paid working hrs per week',
                aggfunc='mean'
            )

            fig = px.imshow(heatmap_data,
                            labels=dict(x="Year", y="Industry", color="Avg Hours"),
                            title='Average Paid Working Hours per Week by Industry',
                            text_auto=True,
                            color_continuous_scale=px.colors.sequential.Viridis)
            fig.update_xaxes(side="bottom", dtick=1)  # Fix year axis to show only integers
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data (Chart 2): {e}")

# ==================== TAB 2: DEMOGRAPHIC ANALYSIS ====================
with tab2:
    # ==================== GENDER SECTION ====================
    st.header("Gender Employment Analysis")

    try:
        # Male data
        df_male = pd.read_csv('M182141.csv')
        df_male = df_male.dropna(thresh=2)
        df_male.columns = ['Data Series'] + list(df_male.iloc[0,1:].astype(int))
        df_male = df_male[['Data Series'] + list(range(min_year, max_year + 1))]
        df_male['Data Series'] = df_male['Data Series'].str.strip()
        df_male['Data Series'] = df_male['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
        df_male['Data Series'] = df_male['Data Series'].replace('All Industries (Employed Male Residents)', 'Total')
        df_male = df_male[df_male['Data Series'].isin(['Total'] + Industries)].reset_index(drop=True)
        df_male = df_male.replace(['-', 'na', np.nan], 0)
        for year in range(min_year, max_year + 1):
            df_male[year] = df_male[year].astype(float)
        df_male = df_male.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

        # Female data
        df_female = pd.read_csv('M182151.csv')
        df_female = df_female.dropna(thresh=2)
        df_female.columns = ['Data Series'] + list(df_female.iloc[0,1:].astype(int))
        df_female = df_female[['Data Series'] + list(range(min_year, max_year + 1))]
        df_female['Data Series'] = df_female['Data Series'].str.strip()
        df_female['Data Series'] = df_female['Data Series'].replace('Other Community, Social & Personal Services', 'Community, Social & Personal Services')
        df_female['Data Series'] = df_female['Data Series'].replace('All Industries (Employed Female Residents)', 'Total')
        df_female = df_female[df_female['Data Series'].isin(['Total'] + Industries)].reset_index(drop=True)
        df_female = df_female.replace(['-', 'na', np.nan], 0)
        for year in range(min_year, max_year + 1):
            df_female[year] = df_female[year].astype(float)
        df_female = df_female.groupby('Data Series')[list(range(min_year, max_year + 1))].sum().reset_index()

        # Combine and filter
        male_melted = df_male.melt(id_vars='Data Series', var_name='Year', value_name='Male Employment')
        female_melted = df_female.melt(id_vars='Data Series', var_name='Year', value_name='Female Employment')
        combined = pd.merge(male_melted, female_melted, on=['Data Series', 'Year'])
        combined = combined[combined['Year'].isin(year_cols)]

        df_to_plot = combined.groupby('Year')[['Male Employment', 'Female Employment']].sum().reset_index()

        # Chart 1: Male and Female Employment (horizontal layout - stacked vertically)
        fig = px.bar(df_to_plot, x='Year', y=['Male Employment', 'Female Employment'],
                     title='Total Male and Female Employed Residents',
                     labels={'value': 'Employed Residents (thousands)', 'variable': 'Gender'},
                     color_discrete_sequence=['#ADD8E6', '#FFB6C1'])
        fig.update_layout(barmode='group', height=500)
        fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
        st.plotly_chart(fig, use_container_width=True)

        # Chart 2: Proportion (horizontal layout - stacked below chart 1)
        df_prop = df_to_plot.copy()
        total = df_prop['Male Employment'] + df_prop['Female Employment']
        df_prop['Male Employment'] = df_prop['Male Employment'] / total
        df_prop['Female Employment'] = df_prop['Female Employment'] / total
        df_prop = df_prop.melt(id_vars='Year', var_name='Gender', value_name='Proportion')

        fig = px.area(df_prop, x='Year', y='Proportion', color='Gender',
                      color_discrete_sequence=['#ADD8E6', '#FFB6C1'],
                      title='Male and Female Employment Proportion',
                      labels={'Proportion': 'Proportion of Employed Residents'})
        fig.update_layout(yaxis=dict(range=[0, 1], dtick=0.1), height=500)
        fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")

    st.markdown("---")

    # ==================== EDUCATION SECTION ====================
    st.header("Education Level Analysis")

    try:
        df = pd.read_csv('M182271.csv')
        df = df.dropna(thresh=2)
        df.columns = ['Data Series'] + list(df.iloc[0,1:].astype(int))
        df = df.iloc[2:7][['Data Series'] + list(range(min_year, max_year + 1))]
        df['Data Series'] = df['Data Series'].str.strip()
        df['Data Series'] = df['Data Series'].replace('Post-Secondary (Non-Tertiary)', 'Post-Secondary')
        df.reset_index(drop=True, inplace=True)

        df_long = df.melt(id_vars='Data Series', var_name='Year', value_name='Value')
        df_long = df_long[df_long['Year'].isin(year_cols)]

        edu_order = ['Degree', 'Diploma & Professional Qualification', 'Post-Secondary', 'Secondary', 'Below Secondary']

        # Chart 1: Education Level Bar Chart (horizontal layout - stacked vertically)
        fig = px.bar(df_long, x='Year', y='Value', color='Data Series',
                     color_discrete_sequence=px.colors.sequential.Viridis,
                     title='Labour Force by Education Level',
                     labels={'Value': 'Number of Residents (thousands)', 'Data Series': 'Education Level'},
                     category_orders={'Data Series': edu_order})
        fig.update_layout(barmode='stack', height=500)
        fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
        st.plotly_chart(fig, use_container_width=True)

        # Chart 2: Education Level Share (horizontal layout - stacked below chart 1)
        df_pivot = df_long.pivot(index='Year', columns='Data Series', values='Value').fillna(0)
        df_share = df_pivot.div(df_pivot.sum(axis=1), axis=0).reset_index().melt(id_vars='Year', var_name='Data Series', value_name='Proportion')

        fig = px.area(df_share, x='Year', y='Proportion', color='Data Series',
                      color_discrete_sequence=px.colors.sequential.Viridis,
                      title='Education Level Share of Labour Force',
                      labels={'Proportion': 'Proportion of Employed Residents', 'Data Series': 'Education Level'},
                      category_orders={'Data Series': edu_order})
        fig.update_layout(yaxis=dict(range=[0, 1]), height=500)
        fig.update_xaxes(dtick=1)  # Fix year axis to show only integers
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")