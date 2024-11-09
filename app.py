import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Set page config
st.set_page_config(
    page_title="Material Acquisitions - Transactions Generator",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    /* Title Styles */
    .app-title-container {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 2.5rem 2rem;
        border-radius: 1rem;
        margin: -1.5rem -1.5rem 2rem -1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .app-title {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .app-subtitle {
        color: #e2e8f0;
        font-size: 1.1rem;
        text-align: center;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        opacity: 0.9;
    }
    
    .version-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        color: #e2e8f0;
        font-size: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Option 1: ProcureSync
def display_procuresync_title():
    st.markdown("""
        <div class="app-title-container">
            <div class="version-badge">BETA</div>
            <h1 class="app-title">ProcureMetrics</h1>
            <p class="app-subtitle">
                Intelligent Procurement Analytics & Simulation Platform
            </p>
        </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        master_df = pd.read_csv('Master.csv')
        suppliers_df = pd.read_csv('Suppliers.csv')
        
        # Validate required columns
        required_master_columns = ['Item #', 'Name', 'CATEGORY', 'Commodity']
        required_supplier_columns = ['Item #', 'Supplier name', 'Supplier price']
        
        if not all(col in master_df.columns for col in required_master_columns):
            st.error(f"Master.csv is missing required columns: {required_master_columns}")
            return None, None
            
        if not all(col in suppliers_df.columns for col in required_supplier_columns):
            st.error(f"Suppliers.csv is missing required columns: {required_supplier_columns}")
            return None, None
            
        # Remove any rows with missing values in required columns
        master_df = master_df.dropna(subset=required_master_columns)
        suppliers_df = suppliers_df.dropna(subset=required_supplier_columns)
        
        # Validate that we have matching items
        matching_items = set(master_df['Item #']) & set(suppliers_df['Item #'])
        if not matching_items:
            st.error("No matching items found between Master.csv and Suppliers.csv")
            return None, None
            
        st.success(f"Found {len(matching_items)} valid items for transaction generation")
        
        return master_df, suppliers_df
    except FileNotFoundError:
        st.error("Please make sure Master.csv and Suppliers.csv are in the same directory as the script.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def generate_quantity(category, commodity):
    if commodity in ['Hardware', 'Screws']:
        return np.random.randint(100, 1000)
    elif commodity in ['Structural Products', 'Drywall']:
        return np.random.randint(50, 600)
    else:
        return np.random.randint(20, 500)
    
##########################################################################################
def analyze_prices(master_df, suppliers_df):
    """
    Analyze price differences between master list and supplier prices.
    Returns price comparison data and summary statistics.
    """
    price_analysis = []
    
    # Analyze each item in master list
    for _, master_item in master_df.iterrows():
        # Get all supplier prices for this item
        supplier_prices = suppliers_df[suppliers_df['Item #'] == master_item['Item #']]['Supplier price']
        
        if not supplier_prices.empty and pd.notnull(master_item['Price']):
            min_supplier_price = supplier_prices.min()
            max_supplier_price = supplier_prices.max()
            avg_supplier_price = supplier_prices.mean()
            potential_savings = master_item['Price'] - min_supplier_price
            savings_percentage = (potential_savings / master_item['Price'] * 100) if master_item['Price'] > 0 else 0
            
            price_analysis.append({
                'Item #': master_item['Item #'],
                'Name': master_item['Name'],
                'Category': master_item['CATEGORY'],
                'Listed Price': master_item['Price'],
                'Best Price': min_supplier_price,
                'Highest Price': max_supplier_price,
                'Avg Supplier Price': avg_supplier_price,
                'Potential Savings': potential_savings,
                'Savings %': savings_percentage,
                'Best Supplier': suppliers_df[
                    (suppliers_df['Item #'] == master_item['Item #']) & 
                    (suppliers_df['Supplier price'] == min_supplier_price)
                ]['Supplier name'].iloc[0]
            })
    
    price_df = pd.DataFrame(price_analysis)
    
    # Calculate summary statistics
    summary_stats = {
        'Total Potential Savings': price_df['Potential Savings'].sum(),
        'Average Savings %': price_df['Savings %'].mean(),
        'Items Analyzed': len(price_df),
        'Items with Savings': len(price_df[price_df['Potential Savings'] > 0])
    }
    
    # Calculate category-level statistics
    category_stats = price_df.groupby('Category').agg({
        'Potential Savings': ['sum', 'mean', 'count'],
        'Savings %': 'mean'
    }).round(2)
    
    category_stats.columns = ['Total Savings', 'Avg Savings per Item', 'Item Count', 'Avg Savings %']
    category_stats = category_stats.reset_index()
    
    return price_df, summary_stats, category_stats

def display_price_analysis_tab(master_df, suppliers_df):
    """
    Display the price analysis tab in Streamlit
    """
    st.markdown("### 💰 Price Analysis")
    st.markdown("Analysis of potential cost savings based on supplier prices vs. listed prices.")
    
    # Calculate price analysis
    price_df, summary_stats, category_stats = analyze_prices(master_df, suppliers_df)
    
    # Display summary metrics in a nice grid
    st.markdown("#### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Potential Savings",
            f"${summary_stats['Total Potential Savings']:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Average Savings %",
            f"{summary_stats['Average Savings %']:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Items Analyzed",
            f"{summary_stats['Items Analyzed']}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Items with Savings",
            f"{summary_stats['Items with Savings']}",
            delta=None
        )
    
    # Category Analysis
    st.markdown("#### Category Analysis")
    
    # Create a bar chart for category analysis
    fig = go.Figure()
    
    # Add bars for Total Savings
    fig.add_trace(go.Bar(
        x=category_stats['Category'],
        y=category_stats['Total Savings'],
        name='Total Savings ($)',
        marker_color='#4f46e5'
    ))
    
    # Add line for Average Savings %
    fig.add_trace(go.Scatter(
        x=category_stats['Category'],
        y=category_stats['Avg Savings %'],
        name='Average Savings %',
        yaxis='y2',
        line=dict(color='#ef4444', width=3)
    ))
    
    fig.update_layout(
        title='Savings Analysis by Category',
        yaxis=dict(
            title='Total Savings ($)',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis2=dict(
            title='Average Savings %',
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        plot_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        height=500,
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            tickangle=45
        ),
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display category statistics table
    st.markdown("#### Category Details")
    
    # Format category stats for display
    formatted_category_stats = category_stats.copy()
    formatted_category_stats['Total Savings'] = formatted_category_stats['Total Savings'].apply(lambda x: f"${x:,.2f}")
    formatted_category_stats['Avg Savings per Item'] = formatted_category_stats['Avg Savings per Item'].apply(lambda x: f"${x:,.2f}")
    formatted_category_stats['Avg Savings %'] = formatted_category_stats['Avg Savings %'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(formatted_category_stats, use_container_width=True)
    
    # Display detailed price analysis
    st.markdown("#### Detailed Price Analysis")
    st.markdown("Items sorted by potential savings (highest to lowest)")
    
    # Sort by potential savings
    detailed_analysis = price_df.sort_values('Potential Savings', ascending=False).copy()
    
    # Format numeric columns for display
    display_cols = [
        'Item #', 'Name', 'Category', 'Listed Price', 'Best Price',
        'Potential Savings', 'Savings %', 'Best Supplier'
    ]
    
    display_df = detailed_analysis[display_cols].copy()
    
    # Format currency columns
    currency_cols = ['Listed Price', 'Best Price', 'Potential Savings']
    for col in currency_cols:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
    
    # Format percentage column
    display_df['Savings %'] = display_df['Savings %'].apply(lambda x: f"{x:.1f}%")
    
    # Display the formatted dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Add export functionality
    st.markdown("#### Export Analysis")
    st.download_button(
        "📥 Download Full Analysis",
        detailed_analysis.to_csv(index=False).encode('utf-8'),
        "price_analysis.csv",
        "text/csv",
        key='download-price-analysis'
    )

####################################################################################################################
def analyze_suppliers(master_df, suppliers_df):
    """
    Enhanced supplier analysis considering price, lead time, and coverage
    """
    # Initialize supplier analysis lists
    price_analysis = []
    supplier_stats = {}
    
    # Analyze each item in master list
    for _, master_item in master_df.iterrows():
        item_suppliers = suppliers_df[suppliers_df['Item #'] == master_item['Item #']]
        
        if not item_suppliers.empty and pd.notnull(master_item['Price']):
            # Get all supplier information for this item
            for _, supplier in item_suppliers.iterrows():
                supplier_name = supplier['Supplier name']
                
                # Initialize supplier stats if not exists
                if supplier_name not in supplier_stats:
                    supplier_stats[supplier_name] = {
                        'total_items': 0,
                        'best_price_count': 0,
                        'total_potential_savings': 0,
                        'avg_lead_time': 0,
                        'lead_times': [],
                        'categories_covered': set(),
                        'competitive_items': 0  # Price within 10% of best price
                    }
                
                supplier_stats[supplier_name]['total_items'] += 1
                supplier_stats[supplier_name]['categories_covered'].add(master_item['CATEGORY'])
                
                if pd.notnull(supplier['Supplier lead time']):
                    supplier_stats[supplier_name]['lead_times'].append(supplier['Supplier lead time'])
            
            # Find best price and analyze supplier performance
            min_price = item_suppliers['Supplier price'].min()
            competitive_threshold = min_price * 1.10  # Within 10% of best price
            
            best_suppliers = item_suppliers[item_suppliers['Supplier price'] == min_price]
            competitive_suppliers = item_suppliers[item_suppliers['Supplier price'] <= competitive_threshold]
            
            # Update supplier stats
            for _, supplier in best_suppliers.iterrows():
                supplier_stats[supplier['Supplier name']]['best_price_count'] += 1
                potential_savings = master_item['Price'] - supplier['Supplier price']
                if potential_savings > 0:
                    supplier_stats[supplier['Supplier name']]['total_potential_savings'] += potential_savings
            
            for _, supplier in competitive_suppliers.iterrows():
                supplier_stats[supplier['Supplier name']]['competitive_items'] += 1
            
            # Create item analysis entry
            price_analysis.append({
                'Item #': master_item['Item #'],
                'Name': master_item['Name'],
                'Category': master_item['CATEGORY'],
                'Listed Price': master_item['Price'],
                'Best Price': min_price,
                'Best Supplier(s)': ', '.join(best_suppliers['Supplier name']),
                'All Suppliers': len(item_suppliers),
                'Competitive Suppliers': len(competitive_suppliers),
                'Price Range': f"${min_price:.2f} - ${item_suppliers['Supplier price'].max():.2f}",
                'Best Lead Time': item_suppliers['Supplier lead time'].min(),
                'Potential Savings': master_item['Price'] - min_price
            })
    
    # Calculate final supplier statistics
    for supplier in supplier_stats:
        lead_times = supplier_stats[supplier]['lead_times']
        supplier_stats[supplier]['avg_lead_time'] = sum(lead_times) / len(lead_times) if lead_times else None
        supplier_stats[supplier]['categories_covered'] = len(supplier_stats[supplier]['categories_covered'])
    
    # Convert to DataFrames
    price_df = pd.DataFrame(price_analysis)
    
    supplier_summary = pd.DataFrame([{
        'Supplier': supplier,
        'Total Items': stats['total_items'],
        'Best Price Count': stats['best_price_count'],
        'Competitive Items': stats['competitive_items'],
        'Categories Covered': stats['categories_covered'],
        'Avg Lead Time': stats['avg_lead_time'],
        'Total Potential Savings': stats['total_potential_savings'],
        'Best Price %': (stats['best_price_count'] / stats['total_items'] * 100) if stats['total_items'] > 0 else 0,
        'Competitive %': (stats['competitive_items'] / stats['total_items'] * 100) if stats['total_items'] > 0 else 0
    } for supplier, stats in supplier_stats.items()])
    
    return price_df, supplier_summary

def display_supplier_analysis_tab(master_df, suppliers_df):
    """
    Display enhanced supplier analysis in Streamlit
    """
    st.markdown("### 🏢 Supplier Analysis")
    st.markdown("Comprehensive analysis of supplier performance across price, coverage, and lead time.")
    
    # Calculate analyses
    price_df, supplier_summary = analyze_suppliers(master_df, suppliers_df)
    
    # Sort suppliers by performance metrics
    supplier_summary = supplier_summary.sort_values('Total Potential Savings', ascending=False)
    
    # Display supplier performance metrics
    st.markdown("#### Supplier Performance Overview")
    
    # Create supplier performance chart
    fig = go.Figure()
    
    # Add bars for Best Price Count
    fig.add_trace(go.Bar(
        x=supplier_summary['Supplier'],
        y=supplier_summary['Best Price Count'],
        name='Best Price Items',
        marker_color='#4f46e5'
    ))
    
    # Add bars for Competitive Items
    fig.add_trace(go.Bar(
        x=supplier_summary['Supplier'],
        y=supplier_summary['Competitive Items'],
        name='Competitive Items',
        marker_color='#10b981'
    ))
    
    # Add line for Average Lead Time
    fig.add_trace(go.Scatter(
        x=supplier_summary['Supplier'],
        y=supplier_summary['Avg Lead Time'],
        name='Avg Lead Time (days)',
        yaxis='y2',
        line=dict(color='#ef4444', width=3)
    ))
    
    fig.update_layout(
        title='Supplier Performance Metrics',
        yaxis=dict(title='Number of Items'),
        yaxis2=dict(
            title='Average Lead Time (days)',
            overlaying='y',
            side='right'
        ),
        barmode='group',
        height=500,
        xaxis=dict(tickangle=45),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed supplier statistics
    st.markdown("#### Detailed Supplier Statistics")
    
    # Format supplier summary for display
    display_summary = supplier_summary.copy()
    display_summary['Total Potential Savings'] = display_summary['Total Potential Savings'].apply(lambda x: f"${x:,.2f}")
    display_summary['Best Price %'] = display_summary['Best Price %'].apply(lambda x: f"{x:.1f}%")
    display_summary['Competitive %'] = display_summary['Competitive %'].apply(lambda x: f"{x:.1f}%")
    display_summary['Avg Lead Time'] = display_summary['Avg Lead Time'].apply(lambda x: f"{x:.1f} days" if pd.notnull(x) else "N/A")
    
    st.dataframe(display_summary, use_container_width=True)
    
    # Display item-level analysis
    st.markdown("#### Item-Level Supplier Analysis")
    st.markdown("Analysis of supplier competition and pricing for each item")
    
    # Format price analysis for display
    detailed_analysis = price_df.sort_values('Potential Savings', ascending=False)
    
    # Format the display columns
    display_cols = [
        'Item #', 'Name', 'Category', 'Listed Price', 'Best Price',
        'Best Supplier(s)', 'All Suppliers', 'Competitive Suppliers',
        'Price Range', 'Best Lead Time'
    ]
    
    display_df = detailed_analysis[display_cols].copy()
    
    # Format currency columns
    display_df['Listed Price'] = display_df['Listed Price'].apply(lambda x: f"${x:,.2f}")
    display_df['Best Price'] = display_df['Best Price'].apply(lambda x: f"${x:,.2f}")
    display_df['Best Lead Time'] = display_df['Best Lead Time'].apply(lambda x: f"{x:.0f} days" if pd.notnull(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Add export functionality
    st.markdown("#### Export Analysis")
    st.download_button(
        "📥 Download Supplier Analysis",
        detailed_analysis.to_csv(index=False).encode('utf-8'),
        "supplier_analysis.csv",
        "text/csv",
        key='download-supplier-analysis'
    )


#########################################################################################################################
def analyze_advanced_insights(master_df, suppliers_df):
    """
    Generate advanced insights from supplier and pricing data
    """
    insights = {
        'pricing_insights': [],
        'supplier_insights': [],
        'risk_insights': [],
        'opportunity_insights': []
    }
    
    # 1. Price Spread Analysis
    price_spreads = {}
    for _, master_item in master_df.iterrows():
        item_suppliers = suppliers_df[suppliers_df['Item #'] == master_item['Item #']]
        if not item_suppliers.empty:
            prices = item_suppliers['Supplier price']
            spread = (prices.max() - prices.min()) / prices.min() * 100 if prices.min() > 0 else 0
            price_spreads[master_item['Item #']] = {
                'name': master_item['Name'],
                'spread': spread,
                'supplier_count': len(item_suppliers),
                'category': master_item['CATEGORY'],
                'commodity': master_item['Commodity']
            }
    
    # High spread items (potential for negotiation)
    high_spread_items = {k: v for k, v in price_spreads.items() if v['spread'] > 30}
    if high_spread_items:
        insights['pricing_insights'].append({
            'type': 'High Price Spread',
            'description': f"Found {len(high_spread_items)} items with price spread >30%",
            'details': high_spread_items
        })
    
    # 2. Supplier Concentration Analysis
    category_suppliers = {}
    for _, row in master_df.iterrows():
        category = row['CATEGORY']
        item_suppliers = suppliers_df[suppliers_df['Item #'] == row['Item #']]
        if category not in category_suppliers:
            category_suppliers[category] = set()
        category_suppliers[category].update(item_suppliers['Supplier name'])
    
    # Categories with low supplier diversity
    low_diversity_categories = {
        cat: len(suppliers) for cat, suppliers in category_suppliers.items()
        if len(suppliers) < 3
    }
    if low_diversity_categories:
        insights['risk_insights'].append({
            'type': 'Low Supplier Diversity',
            'description': 'Categories with less than 3 suppliers',
            'details': low_diversity_categories
        })
    
    # 3. Lead Time Analysis
    supplier_lead_times = {}
    for _, row in suppliers_df.iterrows():
        if pd.notnull(row['Supplier lead time']):
            if row['Supplier name'] not in supplier_lead_times:
                supplier_lead_times[row['Supplier name']] = []
            supplier_lead_times[row['Supplier name']].append(row['Supplier lead time'])
    
    lead_time_stats = {
        supplier: {
            'avg': np.mean(times),
            'std': np.std(times),
            'max': max(times),
            'min': min(times)
        }
        for supplier, times in supplier_lead_times.items()
    }
    
    # High lead time variability suppliers
    variable_lead_time_suppliers = {
        supplier: stats for supplier, stats in lead_time_stats.items()
        if stats['std'] > 5  # More than 5 days standard deviation
    }
    if variable_lead_time_suppliers:
        insights['risk_insights'].append({
            'type': 'Lead Time Variability',
            'description': 'Suppliers with high lead time variability',
            'details': variable_lead_time_suppliers
        })
    
    # 4. Competitive Analysis
    supplier_competitiveness = {}
    for supplier in suppliers_df['Supplier name'].unique():
        supplier_items = suppliers_df[suppliers_df['Supplier name'] == supplier]
        competitive_count = 0
        total_items = len(supplier_items)
        
        for _, item in supplier_items.iterrows():
            all_prices = suppliers_df[suppliers_df['Item #'] == item['Item #']]['Supplier price']
            min_price = all_prices.min()
            if item['Supplier price'] <= min_price * 1.1:  # Within 10% of best price
                competitive_count += 1
        
        supplier_competitiveness[supplier] = {
            'competitive_items': competitive_count,
            'total_items': total_items,
            'competitive_rate': (competitive_count / total_items * 100) if total_items > 0 else 0
        }
    
    # 5. Single Source Risk Analysis
    single_source_items = []
    for _, master_item in master_df.iterrows():
        supplier_count = len(suppliers_df[suppliers_df['Item #'] == master_item['Item #']])
        if supplier_count == 1:
            single_source_items.append({
                'item': master_item['Item #'],
                'name': master_item['Name'],
                'category': master_item['CATEGORY']
            })
    
    if single_source_items:
        insights['risk_insights'].append({
            'type': 'Single Source Risk',
            'description': f"Found {len(single_source_items)} single-sourced items",
            'details': single_source_items
        })
    
    # 6. Savings Opportunities Analysis
    savings_by_category = {}
    for _, master_item in master_df.iterrows():
        category = master_item['CATEGORY']
        item_suppliers = suppliers_df[suppliers_df['Item #'] == master_item['Item #']]
        
        if not item_suppliers.empty and pd.notnull(master_item['Price']):
            min_supplier_price = item_suppliers['Supplier price'].min()
            potential_savings = master_item['Price'] - min_supplier_price
            
            if category not in savings_by_category:
                savings_by_category[category] = 0
            if potential_savings > 0:
                savings_by_category[category] += potential_savings
    
    # Identify top savings opportunities
    top_savings_categories = dict(sorted(
        savings_by_category.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3])
    
    insights['opportunity_insights'].append({
        'type': 'Top Savings Categories',
        'description': 'Categories with highest potential savings',
        'details': top_savings_categories
    })
    
    return insights

def display_advanced_insights_tab(master_df, suppliers_df):
    """
    Display advanced insights in Streamlit
    """
    st.markdown("### 🔍 Advanced Insights")
    st.markdown("Deep analysis of pricing patterns, supplier relationships, and risk factors.")
    
    # Get insights
    insights = analyze_advanced_insights(master_df, suppliers_df)
    
    # Create tabs for different insight categories
    insight_tabs = st.tabs(['💰 Pricing', '🏢 Supplier', '⚠️ Risk', '✨ Opportunities'])
    
    with insight_tabs[0]:
        st.markdown("#### Pricing Insights")
        for insight in insights['pricing_insights']:
            with st.expander(f"📊 {insight['type']}"):
                st.markdown(f"**{insight['description']}**")
                
                # Create a DataFrame from the details
                if isinstance(insight['details'], dict):
                    df = pd.DataFrame.from_dict(insight['details'], orient='index')
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write(insight['details'])
    
    with insight_tabs[1]:
        st.markdown("#### Supplier Insights")
        for insight in insights['supplier_insights']:
            with st.expander(f"🏢 {insight['type']}"):
                st.markdown(f"**{insight['description']}**")
                
                if isinstance(insight['details'], dict):
                    df = pd.DataFrame.from_dict(insight['details'], orient='index')
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write(insight['details'])
    
    with insight_tabs[2]:
        st.markdown("#### Risk Analysis")
        for insight in insights['risk_insights']:
            with st.expander(f"⚠️ {insight['type']}"):
                st.markdown(f"**{insight['description']}**")
                
                if isinstance(insight['details'], dict):
                    if insight['type'] == 'Lead Time Variability':
                        # Create a more readable format for lead time stats
                        df = pd.DataFrame.from_dict(insight['details'], orient='index')
                        df = df.round(2)
                        st.dataframe(df, use_container_width=True)
                    else:
                        df = pd.DataFrame.from_dict(insight['details'], orient='index')
                        st.dataframe(df, use_container_width=True)
                elif isinstance(insight['details'], list):
                    df = pd.DataFrame(insight['details'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write(insight['details'])
    
    with insight_tabs[3]:
        st.markdown("#### Opportunity Analysis")
        for insight in insights['opportunity_insights']:
            with st.expander(f"✨ {insight['type']}"):
                st.markdown(f"**{insight['description']}**")
                
                if insight['type'] == 'Top Savings Categories':
                    # Create a bar chart for savings opportunities
                    fig = go.Figure(go.Bar(
                        x=list(insight['details'].keys()),
                        y=list(insight['details'].values()),
                        marker_color='#4f46e5'
                    ))
                    
                    fig.update_layout(
                        title='Potential Savings by Category',
                        xaxis_title="Category",
                        yaxis_title="Potential Savings ($)",
                        height=400,
                        xaxis=dict(tickangle=45)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed breakdown
                    st.markdown("**Detailed Breakdown:**")
                    for category, savings in insight['details'].items():
                        st.markdown(f"- {category}: ${savings:,.2f}")
                else:
                    st.write(insight['details'])

##########################################################################################################################


def prepare_data_for_correlation(master_df, suppliers_df, transactions_df=None):
    """
    Prepare and combine data from all sources for correlation analysis.
    """
    # Initialize metrics dictionary for each item
    metrics = {}
    
    # Process master and supplier data
    for item in master_df['Item #'].unique():
        # Get item data from master
        master_item = master_df[master_df['Item #'] == item].iloc[0]
        
        # Get supplier data for item
        item_suppliers = suppliers_df[suppliers_df['Item #'] == item]
        
        if not item_suppliers.empty:
            # Basic metrics
            base_metrics = {
                'listed_price': master_item['Price'],
                'supplier_count': len(item_suppliers),
                'avg_supplier_price': item_suppliers['Supplier price'].mean(),
                'min_supplier_price': item_suppliers['Supplier price'].min(),
                'max_supplier_price': item_suppliers['Supplier price'].max(),
                'price_spread': item_suppliers['Supplier price'].max() - item_suppliers['Supplier price'].min(),
                'price_variance': item_suppliers['Supplier price'].var(),
                'price_std': item_suppliers['Supplier price'].std(),
            }
            
            # Lead time metrics if available
            if 'Supplier lead time' in item_suppliers.columns:
                lead_times = item_suppliers['Supplier lead time'].dropna()
                if not lead_times.empty:
                    base_metrics.update({
                        'avg_lead_time': lead_times.mean(),
                        'min_lead_time': lead_times.min(),
                        'max_lead_time': lead_times.max(),
                        'lead_time_spread': lead_times.max() - lead_times.min(),
                        'lead_time_std': lead_times.std()
                    })
            
            # Price competitiveness metrics
            best_price = item_suppliers['Supplier price'].min()
            base_metrics.update({
                'competitive_suppliers': len(item_suppliers[
                    item_suppliers['Supplier price'] <= best_price * 1.1
                ]),
                'price_to_market_ratio': master_item['Price'] / best_price if best_price > 0 else np.nan,
                'potential_savings_pct': (
                    (master_item['Price'] - best_price) / master_item['Price'] * 100
                    if master_item['Price'] > 0 else np.nan
                )
            })
            
            metrics[item] = base_metrics
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    
    # Add transaction metrics if available
    if transactions_df is not None:
        transaction_metrics = transactions_df.groupby('Item #').agg({
            'Quantity': ['count', 'mean', 'std', 'min', 'max'],
            'Total Amount': ['mean', 'sum', 'std']
        })
        
        transaction_metrics.columns = [
            'transaction_count', 'avg_quantity', 'quantity_std',
            'min_quantity', 'max_quantity', 'avg_transaction_value',
            'total_spend', 'spend_std'
        ]
        
        metrics_df = metrics_df.join(transaction_metrics)
    
    return metrics_df
# # # # # ### ####### #############################################
def analyze_correlations(metrics_df, threshold=0.5):
    """
    Perform detailed correlation analysis on the metrics.
    """
    # Calculate correlation matrix
    correlation_matrix = metrics_df.corr()
    
    # Find significant correlations
    significant_correlations = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            correlation = correlation_matrix.iloc[i, j]
            if abs(correlation) >= threshold:
                # Get the column names
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                
                # Get the data, dropping any NA values
                data1 = metrics_df[col1].dropna()
                data2 = metrics_df[col2].dropna()
                
                # Get the common index
                common_idx = data1.index.intersection(data2.index)
                
                # Only calculate p-value if we have enough data points
                if len(common_idx) > 1:
                    # Get aligned data
                    aligned_data1 = data1[common_idx]
                    aligned_data2 = data2[common_idx]
                    
                    try:
                        _, p_value = stats.pearsonr(aligned_data1, aligned_data2)
                    except:
                        p_value = np.nan
                else:
                    p_value = np.nan
                
                significant_correlations.append({
                    'metric_1': col1,
                    'metric_2': col2,
                    'correlation': correlation,
                    'p_value': p_value,
                    'significance': 'Significant' if (not np.isnan(p_value) and p_value < 0.05) else 'Not Significant',
                    'sample_size': len(common_idx)
                })
    
    return correlation_matrix, pd.DataFrame(significant_correlations)

def interpret_correlation(value):
    """
    Provide interpretation for correlation value.
    """
    abs_value = abs(value)
    direction = "positive" if value > 0 else "negative"
    
    if abs_value >= 0.9:
        strength = "Very strong"
    elif abs_value >= 0.7:
        strength = "Strong"
    elif abs_value >= 0.5:
        strength = "Moderate"
    elif abs_value >= 0.3:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    return f"{strength} {direction}"

## ## #### ###########################################################

def display_correlation_analysis(master_df, suppliers_df, transactions_df=None):
    """
    Display comprehensive correlation analysis in Streamlit.
    """
    st.markdown("### 🔗 Correlation Analysis")
    st.markdown("""
        This analysis examines relationships between various metrics in your procurement data. 
        Understanding these correlations can help identify patterns, opportunities, and potential risks.
    """)
    
    # Prepare data
    metrics_df = prepare_data_for_correlation(master_df, suppliers_df, transactions_df)
    
    # Remove columns with all NaN values
    metrics_df = metrics_df.dropna(axis=1, how='all')
    
    # Analysis settings
    col1, col2 = st.columns([2, 1])
    with col1:
        correlation_threshold = st.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Show correlations with absolute value above this threshold"
        )
    
    # Perform correlation analysis
    correlation_matrix, significant_df = analyze_correlations(metrics_df, correlation_threshold)
    
    # 1. Overview Section
    st.markdown("#### Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Metrics Analyzed",
            len(correlation_matrix.columns)
        )
    
    with col2:
        st.metric(
            "Significant Correlations",
            len(significant_df)
        )
    
    with col3:
        if not significant_df.empty:
            max_corr = significant_df['correlation'].abs().max()
            st.metric(
                "Strongest Correlation",
                f"{max_corr:.3f}"
            )
        else:
            st.metric("Strongest Correlation", "N/A")
    
    # 2. Correlation Matrix Heatmap
    st.markdown("#### Correlation Matrix Heatmap")
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title='Correlation Matrix of Procurement Metrics',
        height=800,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Significant Correlations
    if not significant_df.empty:
        st.markdown(f"#### Significant Correlations (|r| ≥ {correlation_threshold})")
        
        # Format and sort correlations
        display_df = significant_df.copy()
        display_df['correlation'] = display_df['correlation'].round(3)
        display_df['p_value'] = display_df['p_value'].round(4)
        display_df['interpretation'] = display_df['correlation'].apply(interpret_correlation)
        display_df = display_df.sort_values('correlation', key=abs, ascending=False)
        
        st.dataframe(display_df, use_container_width=True)
        
        # 4. Top Correlation Details
        st.markdown("#### Top Correlation Analysis")
        
        for idx, row in display_df.head(3).iterrows():
            with st.expander(
                f"{row['metric_1']} vs {row['metric_2']} "
                f"(r = {row['correlation']:.3f})"
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Get data for scatter plot, dropping NaN values
                    plot_data = metrics_df[[row['metric_1'], row['metric_2']]].dropna()
                    
                    if not plot_data.empty:
                        # Scatter plot
                        fig = px.scatter(
                            plot_data,
                            x=row['metric_1'],
                            y=row['metric_2'],
                            trendline="ols",
                            title=f"Correlation Analysis: {row['interpretation']}"
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Insufficient data for scatter plot")
                
                with col2:
                    # Statistical details
                    st.markdown("**Statistical Details:**")
                    st.markdown(f"- Correlation: {row['correlation']:.3f}")
                    st.markdown(f"- P-value: {row['p_value']:.4f}")
                    st.markdown(f"- Significance: {row['significance']}")
                    st.markdown(f"- Sample Size: {row['sample_size']}")
                    st.markdown(f"- Interpretation: {row['interpretation']}")
                    
                    # Basic statistics
                    st.markdown("\n**Basic Statistics:**")
                    stats_df = metrics_df[[row['metric_1'], row['metric_2']]].describe()
                    st.dataframe(stats_df)
    else:
        st.info(f"No correlations found with absolute value ≥ {correlation_threshold}")
    
    # 5. Business Insights
    st.markdown("#### Business Insights")
    
    if not significant_df.empty:
        # Group correlations by type
        price_correlations = significant_df[
            (significant_df['metric_1'].str.contains('price', case=False)) |
            (significant_df['metric_2'].str.contains('price', case=False))
        ]
        
        supplier_correlations = significant_df[
            (significant_df['metric_1'].str.contains('supplier', case=False)) |
            (significant_df['metric_2'].str.contains('supplier', case=False))
        ]
        
        if not price_correlations.empty:
            st.markdown("**Price-Related Insights:**")
            for _, row in price_correlations.iterrows():
                st.markdown(
                    f"- {row['metric_1']} and {row['metric_2']} show a "
                    f"{interpret_correlation(row['correlation']).lower()} relationship "
                    f"(r = {row['correlation']:.3f})"
                )
        
        if not supplier_correlations.empty:
            st.markdown("**Supplier-Related Insights:**")
            for _, row in supplier_correlations.iterrows():
                st.markdown(
                    f"- {row['metric_1']} and {row['metric_2']} show a "
                    f"{interpret_correlation(row['correlation']).lower()} relationship "
                    f"(r = {row['correlation']:.3f})"
                )
    
    # 6. Download Section
    st.markdown("#### Export Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download correlation matrix
        csv = correlation_matrix.to_csv()
        st.download_button(
            label="📥 Download Correlation Matrix",
            data=csv,
            file_name="correlation_matrix.csv",
            mime="text/csv",
        )
    
    with col2:
        # Download metrics data
        csv = metrics_df.to_csv()
        st.download_button(
            label="📥 Download Complete Metrics",
            data=csv,
            file_name="procurement_metrics.csv",
            mime="text/csv",
        )

###############################################################################################################################
# Helper Functions for Tab Content

def display_trends_content(transactions_df):
    """
    Display monthly trends analysis with transaction count and amount.
    
    Args:
        transactions_df (pd.DataFrame): DataFrame containing transaction data
    """
    # Calculate monthly statistics
    monthly_stats = (
        transactions_df.groupby(
            pd.to_datetime(transactions_df['Transaction Date']).dt.strftime('%Y-%m')
        ).agg({
            'Transaction Date': 'count',
            'Total Amount': 'sum'
        })
        .rename(columns={
            'Transaction Date': 'Count',
            'Total Amount': 'Amount'
        })
        .reset_index()
        .rename(columns={
            'Transaction Date': 'Month'
        })
    )
    
    # Create figure with dual axes
    fig = go.Figure()
    
    # Add transaction count trace
    fig.add_trace(go.Scatter(
        x=monthly_stats['Month'],
        y=monthly_stats['Count'],
        name='Transaction Count',
        line=dict(color='#4f46e5', width=3)
    ))
    
    # Add amount trace on secondary axis
    fig.add_trace(go.Scatter(
        x=monthly_stats['Month'],
        y=monthly_stats['Amount'],
        name='Total Amount ($)',
        line=dict(color='#ef4444', width=3),
        yaxis='y2'
    ))
    
    # Update layout with dual axes
    fig.update_layout(
        title='Monthly Transaction Trends',
        yaxis=dict(
            title='Transaction Count',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis2=dict(
            title='Total Amount ($)',
            overlaying='y',
            side='right',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        plot_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        height=600,
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            tickangle=45
        ),
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.markdown("### Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Monthly Transactions",
            f"{monthly_stats['Count'].mean():.0f}"
        )
    
    with col2:
        st.metric(
            "Average Monthly Amount",
            f"${monthly_stats['Amount'].mean():,.2f}"
        )
    
    with col3:
        st.metric(
            "Total Months",
            f"{len(monthly_stats)}"
        )

def display_category_content(transactions_df):
    """
    Display category-wise analysis with transaction count and amount distribution.
    
    Args:
        transactions_df (pd.DataFrame): DataFrame containing transaction data
    """
    # Calculate category statistics
    category_stats = (
        transactions_df.groupby('Category')
        .agg({
            'Transaction Date': 'count',
            'Total Amount': 'sum',
            'Quantity': 'sum'
        })
        .rename(columns={
            'Transaction Date': 'Count',
            'Total Amount': 'Amount',
            'Quantity': 'Total Quantity'
        })
        .reset_index()
    )
    
    # Display category analysis in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction count by category
        fig = px.bar(
            category_stats,
            x='Category',
            y='Count',
            title='Transaction Count by Category',
            color='Category',
            height=400
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Number of Transactions",
            plot_bgcolor='white',
            xaxis=dict(tickangle=45)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amount distribution by category
        fig = px.pie(
            category_stats,
            values='Amount',
            names='Category',
            title='Total Amount by Category',
            height=400
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed category statistics
    st.markdown("### Category Details")
    
    # Format the statistics for display
    display_stats = category_stats.copy()
    display_stats['Amount'] = display_stats['Amount'].apply(lambda x: f"${x:,.2f}")
    display_stats['Percentage'] = (
        category_stats['Amount'] / category_stats['Amount'].sum() * 100
    ).apply(lambda x: f"{x:.1f}%")
    
    # Add metrics per category
    st.dataframe(
        display_stats.style.set_properties(**{'text-align': 'right'}),
        use_container_width=True
    )
    
    # Add summary metrics
    st.markdown("### Overall Category Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Number of Categories",
            f"{len(category_stats)}"
        )
    
    with col2:
        st.metric(
            "Most Active Category",
            f"{category_stats.loc[category_stats['Count'].idxmax(), 'Category']}"
        )
    
    with col3:
        st.metric(
            "Highest Value Category",
            f"{category_stats.loc[category_stats['Amount'].idxmax(), 'Category']}"
        )


#####################################################################################################

def main():
    """
    Main application function for Material Acquisitions Analytics.
    """
    try:
        # 1. Application Header
        display_procuresync_title()

        # 2. How to Use Section
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            ### Quick Guide
            
            1. **Upload Data Files**
            - Ensure `Master.csv` and `Suppliers.csv` are in the application directory
            - Files must contain required columns (Item #, Name, Category, etc.)
            
            2. **Configure Settings ⚙️**
            - **Start & End Date**: Select your desired date range
            - **Min/Max Transactions**: Set the range of transactions per day
            - **Transaction Probability**: Chance of transactions occurring on any day
            
            3. **Generate & Analyze 📈**
            - Click "Generate Transactions" to create your dataset
            - View data in different analysis tabs
            
            4. **Export Results 📥**
            - Use the "Download CSV" button to save your generated data
            """)
            
            # Sample data format
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Sample Master.csv format:")
                st.code("""
    Item #,Name,CATEGORY,Commodity
    V1000,Metal Stud,FRAMING/DRYWALL,Structural
    V1005,Track Metal,FRAMING/DRYWALL,Structural
                """.strip())
            
            with col2:
                st.markdown("#### Sample Suppliers.csv format:")
                st.code("""
    Item #,Supplier name,Supplier price
    V1000,Supplier A,10.99
    V1000,Supplier B,9.99
                """.strip())

        # 3. Load Data
        master_df, suppliers_df = load_data()
        if master_df is None or suppliers_df is None:
            return

        # 4. Settings Configuration
        with st.expander("⚙️ Settings", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    datetime(2022, 1, 1),
                    key="start_date",
                    help="Select the start date for transaction generation"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    datetime(2024, 3, 31),
                    key="end_date",
                    help="Select the end date for transaction generation"
                )
            
            col3, col4, col5 = st.columns(3)
            with col3:
                min_trans = st.number_input(
                    "Min Transactions/Day",
                    min_value=1,
                    max_value=100,
                    value=5,
                    help="Minimum number of transactions per day"
                )
            with col4:
                max_trans = st.number_input(
                    "Max Transactions/Day",
                    min_value=1,
                    max_value=100,
                    value=15,
                    help="Maximum number of transactions per day"
                )
            with col5:
                prob = st.slider(
                    "Transaction Probability (%)",
                    min_value=0,
                    max_value=100,
                    value=70,
                    help="Probability of transactions occurring on any given day"
                )



        # 5. Transaction Generation Button
        if st.button("🚀 Generate Transactions", type="primary", use_container_width=True):
            transactions = []
            current_date = start_date
            
            # Progress bar
            progress_text = "Generating transactions..."
            progress_bar = st.progress(0)
            total_days = (end_date - start_date).days
            
            with st.spinner(progress_text):
                for day_count, _ in enumerate(range(total_days + 1)):
                    if np.random.random() <= prob/100:
                        num_transactions = np.random.randint(min_trans, max_trans)
                        
                        # Get valid items (those that have suppliers)
                        valid_items = master_df[master_df['Item #'].isin(suppliers_df['Item #'].unique())]
                        
                        for _ in range(num_transactions):
                            try:
                                item = valid_items.sample(1).iloc[0]
                                available_suppliers = suppliers_df[
                                    suppliers_df['Item #'] == item['Item #']
                                ]
                                
                                if not available_suppliers.empty:
                                    supplier = available_suppliers.sample(1).iloc[0]
                                    quantity = generate_quantity(item['CATEGORY'], item['Commodity'])
                                    
                                    transactions.append({
                                        'Transaction Date': current_date.strftime('%Y-%m-%d'),
                                        'Item #': item['Item #'],
                                        'Item Name': item['Name'],
                                        'Supplier': supplier['Supplier name'],
                                        'Quantity': quantity,
                                        'Unit Price': supplier['Supplier price'],
                                        'Total Amount': quantity * supplier['Supplier price'],
                                        'Category': item['CATEGORY'],
                                        'Commodity': item['Commodity']
                                    })
                            except Exception as e:
                                st.error(f"Error generating transaction: {str(e)}")
                                continue
                    
                    current_date += timedelta(days=1)
                    progress_bar.progress((day_count + 1) / (total_days + 1))

                if transactions:
                    transactions_df = pd.DataFrame(transactions)
                    st.session_state.transactions_df = transactions_df
                    st.success(f"Generated {len(transactions)} transactions across {total_days + 1} days!")

                # 5. Create All Tabs
        #####
        
        # 5. Create Two Tab Groups
        st.markdown("### Transaction Analysis")
        transaction_tabs = st.tabs([
            "📋 Preview",
            "📈 Monthly Trends",
            "📊 Category Analysis",
            "💰 Price Analysis"
        ])

        st.markdown("### Advanced Analysis")
        analysis_tabs = st.tabs([
            "🏢 Supplier Analysis",
            "🔍 Advanced Insights",
            "🔗 Correlation Analysis"
        ])

        # 6. Transaction Tab Contents
        with transaction_tabs[0]:  # Preview
            if 'transactions_df' in st.session_state:
                st.dataframe(
                    st.session_state.transactions_df.style.format({
                        'Unit Price': '${:.2f}',
                        'Total Amount': '${:.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("Generate transactions to view data preview.")

        with transaction_tabs[1]:  # Monthly Trends
            if 'transactions_df' in st.session_state:
                display_trends_content(st.session_state.transactions_df)
            else:
                st.info("Generate transactions to view monthly trends.")

        with transaction_tabs[2]:  # Category Analysis
            if 'transactions_df' in st.session_state:
                display_category_content(st.session_state.transactions_df)
            else:
                st.info("Generate transactions to view category analysis.")

        with transaction_tabs[3]:  # Price Analysis
            if 'transactions_df' in st.session_state:
                display_price_analysis_tab(master_df, suppliers_df)
            else:
                st.info("Generate transactions to view price analysis.")

        # 7. Analysis Tab Contents
        with analysis_tabs[0]:  # Supplier Analysis
            if 'transactions_df' in st.session_state:
                display_supplier_analysis_tab(master_df, suppliers_df)
            else:
                st.info("Generate transactions to view supplier analysis.")

        with analysis_tabs[1]:  # Advanced Insights
            if 'transactions_df' in st.session_state:
                display_advanced_insights_tab(master_df, suppliers_df)
            else:
                st.info("Generate transactions to view advanced insights.")

        with analysis_tabs[2]:  # Correlation Analysis
            display_correlation_analysis(
                master_df,
                suppliers_df,
                st.session_state.transactions_df if 'transactions_df' in st.session_state else None
            )

        # 8. Download Section
        if 'transactions_df' in st.session_state:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("Your transaction data is ready for download!")
            with col2:
                st.download_button(
                    "📥 Download CSV",
                    st.session_state.transactions_df.to_csv(index=False).encode('utf-8'),
                    f"transactions_{start_date}_to_{end_date}.csv",
                    "text/csv",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data and settings and try again.")
        raise e

if __name__ == "__main__":
    main()
