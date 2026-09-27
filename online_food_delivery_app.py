import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sqlalchemy import create_engine

# Streamlit Dashboard Setup
st.set_page_config(
    page_title="Food Delivery EDA & Analysis Dashboard",
    page_icon="🍔",
    layout="wide",
)

st.title("🍔 Food Delivery Data Analysis & Dashboard")


# Data Processing and MySQL Insertion
@st.cache_data
def load_clean_and_ingest_data():
    df = pd.read_csv(r"C:\Users\acer\Downloads\ONINE_FOOD_DELIVERY_ANALYSIS.csv")

    df_clean = df.copy()

    # Drop Order_Time and handle datetime
    df_clean["Order_Date"] = pd.to_datetime(df_clean["Order_Date"])
    if "Order_Time" in df_clean.columns:
        df_clean = df_clean.drop("Order_Time", axis=1)

    # Calculate null percent quietly
    null_percent = df_clean.isnull().mean() * 100

    str_col = df_clean.select_dtypes(include="object").columns.to_list()

    # Handle missing values for string columns
    str_mode_cols = [
        "Customer_Gender",
        "City",
        "Area",
        "Cuisine_Type",
        "Payment_Mode",
    ]
    for col in str_mode_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    if "Peak_Hour" in df_clean.columns:
        df_clean["Peak_Hour"] = df_clean["Peak_Hour"].fillna("Unknown")

    if "Cancellation_Reason" in df_clean.columns:
        df_clean.loc[
            df_clean["Order_Status"] == "Cancelled", "Cancellation_Reason"
        ] = df_clean.loc[
            df_clean["Order_Status"] == "Cancelled", "Cancellation_Reason"
        ].fillna("Unknown Reason")
        df_clean["Cancellation_Reason"] = df_clean[
            "Cancellation_Reason"
        ].fillna("Not Cancelled")

    # Handle missing values for numerical columns
    num_col = df_clean.select_dtypes(include="number").columns.to_list()
    num_median_cols = [
        "Customer_Age",
        "Delivery_Time_Min",
        "Distance_km",
        "Order_Value",
        "Delivery_Rating",
    ]
    for col in num_median_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    if "Discount_Applied" in df_clean.columns:
        df_clean["Discount_Applied"] = df_clean["Discount_Applied"].fillna(0.0)

    if "Order_Value" in df_clean.columns and "Discount_Applied" in df_clean.columns:
        df_clean["Discount_Applied"] = df_clean[
            ["Discount_Applied", "Order_Value"]
        ].min(axis=1)
        df_clean["Final_Amount"] = (
            df_clean["Order_Value"] - df_clean["Discount_Applied"]
        )
        df_clean["Final_Amount"] = df_clean["Final_Amount"].clip(lower=0.0)

    df_clean = df_clean.dropna(subset=["Order_Date"])

    # Correct Ratings and Margins
    if "Restaurant_Rating" in df_clean.columns:
        df_clean["Restaurant_Rating"] = df_clean["Restaurant_Rating"].clip(
            upper=5.0
        )

    if "Profit_Margin" in df_clean.columns:
        df_clean["Profit_Margin"] = df_clean["Profit_Margin"].clip(lower=0.0)
        df_clean["Profit_Margin_Percentage"] = df_clean["Profit_Margin"] * 100

    # Force Delivery_Rating to NaN for Cancelled orders then fill
    if (
        "Order_Status" in df_clean.columns
        and "Delivery_Rating" in df_clean.columns
    ):
        df_clean.loc[
            df_clean["Order_Status"] == "Cancelled", "Delivery_Rating"
        ] = np.nan
        df_clean["Delivery_Rating"] = df_clean["Delivery_Rating"].fillna(3.0)

    # Feature Engineering Functions
    def del_per_cat(delivery_rating):
        if pd.isna(delivery_rating):
            return "Not Rated"
        elif delivery_rating <= 2.0:
            return "Poor"
        elif delivery_rating <= 3.0:
            return "Average"
        elif delivery_rating <= 5.0:
            return "High"
        else:
            return "Not Rated"

    df_clean["Delivery_Performance"] = df_clean["Delivery_Rating"].apply(
        del_per_cat
    )

    def age_group_category(age):
        if age <= 25:
            return "Young Adult"
        elif age <= 35:
            return "Early Career"
        elif age <= 50:
            return "Mid Adult"
        elif age <= 60:
            return "Senior Adult"
        else:
            return "Unknown"

    if "Customer_Age" in df_clean.columns:
        df_clean["Customer_Age_Group"] = df_clean["Customer_Age"].apply(
            age_group_category
        )
        df_clean["Customer_Age_Group"] = pd.Categorical(
            df_clean["Customer_Age_Group"],
            categories=[
                "Young Adult",
                "Early Career",
                "Mid Adult",
                "Senior Adult",
            ],
            ordered=True,
        )

    # Database Storage
    DB_USER = "root"
    DB_PASS = "yourpassword"
    DB_HOST = "localhost"
    DB_NAME = "food_delivery_db"

    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    )
    df_clean.to_sql(
        name="ofd",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=10000,
    )

    return df_clean


# Execute the ETL Pipeline quietly
with st.spinner("Processing data and setting up MySQL database..."):
    df_clean = load_clean_and_ingest_data()


# Database Connection for Queries
@st.cache_resource
def get_engine():
    DB_USER = "root"
    DB_PASS = "yourpassword"
    DB_HOST = "localhost"
    DB_NAME = "food_delivery_db"
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    )


engine = get_engine()

# Clean column names (strips whitespace, normalizes casing issues)
df_clean.columns = df_clean.columns.str.strip()

# Print columns in terminal/app sidebar for debugging if needed
# st.write("Available columns:", df_clean.columns.tolist())

# ------------------------------------------
# TAB CREATION & LAYOUT
# ------------------------------------------
tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🔍 SQL Queries"])

# ------------------------------------------
# TAB 1: EDA VISUALIZATIONS
# ------------------------------------------
with tab1:
    st.header("Exploratory Data Analysis Insights")

    # Dataset Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df_clean):,}")
    col2.metric("Total Columns", len(df_clean.columns))
    if "Final_Amount" in df_clean.columns:
        col3.metric("Total Revenue", f"₹{df_clean['Final_Amount'].sum():,.2f}")

    st.subheader("Dataset Preview")
    st.dataframe(df_clean.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Visual Analysis")

    # 1. Distribution of Order Value & Delivery Time
    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.markdown("**Order Value Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(
            df_clean["Order_Value"], kde=True, color="skyblue", bins=15, ax=ax1
        )
        ax1.set_title("Order_Value Distribution")
        ax1.set_xlabel("Order_Value")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)
        plt.close(fig1)

    with col_eda2:
        st.markdown("**Delivery Time Distribution**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(
            df_clean["Delivery_Time_Min"],
            kde=True,
            color="orange",
            bins=15,
            ax=ax2,
        )
        ax2.set_title("Delivery Time Distribution")
        ax2.set_xlabel("Delivery Time in Min")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")

    # 2. City-wise Order Analysis
    col_eda3, col_eda4 = st.columns(2)

    with col_eda3:
        st.markdown("**City-wise Order Count**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        city_counts = (
            df_clean.groupby("City")["Order_ID"].count().reset_index()
        )
        a1 = sns.barplot(
            data=city_counts, x="City", y="Order_ID", ax=ax3, palette="mako"
        )
        a1.bar_label(a1.containers[0])
        ax3.set_title("City-wise Order Count")
        ax3.set_xlabel("City")
        ax3.set_ylabel("Order Count")
        st.pyplot(fig3)
        plt.close(fig3)

    with col_eda4:
        st.markdown("**City-wise Total Revenue**")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        city_revenue = (
            df_clean.groupby("City")["Final_Amount"].sum().reset_index()
        )
        a2 = sns.barplot(
            data=city_revenue, x="City", y="Final_Amount", ax=ax4, palette="rocket"
        )
        a2.bar_label(a2.containers[0], fmt="%d")
        ax4.set_title("City-wise Total Revenue")
        ax4.set_xlabel("City")
        ax4.set_ylabel("Total Revenue")
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("---")

    # 3. Cuisine-wise Order Analysis
    col_eda5, col_eda6 = st.columns(2)

    with col_eda5:
        st.markdown("**Cuisine-wise Order Count**")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        cuisine_counts = (
            df_clean.groupby("Cuisine_Type")["Order_ID"].count().reset_index()
        )
        a3 = sns.barplot(
            data=cuisine_counts,
            x="Cuisine_Type",
            y="Order_ID",
            ax=ax5,
            palette="viridis",
        )
        a3.bar_label(a3.containers[0])
        ax5.set_title("Cuisine-wise Order Count")
        ax5.set_xlabel("Cuisine Type")
        ax5.set_ylabel("Order Count")
        plt.xticks(rotation=30)
        st.pyplot(fig5)
        plt.close(fig5)

    with col_eda6:
        st.markdown("**Cuisine-wise Average Revenue**")
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        cuisine_revenue = (
            df_clean.groupby("Cuisine_Type")["Final_Amount"]
            .mean()
            .reset_index()
        )
        a4 = sns.barplot(
            data=cuisine_revenue,
            x="Cuisine_Type",
            y="Final_Amount",
            ax=ax6,
            palette="magma",
        )
        a4.bar_label(a4.containers[0], fmt="%.2f")
        ax6.set_title("Cuisine-wise Average Revenue")
        ax6.set_xlabel("Cuisine Type")
        ax6.set_ylabel("Average Revenue")
        plt.xticks(rotation=30)
        st.pyplot(fig6)
        plt.close(fig6)

    st.markdown("---")

    # 4. Weekend vs Weekday Demand
    col_eda7, col_eda8 = st.columns(2)

    # Note: If your column is 'Order_Date' or 'Day_Name' instead of 'Order_Day', adjust below:
    day_col = "Order_Day" if "Order_Day" in df_clean.columns else "Day_Name"

    with col_eda7:
        st.markdown("**Weekend Vs Weekday Order Counts**")
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        order_counts = df_clean.groupby(day_col)["Order_ID"].count().reset_index()
        a5 = sns.barplot(
            data=order_counts, x=day_col, y="Order_ID", ax=ax7, palette="crest"
        )
        a5.bar_label(a5.containers[0])
        ax7.set_title("Weekend Vs Weekday Order Counts")
        ax7.set_xlabel("Order Day")
        ax7.set_ylabel("Order Count")
        st.pyplot(fig7)
        plt.close(fig7)

    with col_eda8:
        st.markdown("**Weekend Vs Weekday Average Revenue**")
        fig8, ax8 = plt.subplots(figsize=(6, 4))
        order_revenue = (
            df_clean.groupby(day_col)["Final_Amount"].mean().reset_index()
        )
        a6 = sns.barplot(
            data=order_revenue, x=day_col, y="Final_Amount", ax=ax8, palette="flare"
        )
        a6.bar_label(a6.containers[0], fmt="%.2f")
        ax8.set_title("Weekend Vs Weekday Average Revenue")
        ax8.set_xlabel("Order Day")
        ax8.set_ylabel("Average Revenue")
        st.pyplot(fig8)
        plt.close(fig8)

    st.markdown("---")

    # 5. Distance vs Delivery Delay Relationship
    st.markdown("**Distance Vs Delivery Delay Relationship**")
    fig9, ax9 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df_clean,
        x="Distance_km",
        y="Delivery_Time_Min",
        ax=ax9,
        color="teal",
        alpha=0.6,
    )
    ax9.set_title("Distance Vs Delivery Delay Relationship")
    ax9.set_xlabel("Distance in km")
    ax9.set_ylabel("Delivery Time in Mins")
    st.pyplot(fig9)
    plt.close(fig9)

    st.markdown("---")

    # 6. Cancellation Order Analysis
    st.markdown("**Cancellation Order Analysis**")
    df_cancelled = df_clean[df_clean["Cancellation_Reason"] != "Not Cancelled"]
    cancellation_counts = (
        df_cancelled["Cancellation_Reason"]
        .value_counts()
        .reset_index(name="Count")
    )
    cancellation_counts.rename(
        columns={"Cancellation_Reason": "Reason"}, inplace=True
    )

    total_cancelled = cancellation_counts["Count"].sum()
    if total_cancelled > 0:
        cancellation_counts["Percentage"] = (
            cancellation_counts["Count"] / total_cancelled
        ) * 100

    col_cancel_tbl, col_cancel_chart = st.columns([1, 1.5])

    with col_cancel_tbl:
        st.markdown("**Cancellation Reasons Summary**")
        st.dataframe(cancellation_counts, use_container_width=True)

    with col_cancel_chart:
        fig10, ax10 = plt.subplots(figsize=(7, 4))
        a7 = sns.barplot(
            data=cancellation_counts,
            x="Count",
            y="Reason",
            palette="viridis",
            ax=ax10,
        )
        for container in a7.containers:
            a7.bar_label(container, padding=3, fontsize=10, weight="bold")
        ax10.set_title(
            "Distribution of Order Cancellation Reasons",
            fontsize=12,
            pad=10,
            weight="bold",
        )
        ax10.set_xlabel("Number of Cancelled Orders", fontsize=10)
        ax10.set_ylabel("Cancellation Reason", fontsize=10)
        st.pyplot(fig10)
        plt.close(fig10)

    st.markdown("---")

    # 7. Correlation Analysis
    st.markdown("**Correlation Analysis Among Numerical Features**")
    num_col = df_clean.select_dtypes(include="number").columns.tolist()
    if len(num_col) > 1:
        corr_matrix = df_clean[num_col].corr()
        fig11, ax11 = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            data=corr_matrix,
            annot=True,
            cmap="viridis",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax11,
        )
        ax11.set_title(
            "Correlation Analysis among numerical features",
            fontsize=14,
            pad=15,
            weight="bold",
        )
        st.pyplot(fig11)
        plt.close(fig11)

# ------------------------------------------
# TAB 2: SQL QUERIES EXECUTION
# ------------------------------------------
with tab2:
    st.header("Run Analytical SQL Queries")

    queries = {
        "1. Top Spending Customers": """
            SELECT 
                Customer_ID, 
                COUNT(Order_ID) AS total_orders, 
                ROUND(SUM(Final_Amount), 2) AS total_spent, 
                ROUND(AVG(Final_Amount), 2) AS avg_order_value
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY Customer_ID
            ORDER BY total_spent DESC
            LIMIT 10;
        """,
        "2. Analyze Age Group vs Order Value": """
            SELECT 
                Customer_Age_Group, 
                COUNT(*) AS total_orders, 
                ROUND(SUM(Final_Amount), 2) AS total_spent,
                ROUND(AVG(Final_Amount), 2) AS avg_order_value
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY Customer_Age_Group
            ORDER BY 
            CASE Customer_Age_Group
                WHEN 'Young Adult' THEN 1
                WHEN 'Early Career' THEN 2
                WHEN 'Mid Adult' THEN 3
                WHEN 'Senior Adult' THEN 4
                ELSE 5
            END;
        """,
        "3. Weekend vs Weekday Pattern": """
            SELECT 
                DAYNAME(Order_Date) AS order_day,
                COUNT(*) AS total_orders,
                ROUND(SUM(Final_Amount), 2) AS total_revenue,
                ROUND(AVG(Final_Amount), 2) AS avg_order_value,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS order_percentage
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY DAYNAME(Order_Date)
            ORDER BY total_orders DESC;
        """,
        "4. Monthly Revenue Trends": """
            SELECT 
                DATE_FORMAT(Order_Date, '%Y-%m') AS order_month,
                COUNT(*) AS total_orders,
                ROUND(SUM(Final_Amount), 2) AS monthly_revenue,
                ROUND(AVG(Final_Amount), 2) AS avg_order_value
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY DATE_FORMAT(Order_Date, '%Y-%m')
            ORDER BY order_month ASC;
        """,
        "5. Impact of Discount on Profit": """
            SELECT 
                CASE 
                    WHEN Discount_Applied = 0 THEN '1. No Discount (0)'
                    WHEN Discount_Applied <= 50 THEN '2. Low Discount (1 - 50)'
                    WHEN Discount_Applied <= 150 THEN '3. Moderate Discount (51 - 150)'
                    ELSE '4. High Discount (> 150)'
                END AS discount_tier,    
                COUNT(*) AS total_orders,
                ROUND(SUM(Final_Amount), 2) AS total_revenue,
                ROUND(SUM(Profit_Margin), 2) AS total_profit,
                ROUND(AVG(Profit_Margin), 2) AS avg_profit_per_order,
                ROUND(AVG(Profit_Margin_Percentage), 2) AS avg_profit_percentage
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY 
                CASE 
                    WHEN Discount_Applied = 0 THEN '1. No Discount (0)'
                    WHEN Discount_Applied <= 50 THEN '2. Low Discount (1 - 50)'
                    WHEN Discount_Applied <= 150 THEN '3. Moderate Discount (51 - 150)'
                    ELSE '4. High Discount (> 150)'
                END
            ORDER BY discount_tier ASC;
        """,
        "6. High-Revenue Cities and Cuisines": """
            SELECT 
                City,
                Cuisine_Type,
                COUNT(*) AS total_orders,
                ROUND(SUM(Final_Amount), 2) AS total_revenue,
                ROUND(AVG(Final_Amount), 2) AS avg_order_value,
                ROUND(SUM(Profit_Margin), 2) AS total_profit
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY City, Cuisine_Type
            ORDER BY total_revenue DESC;
        """,
        "7. Average Delivery Time by City": """
            SELECT 
                City,
                COUNT(*) AS total_completed_orders,
                ROUND(AVG(Delivery_Time_Min), 2) AS average_delivery_time
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY City
            ORDER BY average_delivery_time DESC;
        """,
        "8. Distance vs Delivery Delay Analysis": """
            SELECT 
                CASE 
                    WHEN Distance_km <= 3 THEN '1. Short (0 - 3 km)'
                    WHEN Distance_km <= 7 THEN '2. Medium (3.1 - 7 km)'
                    WHEN Distance_km <= 12 THEN '3. Long (7.1 - 12 km)'
                    ELSE '4. Very Long (> 12 km)'
                END AS distance_tier,
                COUNT(*) AS total_orders,
                SUM(CASE WHEN Cancellation_Reason = 'Late Delivery' THEN 1 ELSE 0 END) AS late_delivery_count,
                ROUND(
                    SUM(CASE WHEN Cancellation_Reason = 'Late Delivery' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
                ) AS late_delivery_rate_pct,
                ROUND(AVG(CASE WHEN Order_Status = 'Delivered' THEN Delivery_Time_Min END), 2) AS avg_delivery_time_min
            FROM ofd
            GROUP BY 
                CASE 
                    WHEN Distance_km <= 3 THEN '1. Short (0 - 3 km)'
                    WHEN Distance_km <= 7 THEN '2. Medium (3.1 - 7 km)'
                    WHEN Distance_km <= 12 THEN '3. Long (7.1 - 12 km)'
                    ELSE '4. Very Long (> 12 km)'
                END
            ORDER BY distance_tier ASC;
        """,
        "9. Delivery Rating vs Delivery Time": """
            SELECT 
                Delivery_Rating,
                COUNT(*) AS total_orders,
                ROUND(AVG(Delivery_Time_Min), 2) AS avg_delivery_time_min,
                MIN(Delivery_Time_Min) AS min_delivery_time,
                MAX(Delivery_Time_Min) AS max_delivery_time
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled' AND Delivery_Rating IS NOT NULL
            GROUP BY Delivery_Rating
            ORDER BY Delivery_Rating DESC;
        """,
        "10. Top-rated Restaurants": """
            SELECT 
                Restaurant_ID,
                Restaurant_Name,
                MAX(Restaurant_Rating) AS restaurant_rating,
                COUNT(*) AS total_orders,
                ROUND(SUM(Final_Amount), 2) AS total_revenue
            FROM ofd
            WHERE Cancellation_Reason = 'Not Cancelled'
            GROUP BY Restaurant_ID, Restaurant_Name
            ORDER BY restaurant_rating DESC, total_orders DESC
            LIMIT 10;
        """,
        "11. Cancellation Rate by Restaurants": """
            SELECT 
                Restaurant_ID, 
                Restaurant_Name, 
                COUNT(*) AS total_orders,
                SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
                SUM(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN 1 ELSE 0 END) AS delivered_orders,
                ROUND(
                    SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
                ) AS cancellation_rate_pct
            FROM ofd
            GROUP BY Restaurant_ID, Restaurant_Name
            HAVING COUNT(*) >= 1
            ORDER BY cancellation_rate_pct DESC;
        """,
        "12. Cuisine-wise Performance": """
            SELECT 
                Cuisine_Type, 
                COUNT(*) AS total_orders,    
                ROUND(SUM(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Final_Amount ELSE 0 END), 2) AS total_revenue,
                ROUND(AVG(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Final_Amount END), 2) AS avg_order_value,    
                ROUND(AVG(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Restaurant_Rating END), 2) AS avg_restaurant_rating,
                ROUND(AVG(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Delivery_Time_Min END), 2) AS avg_delivery_time_min,    
                SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
                ROUND(
                    SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
                ) AS cancellation_rate_pct
            FROM ofd
            GROUP BY Cuisine_Type
            ORDER BY total_revenue DESC;
        """,
        "13. Peak Hour Demand Analysis": """
            SELECT 
                CASE 
                    WHEN Peak_Hour = '1' OR Peak_Hour = 1 THEN 'Peak Hours'
                    WHEN Peak_Hour = '0' OR Peak_Hour = 0 THEN 'Non-Peak Hours'
                    ELSE 'Unknown / Unclassified'
                END AS peak_hour_status,
                COUNT(*) AS total_orders,    
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS order_share_pct,    
                ROUND(SUM(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Final_Amount ELSE 0 END), 2) AS total_revenue,
                ROUND(AVG(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Final_Amount END), 2) AS avg_order_value,    
                ROUND(AVG(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Delivery_Time_Min END), 2) AS avg_delivery_time_min,    
                SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) AS total_cancelled_orders,
                ROUND(
                    SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
                ) AS cancellation_rate_pct
            FROM ofd
            GROUP BY 
                CASE 
                    WHEN Peak_Hour = '1' OR Peak_Hour = 1 THEN 'Peak Hours'
                    WHEN Peak_Hour = '0' OR Peak_Hour = 0 THEN 'Non-Peak Hours'
                    ELSE 'Unknown / Unclassified'
                END
            ORDER BY total_orders DESC;
        """,
        "14. Payment Mode Preferences": """
            SELECT 
                Payment_Mode,
                COUNT(*) AS total_orders,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS payment_share_pct,
                ROUND(SUM(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Final_Amount ELSE 0 END), 2) AS total_revenue,
                ROUND(AVG(CASE WHEN Cancellation_Reason = 'Not Cancelled' THEN Final_Amount END), 2) AS avg_order_value,
                SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,
                ROUND(
                    SUM(CASE WHEN Cancellation_Reason != 'Not Cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
                ) AS cancellation_rate_pct
            FROM ofd
            WHERE Payment_Mode IS NOT NULL
            GROUP BY Payment_Mode
            ORDER BY total_orders DESC;
        """,
        "15. Cancellation Reason Analysis": """
            SELECT 
                Cancellation_Reason,
                COUNT(*) AS total_orders,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS share_of_total_orders_pct,
                ROUND(SUM(Final_Amount), 2) AS lost_revenue_impact,
                ROUND(AVG(Final_Amount), 2) AS avg_cancelled_order_value
            FROM ofd
            WHERE Cancellation_Reason IS NOT NULL
            GROUP BY Cancellation_Reason
            ORDER BY total_orders DESC;
        """,
    }

    task = st.selectbox("Choose Problem Statement", list(queries.keys()))

    if st.button("Run SQL Query", type="primary"):
        selected_query = queries[task]

        with st.spinner("Querying MySQL database..."):
            try:
                with engine.connect() as conn:
                    result_df = pd.read_sql(selected_query, conn)

                st.subheader(f"Results: {task}")

                # Display Metrics
                m1, m2 = st.columns(2)
                m1.metric("Rows Returned", len(result_df))
                m2.metric("Columns", len(result_df.columns))

                # Display Resulting Table
                st.dataframe(result_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error executing query: {e}")
