import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from sklearn.linear_model import LinearRegression
from io import BytesIO

st.set_page_config(layout="wide", page_title="Hypothesis Testing in Regression")

st.title("Interactive Hypothesis Testing in Regression")
st.markdown("""
This app demonstrates key concepts in hypothesis testing for regression analysis.
Explore test statistics, distributions, rejection regions, and p-values for different scenarios.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a section:",
                        ["Introduction",
                         "T-Distribution & Tests",
                         "F-Distribution & Tests",
                         "Interactive Regression Example",
                         "P-value Visualization",
                         "Multiple Hypothesis Tests",
                         "Distribution Properties"])

# Introduction page
if page == "Introduction":
    st.header("Introduction to Hypothesis Testing in Regression")

    st.markdown("""
    ### Key Concepts:
    - **Null Hypothesis (H₀)**: Typically assumes no effect or relationship (e.g., β = 0)
    - **Alternative Hypothesis (H₁)**: Contradicts the null (e.g., β ≠ 0, β > 0, or β < 0)
    - **Test Statistic**: Value calculated from sample data used to make decisions
    - **Critical Value**: Threshold for rejecting H₀
    - **Rejection Region**: Area where we reject H₀
    - **p-value**: Probability of observing a test statistic as extreme or more extreme than observed, assuming H₀ is true
    """)

    st.markdown("### Types of Tests in Regression:")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **t-test**:
        - Tests significance of individual coefficients
        - H₀: βⱼ = 0 (no effect)
        - H₁: βⱼ ≠ 0 (two-tailed)
        - H₁: βⱼ > 0 (right-tailed)
        - H₁: βⱼ < 0 (left-tailed)
        """)

    with col2:
        st.markdown("""
        **F-test**:
        - Tests overall significance of the model
        - H₀: β₁ = β₂ = ... = βₖ = 0
        - H₁: At least one βⱼ ≠ 0
        - Also used for testing groups of coefficients
        """)

    st.subheader("Decision Rules")
    st.markdown("""
    We reject H₀ if:
    1. The test statistic falls in the rejection region, OR
    2. The p-value is less than the significance level (α)
    """)

# T-Distribution & Tests page
elif page == "T-Distribution & Tests":
    st.header("T-Distribution and t-tests")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        df = st.slider("Degrees of Freedom", 1, 100, 10)
    with col2:
        alpha = st.slider("Significance Level (α)", 0.01, 0.20, 0.05)
    with col3:
        test_type = st.radio("Test Type", ["Two-tailed", "Right-tailed", "Left-tailed"])

    # Generate t-distribution
    x = np.linspace(-5, 5, 1000)
    y = stats.t.pdf(x, df)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot t-distribution
    ax.plot(x, y, 'b-', lw=2, label=f't-distribution (df={df})')
    ax.set_xlabel('t-value')
    ax.set_ylabel('Probability Density')

    # Plot rejection regions
    if test_type == "Two-tailed":
        t_crit_lower = stats.t.ppf(alpha / 2, df)
        t_crit_upper = stats.t.ppf(1 - alpha / 2, df)

        # Shade rejection regions
        ax.fill_between(x, 0, y, where=(x <= t_crit_lower), color='red', alpha=0.3)
        ax.fill_between(x, 0, y, where=(x >= t_crit_upper), color='red', alpha=0.3)

        # Add critical values
        ax.axvline(t_crit_lower, color='red', linestyle='--', label=f'Critical values: ±{abs(t_crit_lower):.3f}')
        ax.axvline(t_crit_upper, color='red', linestyle='--')

        # Set title
        ax.set_title(f"Two-tailed t-test (α={alpha})")

    elif test_type == "Right-tailed":
        t_crit = stats.t.ppf(1 - alpha, df)

        # Shade rejection region
        ax.fill_between(x, 0, y, where=(x >= t_crit), color='red', alpha=0.3)

        # Add critical value
        ax.axvline(t_crit, color='red', linestyle='--', label=f'Critical value: {t_crit:.3f}')

        # Set title
        ax.set_title(f"Right-tailed t-test (α={alpha})")

    else:  # Left-tailed
        t_crit = stats.t.ppf(alpha, df)

        # Shade rejection region
        ax.fill_between(x, 0, y, where=(x <= t_crit), color='red', alpha=0.3)

        # Add critical value
        ax.axvline(t_crit, color='red', linestyle='--', label=f'Critical value: {t_crit:.3f}')

        # Set title
        ax.set_title(f"Left-tailed t-test (α={alpha})")

    # Add legend
    ax.legend()

    # Show plot
    st.pyplot(fig)

    # Explanation
    st.subheader("Interactive Exploration:")
    st.markdown("""
    Experiment with the controls to see how the t-distribution and rejection regions change.

    **Key Points:**
    - The red shaded areas represent rejection regions
    - If your test statistic falls in the red area, you reject H₀
    - The t-distribution approaches the normal distribution as degrees of freedom increase
    - Higher α means larger rejection regions (more likely to reject H₀)
    """)

    # Example of t-test in regression
    st.subheader("Example: t-test for Regression Coefficient")
    st.markdown("""
    In regression, the t-statistic for a coefficient βⱼ is:

    $$t = \\frac{\\hat{\\beta}_j - \\beta_{j0}}{SE(\\hat{\\beta}_j)}$$

    where:
    - $\\hat{\\beta}_j$ is the estimated coefficient
    - $\\beta_{j0}$ is the hypothesized value (usually 0)
    - $SE(\\hat{\\beta}_j)$ is the standard error of the coefficient

    This t-statistic follows a t-distribution with (n-k-1) degrees of freedom, where:
    - n is the sample size
    - k is the number of predictors
    """)

    # Let the user input a test statistic
    st.subheader("Calculate p-value")
    t_stat = st.number_input("Enter a t-statistic value:", value=2.0, step=0.1)

    if test_type == "Two-tailed":
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if t_stat != 0 else 1.0
        st.write(f"p-value for t = {t_stat} (two-tailed): {p_value:.4f}")
        if p_value < alpha:
            st.write("Decision: Reject H₀")
        else:
            st.write("Decision: Fail to reject H₀")
    elif test_type == "Right-tailed":
        p_value = 1 - stats.t.cdf(t_stat, df)
        st.write(f"p-value for t = {t_stat} (right-tailed): {p_value:.4f}")
        if p_value < alpha:
            st.write("Decision: Reject H₀")
        else:
            st.write("Decision: Fail to reject H₀")
    else:  # Left-tailed
        p_value = stats.t.cdf(t_stat, df)
        st.write(f"p-value for t = {t_stat} (left-tailed): {p_value:.4f}")
        if p_value < alpha:
            st.write("Decision: Reject H₀")
        else:
            st.write("Decision: Fail to reject H₀")

# F-Distribution & Tests page
elif page == "F-Distribution & Tests":
    st.header("F-Distribution and F-tests")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        df1 = st.slider("Numerator Degrees of Freedom", 1, 20, 3)
    with col2:
        df2 = st.slider("Denominator Degrees of Freedom", 1, 100, 30)
    with col3:
        alpha_f = st.slider("Significance Level (α)", 0.01, 0.20, 0.05, key="alpha_f")

    # Generate F-distribution
    x = np.linspace(0, 8, 1000)
    y = stats.f.pdf(x, df1, df2)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot F-distribution
    ax.plot(x, y, 'b-', lw=2, label=f'F-distribution (df1={df1}, df2={df2})')
    ax.set_xlabel('F-value')
    ax.set_ylabel('Probability Density')

    # Calculate critical value and shade rejection region
    f_crit = stats.f.ppf(1 - alpha_f, df1, df2)
    ax.fill_between(x, 0, y, where=(x >= f_crit), color='red', alpha=0.3)
    ax.axvline(f_crit, color='red', linestyle='--', label=f'Critical value: {f_crit:.3f}')

    # Set title and add legend
    ax.set_title(f"F-test (α={alpha_f})")
    ax.legend()

    # Show plot
    st.pyplot(fig)

    # Explanation
    st.subheader("F-Distribution Properties")
    st.markdown("""
    **Key Properties:**
    - Always right-skewed
    - Always non-negative
    - Shape determined by two parameters: df1 and df2
    - As df1 and df2 increase, the F-distribution approaches normal

    **F-tests in Regression:**
    1. **Overall Model Significance:**

       H₀: All slope coefficients are zero (model has no explanatory power)

       H₁: At least one coefficient is non-zero (model has some explanatory power)

       The F-statistic is calculated as:

       $$F = \\frac{R^2/k}{(1-R^2)/(n-k-1)}$$

       where:
       - R² is the coefficient of determination
       - k is the number of predictors
       - n is the sample size

    2. **Testing Coefficient Groups:**
       Tests whether a subset of variables significantly improves the model
    """)

    # Let the user input an F-statistic
    st.subheader("Calculate p-value")
    f_stat = st.number_input("Enter an F-statistic value:", value=3.0, step=0.1)

    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    st.write(f"p-value for F = {f_stat}: {p_value:.4f}")
    if p_value < alpha_f:
        st.write("Decision: Reject H₀")
    else:
        st.write("Decision: Fail to reject H₀")

# Interactive Regression Example page
elif page == "Interactive Regression Example":
    st.header("Interactive Regression Example")

    # Option to use sample data or upload own
    data_option = st.radio("Choose data source:", ["Use sample data", "Enter your own data"])

    if data_option == "Use sample data":
        # Generate sample data
        np.random.seed(42)
        n = 50
        x = np.linspace(0, 10, n)
        noise = np.random.normal(0, 2, n)

        # Let the user adjust the coefficient to see different scenarios
        true_beta = st.slider("Set true coefficient (β):", -2.0, 2.0, 0.8)
        y = 2 + true_beta * x + noise

        data = pd.DataFrame({'X': x, 'Y': y})
    else:
        # Option to enter data manually or upload CSV
        input_method = st.radio("Input method:", ["Enter manually", "Upload CSV"])

        if input_method == "Enter manually":
            data_input = st.text_area("Enter X and Y data (comma-separated pairs, one per line):",
                                      "1,3\n2,5\n3,4\n4,8\n5,7")
            try:
                lines = data_input.strip().split('\n')
                x_data = []
                y_data = []
                for line in lines:
                    x_val, y_val = line.split(',')
                    x_data.append(float(x_val))
                    y_data.append(float(y_val))
                data = pd.DataFrame({'X': x_data, 'Y': y_data})
            except:
                st.error("Error parsing data. Please ensure it's in the correct format.")
                data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'Y': [3, 5, 4, 8, 7]})
        else:
            uploaded_file = st.file_uploader("Upload CSV file with columns named 'X' and 'Y'", type=['csv'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    if 'X' not in data.columns or 'Y' not in data.columns:
                        st.error("CSV must have columns named 'X' and 'Y'")
                        data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'Y': [3, 5, 4, 8, 7]})
                except:
                    st.error("Error reading CSV file")
                    data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'Y': [3, 5, 4, 8, 7]})
            else:
                data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'Y': [3, 5, 4, 8, 7]})

    # Display the first few rows of data
    st.subheader("Data Preview")
    st.write(data.head())

    # Fit regression model
    X = data['X'].values.reshape(-1, 1)
    y = data['Y'].values
    model = LinearRegression()
    model.fit(X, y)

    # Calculate statistics
    n = len(data)
    k = 1  # number of predictors
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Calculate SSR, SSE, and SST
    y_mean = np.mean(y)
    SSR = np.sum((y_pred - y_mean) ** 2)
    SSE = np.sum(residuals ** 2)
    SST = np.sum((y - y_mean) ** 2)

    # Calculate R-squared and adjusted R-squared
    r_squared = SSR / SST
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    # Calculate standard error of regression (SER)
    SER = np.sqrt(SSE / (n - k - 1))

    # Calculate standard error of coefficient
    X_mean = np.mean(X)
    SE_beta = SER / np.sqrt(np.sum((X - X_mean) ** 2))

    # Calculate t-statistic for slope coefficient
    t_stat = model.coef_[0] / SE_beta

    # Calculate p-value for t-statistic
    p_value_t = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))

    # Calculate F-statistic for overall model
    F_stat = (SSR / k) / (SSE / (n - k - 1))

    # Calculate p-value for F-statistic
    p_value_F = 1 - stats.f.cdf(F_stat, k, n - k - 1)

    # Set significance level
    alpha = st.slider("Significance Level (α):", 0.01, 0.20, 0.05, key="alpha_reg")

    # Plot the data and regression line
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(X, y, color='blue', alpha=0.5, label='Data points')
    ax1.plot(X, y_pred, color='red', label=f'Regression line: Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}X')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Regression Line')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    st.pyplot(fig1)

    # Display regression results
    st.subheader("Regression Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Coefficient Estimates:**")
        st.write(f"Intercept (β₀): {model.intercept_:.4f}")
        st.write(f"Slope (β₁): {model.coef_[0]:.4f}")
        st.write(f"Standard Error of β₁: {SE_beta:.4f}")

        st.markdown("**Coefficient Hypothesis Test:**")
        st.write(f"H₀: β₁ = 0")
        st.write(f"H₁: β₁ ≠ 0")
        st.write(f"t-statistic: {t_stat:.4f}")
        st.write(f"p-value: {p_value_t:.4f}")

        if p_value_t < alpha:
            st.markdown("**Decision:** Reject H₀ (Coefficient is statistically significant)")
        else:
            st.markdown("**Decision:** Fail to reject H₀ (Insufficient evidence of significance)")

    with col2:
        st.markdown("**Model Statistics:**")
        st.write(f"R²: {r_squared:.4f}")
        st.write(f"Adjusted R²: {adj_r_squared:.4f}")
        st.write(f"Standard Error of Regression: {SER:.4f}")

        st.markdown("**Overall Model Hypothesis Test:**")
        st.write(f"H₀: β₁ = 0 (Model has no explanatory power)")
        st.write(f"H₁: β₁ ≠ 0 (Model has explanatory power)")
        st.write(f"F-statistic: {F_stat:.4f}")
        st.write(f"p-value: {p_value_F:.4f}")

        if p_value_F < alpha:
            st.markdown("**Decision:** Reject H₀ (Model is statistically significant)")
        else:
            st.markdown("**Decision:** Fail to reject H₀ (Insufficient evidence of model significance)")

    # Visualization of t-distribution and test
    st.subheader("Visualizing the t-test for the Slope Coefficient")

    # Generate t-distribution
    x_t = np.linspace(-4, 4, 1000)
    y_t = stats.t.pdf(x_t, n - k - 1)

    # Create figure
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Plot t-distribution
    ax2.plot(x_t, y_t, 'b-', lw=2, label=f't-distribution (df={n - k - 1})')
    ax2.set_xlabel('t-value')
    ax2.set_ylabel('Probability Density')

    # Plot critical values and test statistic
    t_crit_lower = stats.t.ppf(alpha / 2, n - k - 1)
    t_crit_upper = stats.t.ppf(1 - alpha / 2, n - k - 1)

    # Shade rejection regions
    ax2.fill_between(x_t, 0, y_t, where=(x_t <= t_crit_lower), color='red', alpha=0.3, label='Rejection region')
    ax2.fill_between(x_t, 0, y_t, where=(x_t >= t_crit_upper), color='red', alpha=0.3)

    # Add critical values
    ax2.axvline(t_crit_lower, color='red', linestyle='--', label=f'Critical values: ±{abs(t_crit_lower):.3f}')
    ax2.axvline(t_crit_upper, color='red', linestyle='--')

    # Add test statistic
    ax2.axvline(t_stat, color='green', linestyle='-', linewidth=2, label=f'Test statistic: {t_stat:.3f}')

    # Highlight p-value
    if abs(t_stat) > abs(t_crit_lower):
        # Calculate where to start shading for p-value
        if t_stat > 0:
            x_pval = np.linspace(t_stat, 4, 200)
            y_pval = stats.t.pdf(x_pval, n - k - 1)
            ax2.fill_between(x_pval, 0, y_pval, color='green', alpha=0.2)

            x_pval_neg = np.linspace(-4, -t_stat, 200)
            y_pval_neg = stats.t.pdf(x_pval_neg, n - k - 1)
            ax2.fill_between(x_pval_neg, 0, y_pval_neg, color='green', alpha=0.2, label='p-value area')
        else:
            x_pval = np.linspace(-4, t_stat, 200)
            y_pval = stats.t.pdf(x_pval, n - k - 1)
            ax2.fill_between(x_pval, 0, y_pval, color='green', alpha=0.2)

            x_pval_pos = np.linspace(-t_stat, 4, 200)
            y_pval_pos = stats.t.pdf(x_pval_pos, n - k - 1)
            ax2.fill_between(x_pval_pos, 0, y_pval_pos, color='green', alpha=0.2, label='p-value area')

    # Set title and add legend
    ax2.set_title(f"Two-tailed t-test for Slope Coefficient (α={alpha})")
    ax2.legend()

    # Show plot
    st.pyplot(fig2)

# P-value Visualization page
elif page == "P-value Visualization":
    st.header("P-value Visualization")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        dist_type = st.radio("Distribution:", ["t-distribution", "F-distribution"])
    with col2:
        if dist_type == "t-distribution":
            df_t = st.slider("Degrees of Freedom:", 1, 100, 20, key="df_t_pval")
            test_type_pval = st.radio("Test Type:", ["Two-tailed", "Right-tailed", "Left-tailed"])
        else:
            df1_pval = st.slider("Numerator df:", 1, 20, 3, key="df1_pval")
            df2_pval = st.slider("Denominator df:", 1, 100, 30, key="df2_pval")

    # Allow user to set test statistic
    if dist_type == "t-distribution":
        test_stat = st.slider("Test Statistic (t):", -5.0, 5.0, 2.0, 0.1)
    else:
        test_stat = st.slider("Test Statistic (F):", 0.1, 10.0, 3.0, 0.1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    if dist_type == "t-distribution":
        # Generate t-distribution
        x = np.linspace(-5, 5, 1000)
        y = stats.t.pdf(x, df_t)

        # Plot t-distribution
        ax.plot(x, y, 'b-', lw=2, label=f't-distribution (df={df_t})')
        ax.set_xlabel('t-value')

        # Calculate p-value and shade area
        if test_type_pval == "Two-tailed":
            # Calculate p-value
            p_value = 2 * (1 - stats.t.cdf(abs(test_stat), df_t))

            # Shade p-value area
            if test_stat > 0:
                # Shade right tail
                x_pval_right = np.linspace(test_stat, 5, 200)
                y_pval_right = stats.t.pdf(x_pval_right, df_t)
                ax.fill_between(x_pval_right, 0, y_pval_right, color='green', alpha=0.4)

                # Shade left tail
                x_pval_left = np.linspace(-5, -test_stat, 200)
                y_pval_left = stats.t.pdf(x_pval_left, df_t)
                ax.fill_between(x_pval_left, 0, y_pval_left, color='green', alpha=0.4, label=f'p-value = {p_value:.4f}')
            else:
                # Shade left tail
                x_pval_left = np.linspace(-5, test_stat, 200)
                y_pval_left = stats.t.pdf(x_pval_left, df_t)
                ax.fill_between(x_pval_left, 0, y_pval_left, color='green', alpha=0.4)

                # Shade right tail
                x_pval_right = np.linspace(-test_stat, 5, 200)
                y_pval_right = stats.t.pdf(x_pval_right, df_t)
                ax.fill_between(x_pval_right, 0, y_pval_right, color='green', alpha=0.4,
                                label=f'p-value = {p_value:.4f}')

            ax.axvline(test_stat, color='red', linestyle='-', linewidth=2, label=f'Test statistic: {test_stat:.2f}')
            ax.axvline(-test_stat, color='red', linestyle='--', linewidth=1)

            ax.set_title(f"Two-tailed t-test (p-value = {p_value:.4f})")

        elif test_type_pval == "Right-tailed":
            # Calculate p-value
            p_value = 1 - stats.t.cdf(test_stat, df_t)

            # Shade p-value area
            x_pval = np.linspace(test_stat, 5, 200)
            y_pval = stats.t.pdf(x_pval, df_t)
            ax.fill_between(x_pval, 0, y_pval, color='green', alpha=0.4, label=f'p-value = {p_value:.4f}')

            ax.axvline(test_stat, color='red', linestyle='-', linewidth=2, label=f'Test statistic: {test_stat:.2f}')

            ax.set_title(f"Right-tailed t-test (p-value = {p_value:.4f})")

        else:  # Left-tailed
            # Calculate p-value
            p_value = stats.t.cdf(test_stat, df_t)

            # Shade p-value area
            x_pval = np.linspace(-5, test_stat, 200)
            y_pval = stats.t.pdf(x_pval, df_t)
            ax.fill_between(x_pval, 0, y_pval, color='green', alpha=0.4, label=f'p-value = {p_value:.4f}')

            ax.axvline(test_stat, color='red', linestyle='-', linewidth=2, label=f'Test statistic: {test_stat:.2f}')

            ax.set_title(f"Left-tailed t-test (p-value = {p_value:.4f})")

    else:  # F-distribution
        # Generate F-distribution
        x = np.linspace(0, 10, 1000)
        y = stats.f.pdf(x, df1_pval, df2_pval)

        # Plot F-distribution
        ax.plot(x, y, 'b-', lw=2, label=f'F-distribution (df1={df1_pval}, df2={df2_pval})')
        ax.set_xlabel('F-value')

        # Calculate p-value and shade area
        p_value = 1 - stats.f.cdf(test_stat, df1_pval, df2_pval)

        # Shade p-value area
        x_pval = np.linspace(test_stat, 10, 200)
        y_pval = stats.f.pdf(x_pval, df1_pval, df2_pval)
        ax.fill_between(x_pval, 0, y_pval, color='green', alpha=0.4, label=f'p-value = {p_value:.4f}')

        ax.axvline(test_stat, color='red', linestyle='-', linewidth=2, label=f'Test statistic: {test_stat:.2f}')

        ax.set_title(f"F-test (p-value = {p_value:.4f})")

    ax.set_ylabel('Probability Density')
    ax.legend()

    # Show plot
    st.pyplot(fig)

    # Explanation
    st.subheader("Understanding P-values")
    st.markdown("""
    **Definition:**
    The p-value is the probability of observing a test statistic as extreme or more extreme than the one calculated from your sample data, assuming the null hypothesis is true.

    **Interpretation:**
    - A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis
    - A large p-value (> 0.05) indicates weak evidence against the null hypothesis
    - p-values indicate the strength of evidence, not the size of the effect

    **Decision Rule:**
    If p-value ≤ α, reject H₀
    If p-value > α, fail to reject H₀

    **Remember:**
    - The p-value is not the probability that the null hypothesis is true
    - A low p-value does not guarantee practical significance
    - Multiple testing increases the chance of false positives
    """)

    # Add significance level selector
    alpha_pval = st.slider("Set significance level (α):", 0.01, 0.20, 0.05, key="alpha_pval_page")

    # Decision based on p-value
    st.subheader("Hypothesis Test Decision")
    if p_value <= alpha_pval:
        st.markdown(f"**Decision:** Reject H₀ (p-value = {p_value:.4f} ≤ α = {alpha_pval})")
    else:
        st.markdown(f"**Decision:** Fail to reject H₀ (p-value = {p_value:.4f} > α = {alpha_pval})")

# Multiple Hypothesis Tests page
elif page == "Multiple Hypothesis Tests":
    st.header("Multiple Hypothesis Tests in Regression")

    st.markdown("""
    In multiple regression, we often perform several types of hypothesis tests:

    1. **Tests for Individual Coefficients** (t-tests)
    2. **Test for Overall Model Significance** (F-test)
    3. **Tests for Groups of Coefficients** (Partial F-tests)
    4. **Tests for Model Specification** (e.g., nested models)
    """)

    # Interactive example with synthetic data
    st.subheader("Interactive Multiple Regression Example")

    # Generate synthetic data
    np.random.seed(123)
    n_samples = 100

    # Let users control which predictors have real effects
    st.write("Set the true coefficients for each predictor:")
    col1, col2, col3 = st.columns(3)

    with col1:
        beta1 = st.slider("Effect of X₁:", -2.0, 2.0, 0.8)
    with col2:
        beta2 = st.slider("Effect of X₂:", -2.0, 2.0, -0.3)
    with col3:
        beta3 = st.slider("Effect of X₃:", -2.0, 2.0, 0.0)

    # Add noise level control
    noise_level = st.slider("Noise Level:", 0.5, 5.0, 2.0)

    # Generate predictors
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)

    # Generate response
    intercept = 5
    Y = intercept + beta1 * X1 + beta2 * X2 + beta3 * X3 + np.random.normal(0, noise_level, n_samples)

    # Create dataframe
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'Y': Y
    })

    # Display data preview
    st.write("Data Preview:")
    st.write(data.head())

    # Fit multiple regression model
    X = data[['X1', 'X2', 'X3']]
    X = np.column_stack([np.ones(n_samples), X])  # Add intercept column
    y = data['Y']

    # Manually compute coefficients
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    # Compute predicted values and residuals
    y_pred = X @ beta_hat
    residuals = y - y_pred

    # Compute SSR, SSE, and SST
    y_mean = np.mean(y)
    SSR = np.sum((y_pred - y_mean) ** 2)
    SSE = np.sum(residuals ** 2)
    SST = np.sum((y - y_mean) ** 2)

    # Calculate R-squared and adjusted R-squared
    r_squared = SSR / SST
    k = 3  # number of predictors
    adj_r_squared = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - k - 1)

    # Calculate standard error of regression (SER)
    SER = np.sqrt(SSE / (n_samples - k - 1))

    # Calculate variance-covariance matrix of coefficients
    cov_beta = SER ** 2 * np.linalg.inv(X.T @ X)

    # Calculate standard errors for coefficients
    se_beta = np.sqrt(np.diag(cov_beta))

    # Calculate t-statistics
    t_stats = beta_hat / se_beta

    # Calculate p-values
    p_values = [2 * (1 - stats.t.cdf(abs(t), n_samples - k - 1)) for t in t_stats]

    # Calculate F-statistic for overall model
    F_stat = (SSR / k) / (SSE / (n_samples - k - 1))

    # Calculate p-value for F-statistic
    p_value_F = 1 - stats.f.cdf(F_stat, k, n_samples - k - 1)

    # Set significance level
    alpha_mult = st.slider("Significance Level (α):", 0.01, 0.20, 0.05, key="alpha_mult")

    # Display regression results
    st.subheader("Multiple Regression Results")

    # Create a table of coefficients
    coef_data = {
        'Coefficient': ['Intercept', 'X₁', 'X₂', 'X₃'],
        'Estimate': beta_hat,
        'Std. Error': se_beta,
        't-statistic': t_stats,
        'p-value': p_values,
        'Significant': [p < alpha_mult for p in p_values]
    }

    coef_df = pd.DataFrame(coef_data)
    coef_df['Estimate'] = coef_df['Estimate'].round(4)
    coef_df['Std. Error'] = coef_df['Std. Error'].round(4)
    coef_df['t-statistic'] = coef_df['t-statistic'].round(4)
    coef_df['p-value'] = coef_df['p-value'].round(4)

    st.write(coef_df)

    # Model summary statistics
    st.subheader("Model Summary")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"R²: {r_squared:.4f}")
        st.write(f"Adjusted R²: {adj_r_squared:.4f}")
        st.write(f"Standard Error of Regression: {SER:.4f}")

    with col2:
        st.write(f"F-statistic: {F_stat:.4f}")
        st.write(f"p-value (F): {p_value_F:.4f}")
        if p_value_F < alpha_mult:
            st.write("Overall model is statistically significant")
        else:
            st.write("Overall model is not statistically significant")

    # Test for groups of coefficients
    st.subheader("Test for Groups of Coefficients")
    st.markdown("""
    Let's test if X₂ and X₃ jointly have a significant effect on Y.

    H₀: β₂ = β₃ = 0

    H₁: At least one of β₂ or β₃ is not equal to 0
    """)

    # Fit the restricted model (only X1)
    X_restricted = np.column_stack([np.ones(n_samples), X1])
    beta_restricted = np.linalg.inv(X_restricted.T @ X_restricted) @ X_restricted.T @ y
    y_pred_restricted = X_restricted @ beta_restricted
    residuals_restricted = y - y_pred_restricted
    SSE_restricted = np.sum(residuals_restricted ** 2)

    # Calculate partial F-statistic
    q = 2  # number of restrictions
    F_partial = ((SSE_restricted - SSE) / q) / (SSE / (n_samples - k - 1))

    # Calculate p-value for partial F-statistic
    p_value_partial = 1 - stats.f.cdf(F_partial, q, n_samples - k - 1)

    st.write(f"Partial F-statistic: {F_partial:.4f}")
    st.write(f"p-value: {p_value_partial:.4f}")

    if p_value_partial < alpha_mult:
        st.write("Decision: Reject H₀ (X₂ and X₃ are jointly significant)")
    else:
        st.write("Decision: Fail to reject H₀ (X₂ and X₃ are not jointly significant)")

    # Visualize the joint distribution of coefficients
    st.subheader("Visualizing Joint Confidence Region")

    # Extract covariance submatrix for X2 and X3
    cov_23 = cov_beta[2:4, 2:4]

    # Generate grid of values
    b2_grid = np.linspace(beta_hat[2] - 3 * se_beta[2], beta_hat[2] + 3 * se_beta[2], 100)
    b3_grid = np.linspace(beta_hat[3] - 3 * se_beta[3], beta_hat[3] + 3 * se_beta[3], 100)
    B2, B3 = np.meshgrid(b2_grid, b3_grid)

    # Calculate multivariate normal density
    Z = np.zeros_like(B2)
    for i in range(len(b2_grid)):
        for j in range(len(b3_grid)):
            beta_diff = np.array([B2[i, j] - beta_hat[2], B3[i, j] - beta_hat[3]])
            Z[i, j] = np.exp(-0.5 * beta_diff.T @ np.linalg.inv(cov_23) @ beta_diff)

    # Plot joint distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(B2, B3, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax)

    # Add true values
    ax.scatter(beta2, beta3, color='red', s=100, marker='*', label='True values')

    # Add estimated values
    ax.scatter(beta_hat[2], beta_hat[3], color='blue', s=100, marker='o', label='Estimates')

    # Draw 95% confidence ellipse
    from matplotlib.patches import Ellipse

    # Calculate eigenvalues and eigenvectors of covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov_23)

    # Calculate angles and width/height of ellipse
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * 1.96 * np.sqrt(eigenvals[0])  # 95% confidence interval
    height = 2 * 1.96 * np.sqrt(eigenvals[1])  # 95% confidence interval

    # Create and add ellipse
    ellipse = Ellipse(xy=(beta_hat[2], beta_hat[3]), width=width, height=height,
                      angle=angle, edgecolor='black', fc='none', lw=2, label='95% Confidence Region')
    ax.add_patch(ellipse)

    # Set labels and title
    ax.set_xlabel('Coefficient of X₂')
    ax.set_ylabel('Coefficient of X₃')
    ax.set_title('Joint Distribution of X₂ and X₃ Coefficients')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    st.pyplot(fig)

    st.markdown("""
    **Key Points for Multiple Hypothesis Tests:**

    1. **Individual t-tests** examine whether each predictor has a significant effect on the response, holding all other predictors constant

    2. **The overall F-test** examines whether the model as a whole has explanatory power

    3. **Partial F-tests** examine whether a subset of predictors jointly has a significant effect on the response

    4. **Joint confidence regions** show the uncertainty in estimating multiple parameters simultaneously

    5. **Multiple testing issues:**
       - When conducting many tests, the chance of Type I errors increases
       - Solutions include Bonferroni correction, False Discovery Rate control, etc.
    """)

# Distribution Properties page
elif page == "Distribution Properties":
    st.header("Properties of t and F Distributions")

    st.subheader("t-Distribution Properties")

    # Interactive visualization of how t-distribution changes with df
    df_values = [1, 5, 10, 30, 100]
    x = np.linspace(-5, 5, 1000)

    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for df in df_values:
        y = stats.t.pdf(x, df)
        ax1.plot(x, y, label=f'df = {df}')

    # Add standard normal for comparison
    y_norm = stats.norm.pdf(x, 0, 1)
    ax1.plot(x, y_norm, 'k--', label='Standard Normal')

    ax1.set_xlabel('t-value')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('t-Distribution for Different Degrees of Freedom')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    st.pyplot(fig1)

    st.markdown("""
    **Key Properties of the t-Distribution:**

    1. **Shape:**
       - Symmetric around 0 (just like the normal distribution)
       - Heavier tails than the normal distribution
       - Approaches the standard normal distribution as degrees of freedom increase

    2. **Degrees of Freedom:**
       - In regression, df = n-k-1 (sample size minus number of predictors minus 1)
       - Lower df means fatter tails (more uncertainty)

    3. **Use in Regression:**
       - Used for testing individual regression coefficients
       - Used for calculating confidence intervals for coefficients
       - Accounts for estimating variance from sample data
    """)

    # Interactive exploration of one vs two-tailed tests
    st.subheader("One-tailed vs. Two-tailed t-tests")

    col1, col2 = st.columns(2)
    with col1:
        df_tails = st.slider("Degrees of Freedom:", 1, 50, 20)
    with col2:
        alpha_tails = st.slider("Significance Level (α):", 0.01, 0.20, 0.05, key="alpha_tails")

    # Create figure for tail comparison
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(15, 5))

    # Generate t-distribution
    x_tails = np.linspace(-4, 4, 1000)
    y_tails = stats.t.pdf(x_tails, df_tails)

    # Two-tailed test
    t_crit_lower = stats.t.ppf(alpha_tails / 2, df_tails)
    t_crit_upper = stats.t.ppf(1 - alpha_tails / 2, df_tails)

    ax2a.plot(x_tails, y_tails, 'b-')
    ax2a.fill_between(x_tails, 0, y_tails, where=(x_tails <= t_crit_lower), color='red', alpha=0.3)
    ax2a.fill_between(x_tails, 0, y_tails, where=(x_tails >= t_crit_upper), color='red', alpha=0.3)
    ax2a.axvline(t_crit_lower, color='red', linestyle='--')
    ax2a.axvline(t_crit_upper, color='red', linestyle='--')
    ax2a.set_title(f"Two-tailed (α={alpha_tails})")
    ax2a.set_xlabel("t-value")
    ax2a.text(t_crit_lower - 0.2, 0.01, f"{t_crit_lower:.2f}", color='red')
    ax2a.text(t_crit_upper + 0.1, 0.01, f"{t_crit_upper:.2f}", color='red')

    # Right-tailed test
    t_crit_right = stats.t.ppf(1 - alpha_tails, df_tails)

    ax2b.plot(x_tails, y_tails, 'b-')
    ax2b.fill_between(x_tails, 0, y_tails, where=(x_tails >= t_crit_right), color='red', alpha=0.3)
    ax2b.axvline(t_crit_right, color='red', linestyle='--')
    ax2b.set_title(f"Right-tailed (α={alpha_tails})")
    ax2b.set_xlabel("t-value")
    ax2b.text(t_crit_right + 0.1, 0.01, f"{t_crit_right:.2f}", color='red')

    # Left-tailed test
    t_crit_left = stats.t.ppf(alpha_tails, df_tails)

    ax2c.plot(x_tails, y_tails, 'b-')
    ax2c.fill_between(x_tails, 0, y_tails, where=(x_tails <= t_crit_left), color='red', alpha=0.3)
    ax2c.axvline(t_crit_left, color='red', linestyle='--')
    ax2c.set_title(f"Left-tailed (α={alpha_tails})")
    ax2c.set_xlabel("t-value")
    ax2c.text(t_crit_left - 0.2, 0.01, f"{t_crit_left:.2f}", color='red')

    st.pyplot(fig2)

    st.markdown("""
    **Types of t-tests and When to Use Them:**

    1. **Two-tailed Test:**
       - H₀: βⱼ = 0
       - H₁: βⱼ ≠ 0
       - Use when interested in any deviation from zero (most common)
       - Rejection regions in both tails

    2. **Right-tailed Test:**
       - H₀: βⱼ ≤ 0
       - H₁: βⱼ > 0
       - Use when only positive effects are of interest
       - Rejection region only in right tail

    3. **Left-tailed Test:**
       - H₀: βⱼ ≥ 0
       - H₁: βⱼ < 0
       - Use when only negative effects are of interest
       - Rejection region only in left tail
    """)

    # F-distribution section
    st.subheader("F-Distribution Properties")

    # Interactive visualization of how F-distribution changes with df
    df1_values = [1, 2, 5, 10]
    df2_values = [5, 20, 100]

    # Create figure
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    x_f = np.linspace(0, 5, 1000)
    line_styles = ['-', '--', '-.', ':']

    for i, df1 in enumerate(df1_values):
        for j, df2 in enumerate(df2_values):
            y_f = stats.f.pdf(x_f, df1, df2)
            ax3.plot(x_f, y_f, linestyle=line_styles[i % len(line_styles)],
                     label=f'df1 = {df1}, df2 = {df2}')

    ax3.set_xlabel('F-value')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('F-Distribution for Different Degrees of Freedom')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)

    st.pyplot(fig3)

    st.markdown("""
    **Key Properties of the F-Distribution:**

    1. **Shape:**
       - Always non-negative
       - Always right-skewed
       - Shape depends on two parameters: df1 (numerator) and df2 (denominator)

    2. **Degrees of Freedom:**
       - df1 (numerator df): Usually represents number of restrictions or parameters being tested
       - df2 (denominator df): Usually n-k-1 in regression context
       - Lower degrees of freedom result in more right-skewed distributions

    3. **Use in Regression:**
       - Testing overall model significance
       - Testing groups of coefficients (partial F-tests)
       - Testing nested models
       - Only right-tailed tests are used with F-distributions
    """)

    # Interactive example for F-distribution critical values
    st.subheader("F-test Critical Values and Rejection Regions")

    col1, col2, col3 = st.columns(3)
    with col1:
        df1_f = st.slider("Numerator df:", 1, 10, 3, key="df1_f")
    with col2:
        df2_f = st.slider("Denominator df:", 5, 100, 30, key="df2_f")
    with col3:
        alpha_f = st.slider("Significance Level (α):", 0.01, 0.20, 0.05, key="alpha_f_dist")

    # Create figure
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    # Generate F-distribution
    x_f_crit = np.linspace(0, 8, 1000)
    y_f_crit = stats.f.pdf(x_f_crit, df1_f, df2_f)

    # Calculate critical value
    f_crit = stats.f.ppf(1 - alpha_f, df1_f, df2_f)

    # Plot F-distribution
    ax4.plot(x_f_crit, y_f_crit, 'b-', lw=2, label=f'F({df1_f}, {df2_f})')

    # Shade rejection region
    ax4.fill_between(x_f_crit, 0, y_f_crit, where=(x_f_crit >= f_crit), color='red', alpha=0.3,
                     label='Rejection region')
    ax4.axvline(f_crit, color='red', linestyle='--', label=f'Critical value: {f_crit:.3f}')

    ax4.set_xlabel('F-value')
    ax4.set_ylabel('Probability Density')
    ax4.set_title(f'F-Distribution and Rejection Region (α={alpha_f})')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)

    st.pyplot(fig4)

    st.markdown("""
    **F-test in Regression Analysis:**

    1. **Overall Model F-test:**

       H₀: β₁ = β₂ = ... = βₖ = 0 (Model has no explanatory power)

       H₁: At least one βⱼ ≠ 0 (Model has some explanatory power)

       F-statistic = (R²/k) / [(1-R²)/(n-k-1)]

       where R² is the coefficient of determination

    2. **Partial F-test:**

       Used to test whether a subset of variables adds significant explanatory power

       Compares full model to a restricted model

       F-statistic = [(SSE_restricted - SSE_full)/q] / [SSE_full/(n-k-1)]

       where q is the number of restrictions (parameters being tested)

    3. **Relationship to t-tests:**

       When testing a single coefficient, F = t²

       F-test with (1, n-k-1) df is equivalent to a two-tailed t-test with n-k-1 df
    """)

st.sidebar.markdown("""
---
### About this App

This application was created to help students understand hypothesis testing in regression.

**Topics covered:**
- t-distribution and t-tests
- F-distribution and F-tests
- Rejection regions and critical values
- p-values and their interpretation 
- Multiple hypothesis tests

Use the navigation panel above to explore different concepts.
""")

# Help and guidance
with st.sidebar.expander("Tips for Learning"):
    st.markdown("""
    * Experiment with different parameters to see how distributions change
    * Pay attention to critical values and rejection regions
    * Try different test statistics to understand how decisions are made
    * Use the interactive regression example to see how real data affects test results
    * Focus on interpreting p-values correctly
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Hypothesis Testing Framework by Dr Merwan Roudane")