import requests
import pandas as pd
import streamlit as st
import altair as alt

from src.config import (
    API_URL,
    DECISION_THRESHOLD,
)


# ============================
# Page Configuration
# ============================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="\u2699\ufe0f",
    layout="wide",
)


# ============================
# Session State
# ============================

if "api_result" not in st.session_state:
    st.session_state.api_result = None

if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False


def reset_dashboard():
    """
    Reset prediction results and dashboard state.
    """

    st.session_state.clear()


# ============================
# Header
# ============================

st.title(
    "Industrial Predictive Maintenance"
)

st.subheader(
    "Dashboard"
)

st.markdown(
    """
    Predict machine failure using an explainable
    XGBoost machine learning model.
    """
)


# ============================
# Sidebar
# ============================

st.sidebar.header(
    "Machine Information"
)


machine_type = st.sidebar.selectbox(
    "Machine Type",
    [
        "Low",
        "Medium",
        "High",
    ],
    index=1,
)


air_temperature = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=300.2,
    step=0.1,
)


process_temperature = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=450.0,
    value=309.4,
    step=0.1,
)


rotational_speed = st.sidebar.number_input(
    "Rotational Speed (RPM)",
    min_value=0,
    value=1500,
    step=1,
)


torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    value=42.5,
    step=0.1,
)


tool_wear = st.sidebar.number_input(
    "Tool Wear (minutes)",
    min_value=0,
    value=120,
    step=1,
)


# ============================
# Encode Machine Type
# ============================

type_m = (
    1
    if machine_type == "Medium"
    else 0
)

type_h = (
    1
    if machine_type == "High"
    else 0
)


# ============================
# Current Input Payload
# ============================

payload = {
    "Air_temperature_K": float(
        air_temperature
    ),
    "Process_temperature_K": float(
        process_temperature
    ),
    "Rotational_speed_rpm": int(
        rotational_speed
    ),
    "Torque_Nm": float(
        torque
    ),
    "Tool_wear_min": int(
        tool_wear
    ),
    "Type_M": type_m,
    "Type_H": type_h,
}


# ============================
# Sidebar Prediction Controls
# ============================

st.sidebar.divider()

st.sidebar.subheader(
    "Monitoring"
)

start_monitoring = st.sidebar.button(
    "Start Monitoring",
    width="stretch",
)

stop_monitoring = st.sidebar.button(
    "Stop Monitoring",
    width="stretch",
)

if start_monitoring:
    st.session_state.monitoring_active = True

if stop_monitoring:
    st.session_state.monitoring_active = False

if st.session_state.monitoring_active:
    st.sidebar.success(
        "Monitoring: Active"
    )
else:
    st.sidebar.info(
        "Monitoring: Stopped"
    )

st.sidebar.divider()

st.sidebar.subheader(
    "Prediction"
)


predict_button = st.sidebar.button(
    "Predict Machine Failure",
    width="stretch",
)


st.sidebar.button(
    "Reset",
    width="stretch",
    on_click=reset_dashboard,
)


# ============================
# FastAPI Request
# ============================

if predict_button:

    try:

        with st.spinner(
            "Running machine failure prediction..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=90,
            )

        if response.status_code == 200:

            st.session_state.api_result = (
                response.json()
            )

        else:

            st.session_state.api_result = None

            st.error(
                f"Prediction failed "
                f"(HTTP {response.status_code})."
            )

            try:

                error_detail = response.json()

                st.caption(
                    str(
                        error_detail
                    )
                )

            except ValueError:

                st.caption(
                    response.text
                )

    except requests.RequestException as error:

        st.session_state.api_result = None

        st.error(
            "Unable to connect to the "
            "FastAPI server."
        )

        st.caption(
            str(error)
        )


api_result = (
    st.session_state.api_result
)


# ============================
# Placeholder Data
# ============================

placeholder = pd.DataFrame(
    {
        "Feature": [
            "Waiting for prediction..."
        ],
        "Feature Value": [
            "-"
        ],
        "SHAP Contribution": [
            "-"
        ],
    }
)


# ============================
# Dashboard
# ============================

st.divider()


col1, col2 = st.columns(
    [1, 1]
)


# ============================
# Prediction Results
# ============================

with col1:

    st.subheader(
        "Prediction Result"
    )

    metric1, metric2 = (
        st.columns(2)
    )


    # ----------------------------
    # Predicted Class
    # ----------------------------

    with metric1:

        st.write(
            "**Predicted Class**"
        )

        if api_result:

            if (
                api_result[
                    "predicted_class"
                ]
                == 0
            ):

                st.success(
                    "No Failure"
                )

            else:

                st.error(
                    "Failure"
                )

        else:

            st.info(
                "--"
            )


    # ----------------------------
    # Failure Probability
    # ----------------------------

    with metric2:

        failure_probability = (
            (
                f"{api_result['failure_probability']:.2%}"
            )
            if api_result
            else "--"
        )

        st.metric(
            "Failure Probability",
            failure_probability,
        )

        if api_result:

            probability = float(
                api_result[
                    "failure_probability"
                ]
            )

            threshold = float(
                api_result[
                    "decision_threshold"
                ]
            )

            if probability >= threshold:

                st.error(
                    "High Failure Risk"
                )

            else:

                st.success(
                    "Low Failure Risk"
                )


    metric3, metric4 = (
        st.columns(2)
    )


    # ----------------------------
    # Decision Threshold
    # ----------------------------

    with metric3:

        decision_threshold = (
            (
                f"{api_result['decision_threshold']:.4f}"
            )
            if api_result
            else f"{DECISION_THRESHOLD:.4f}"
        )

        st.metric(
            "Decision Threshold",
            decision_threshold,
        )


    # ----------------------------
    # Operational Decision
    # ----------------------------

    with metric4:

        st.write(
            "**Decision**"
        )

        if api_result:

            if (
                api_result[
                    "decision_prediction"
                ]
                == 0
            ):

                st.success(
                    "Normal Operation"
                )

            else:

                st.error(
                    "Maintenance Required"
                )

        else:

            st.info(
                "--"
            )


    # ----------------------------
    # Overall Status
    # ----------------------------

    if api_result:

        if (
            api_result[
                "decision_prediction"
            ]
            == 0
        ):

            st.success(
                "Machine operating normally."
            )

        else:

            st.error(
                "Maintenance recommended. "
                "Inspect the machine before "
                "continued operation."
            )


    # ----------------------------
    # Prediction Summary
    # ----------------------------

    if api_result:

        st.divider()

        st.subheader(
            "Prediction Summary"
        )

        predicted_class_text = (
            "No Failure"
            if api_result[
                "predicted_class"
            ] == 0
            else "Failure"
        )

        decision_text = (
            "Normal Operation"
            if api_result[
                "decision_prediction"
            ] == 0
            else "Maintenance Required"
        )

        st.write(
            f"**Machine Type:** "
            f"{machine_type}"
        )

        st.write(
            f"**Predicted Class:** "
            f"{predicted_class_text}"
        )

        st.write(
            f"**Failure Probability:** "
            f"{api_result['failure_probability']:.2%}"
        )

        st.write(
            f"**Decision Threshold:** "
            f"{api_result['decision_threshold']:.4f}"
        )

        st.write(
            f"**Decision:** "
            f"{decision_text}"
        )


# ----------------------------
# Input Data
# ----------------------------

with st.expander(
     "Input Data"
):

     st.json(
         payload
    )


# ============================
# SHAP Explanation
# ============================

with col2:

    st.subheader(
        "Top Contributing Features"
    )

    if api_result:

        shap_df = pd.DataFrame(
            api_result[
                "top_contributors"
            ]
        )


        # ----------------------------
        # Numeric Formatting
        # ----------------------------

        shap_df[
            "feature_value"
        ] = pd.to_numeric(
            shap_df[
                "feature_value"
            ],
            errors="coerce",
        ).round(2)


        shap_df[
            "shap_value"
        ] = pd.to_numeric(
            shap_df[
                "shap_value"
            ],
            errors="coerce",
        ).round(3)


        shap_df = shap_df[
            [
                "feature",
                "feature_value",
                "shap_value",
            ]
        ]


        shap_df.columns = [
            "Feature",
            "Feature Value",
            "SHAP Contribution",
        ]


        # ----------------------------
        # Readable Feature Names
        # ----------------------------

        feature_names = {
            "Torque_Nm":
                "Torque (Nm)",

            "Tool_wear_min":
                "Tool Wear (min)",

            "Rotational_speed_rpm":
                "Rotational Speed (RPM)",

            "Air_temperature_K":
                "Air Temperature (K)",

            "Process_temperature_K":
                "Process Temperature (K)",

            "Type_M":
                "Machine Type: Medium",

            "Type_H":
                "Machine Type: High",
        }


        shap_df[
            "Feature"
        ] = shap_df[
            "Feature"
        ].replace(
            feature_names
        )


        # ----------------------------
        # SHAP Table
        # ----------------------------

        st.dataframe(
            shap_df,
            width="stretch",
            hide_index=True,
        )


        # ----------------------------
        # Feature Impact Chart
        # ----------------------------

        st.subheader(
            "Feature Impact on Prediction"
        )


        chart_df = (
            shap_df
            .set_index(
                "Feature"
            )[
                "SHAP Contribution"
            ]
        )


        st.bar_chart(
            chart_df,
            horizontal=True,
            height=300,
        )


        # ----------------------------
        # Local SHAP Interpretation
        # ----------------------------

        st.info(
            "The table and chart show the most "
            "influential features for this individual "
            "prediction. Positive SHAP values increase "
            "the model's predicted failure risk, while "
            "negative SHAP values decrease it. Larger "
            "absolute SHAP values indicate greater "
            "influence on this prediction."
        )

    else:

        st.dataframe(
            placeholder,
            width="stretch",
            hide_index=True,
        )

        st.info(
            "Run a prediction to view the most "
            "important contributing features."
)


# ============================
# Model Information
# ============================

with st.expander(
    "Model Information"
):

    st.write(
        "**Model:** Tuned XGBoost"
    )

    st.write(
        "**Task:** Predictive maintenance using "
        "binary classification"
    )

    st.write(
        "**Explainability:** SHAP"
    )

    st.write(
        f"**Operational Decision Threshold:** "
        f"{DECISION_THRESHOLD:.4f}"
    )

    st.write(
        f"""
        The predicted class represents the model's
        standard classification result.

        The operational decision uses the optimized
        probability threshold of
        {DECISION_THRESHOLD:.4f}.
        """
    )
# ============================
# Prediction History
# ============================

st.divider()

st.subheader("Prediction History")

st.caption(
    "Recent machine failure predictions stored "
    "in the Supabase database."
)

history_limit = st.selectbox(
    "Number of recent predictions",
    options=[5, 10, 20, 50],
    index=3,
)

try:
    history_url = API_URL.replace(
        "/predict",
        "/history",
    )

    history_response = requests.get(
        history_url,
        params={"limit": history_limit},
        timeout=90,
    )

    if history_response.status_code == 200:
        history_data = history_response.json()

        predictions = history_data.get(
            "predictions",
            [],
        )

        if predictions:
            history_df = pd.DataFrame(predictions)

            # ============================
            # Normalise Historical Values
            # ============================

            history_df["decision"] = (
                history_df["decision"]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "0": "Normal Operation",
                        "Normal": "Normal Operation",
                    }
                )
            )

            history_df["predicted_class"] = (
                history_df["predicted_class"]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "0": "No Failure",
                        "1": "Failure",
                    }
                )
            )

            # ============================
            # Operational Overview
            # ============================

            st.subheader("Operational Overview")

            total_predictions = len(history_df)

            failures_detected = (
                history_df["predicted_class"]
                .eq("Failure")
                .sum()
            )

            maintenance_required = (
                history_df["decision"]
                .eq("Maintenance Required")
                .sum()
            )

            average_failure_probability = (
                pd.to_numeric(
                    history_df["failure_probability"],
                    errors="coerce",
                )
                .mean()
            )

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                st.metric(
                    "Total Predictions",
                    total_predictions,
                )

            with kpi2:
                st.metric(
                    "Failures Detected",
                    int(failures_detected),
                )

            with kpi3:
                st.metric(
                    "Maintenance Required",
                    int(maintenance_required),
                )

            with kpi4:
                if pd.notna(
                    average_failure_probability
                ):
                    average_probability_text = (
                        f"{average_failure_probability:.2%}"
                    )
                else:
                    average_probability_text = "--"

                st.metric(
                    "Average Failure Probability",
                    average_probability_text,
                )

            # ============================
            # Risk Distribution
            # ============================

            st.divider()

            st.subheader("Risk Distribution")

            risk_counts = (
                history_df["decision"]
                .value_counts()
            )

            risk_chart_df = pd.DataFrame(
                {
                    "Decision": risk_counts.index,
                    "Predictions": risk_counts.values,
                }
            ).set_index(
                "Decision"
            )

            st.bar_chart(
                risk_chart_df,
                height=300,
            )

            st.caption(
                "Distribution of operational decisions "
                "across the selected prediction history."
            )

            # ============================
            # Failure Probability Trend
            # ============================

            st.subheader(
                "Failure Probability Trend"
            )

            trend_df = history_df[
                [
                    "created_at",
                    "failure_probability",
                ]
            ].copy()

            trend_df["created_at"] = pd.to_datetime(
                trend_df["created_at"],
                errors="coerce",
            )

            trend_df["failure_probability"] = (
                pd.to_numeric(
                    trend_df["failure_probability"],
                    errors="coerce",
                )
            )

            trend_df = trend_df.dropna(
                subset=[
                    "created_at",
                    "failure_probability",
                ]
            )

            trend_df = trend_df.sort_values(
                "created_at"
            )

            if not trend_df.empty:

                trend_df[
                    "failure_probability_pct"
                ] = (
                    trend_df["failure_probability"]
                    * 100
                )

                threshold_percentage = (
                    DECISION_THRESHOLD * 100
                )

                # ----------------------------
                # Probability Trend
                # ----------------------------

                probability_line = (
                    alt.Chart(trend_df)
                    .mark_line(
                        strokeWidth=3,
                    )
                    .encode(
                        x=alt.X(
                            "created_at:T",
                            title="Prediction Time",
                        ),
                        y=alt.Y(
                            "failure_probability_pct:Q",
                            title="Failure Probability (%)",
                            scale=alt.Scale(
                                domain=[0, 100]
                            ),
                        ),
                    )
                )

                # ----------------------------
                # Risk Status
                # ----------------------------

                trend_df["risk_status"] = (
                    trend_df[
                        "failure_probability_pct"
                    ]
                    .ge(
                        threshold_percentage
                    )
                    .map(
                        {
                            True:
                                "Maintenance Required",
                            False:
                                "Normal Operation",
                        }
                    )
                )

                probability_points = (
                    alt.Chart(trend_df)
                    .mark_circle(
                        size=90,
                    )
                    .encode(
                        x=alt.X(
                            "created_at:T",
                            title="Prediction Time",
                        ),
                        y=alt.Y(
                            "failure_probability_pct:Q",
                            title="Failure Probability (%)",
                            scale=alt.Scale(
                                domain=[0, 100]
                            ),
                        ),
                        color=alt.Color(
                            "risk_status:N",
                            title="Operational Status",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "created_at:T",
                                title="Prediction Time",
                            ),
                            alt.Tooltip(
                                "failure_probability_pct:Q",
                                title=(
                                    "Failure Probability (%)"
                                ),
                                format=".2f",
                            ),
                            alt.Tooltip(
                                "risk_status:N",
                                title="Operational Status",
                            ),
                        ],
                    )
                )

                # ----------------------------
                # Decision Threshold
                # ----------------------------

                threshold_df = pd.DataFrame(
                    {
                        "threshold": [
                            threshold_percentage
                        ],
                        "label": [
                            (
                                "Decision Threshold: "
                                f"{threshold_percentage:.2f}%"
                            )
                        ],
                    }
                )

                threshold_rule = (
                    alt.Chart(threshold_df)
                    .mark_rule(
                        strokeDash=[8, 5],
                        strokeWidth=3,
                    )
                    .encode(
                        y=alt.Y(
                            "threshold:Q",
                            scale=alt.Scale(
                                domain=[0, 100]
                            ),
                        )
                    )
                )

                threshold_label = (
                    alt.Chart(threshold_df)
                    .mark_text(
                        align="right",
                        baseline="bottom",
                        dx=-8,
                        dy=-6,
                        fontSize=13,
                        fontWeight="bold",
                    )
                    .encode(
                        y=alt.Y(
                            "threshold:Q",
                            scale=alt.Scale(
                                domain=[0, 100]
                            ),
                        ),
                        text="label:N",
                    )
                )

                # ----------------------------
                # Combined Trend Chart
                # ----------------------------

                trend_chart = (
                    probability_line
                    + probability_points
                    + threshold_rule
                    + threshold_label
                ).properties(
                    height=340
                )

                st.altair_chart(
                    trend_chart,
                    use_container_width=True,
                )

                st.caption(
                    "Failure probability over time. "
                    f"The horizontal reference line marks "
                    f"the {threshold_percentage:.2f}% "
                    "maintenance decision threshold."
                )

            else:

                st.info(
                    "Not enough prediction history "
                    "is available to display the trend."
                )
            # ============================
            # Maintenance Alerts
            # ============================

            st.divider()

            st.subheader(
                "Maintenance Alerts"
            )

            st.caption(
                "Prioritised predictions exceeding the "
                "operational maintenance decision threshold."
            )

            # Operational severity rule.
            # Critical is currently defined as
            # 5 percentage points above the
            # maintenance decision threshold.
            CRITICAL_MARGIN = 0.05

            critical_threshold = min(
                1.0,
                DECISION_THRESHOLD + CRITICAL_MARGIN,
            )

            alert_df = history_df.copy()

            alert_df[
                "failure_probability"
            ] = pd.to_numeric(
                alert_df[
                    "failure_probability"
                ],
                errors="coerce",
            )

            # Keep predictions that meet or exceed
            # the maintenance decision threshold.
            high_risk_df = alert_df[
                alert_df[
                    "failure_probability"
                ].ge(
                    DECISION_THRESHOLD
                )
            ].copy()

            # Assign operational severity.
            if not high_risk_df.empty:

                high_risk_df[
                    "severity"
                ] = (
                    high_risk_df[
                        "failure_probability"
                    ]
                    .ge(
                        critical_threshold
                    )
                    .map(
                        {
                            True: "Critical",
                            False: "High",
                        }
                    )
                )

                high_risk_df[
                    "threshold_margin_pct"
                ] = (
                    (
                        high_risk_df[
                            "failure_probability"
                        ]
                        - DECISION_THRESHOLD
                    )
                    * 100
                )

            # Convert timestamps and place
            # the newest alert first.
            if (
                "created_at"
                in high_risk_df.columns
            ):

                high_risk_df[
                    "created_at"
                ] = pd.to_datetime(
                    high_risk_df[
                        "created_at"
                    ],
                    errors="coerce",
                )

                high_risk_df = (
                    high_risk_df.sort_values(
                        "created_at",
                        ascending=False,
                    )
                )

            # ============================
            # Alert Summary
            # ============================

            active_alerts = len(
                high_risk_df
            )

            if not high_risk_df.empty:

                critical_alerts = (
                    high_risk_df[
                        "severity"
                    ]
                    .eq("Critical")
                    .sum()
                )

            else:

                critical_alerts = 0

            alert_col1, alert_col2, alert_col3 = (
                st.columns(3)
            )

            with alert_col1:

                st.metric(
                    "Active Alerts",
                    active_alerts,
                )

            with alert_col2:

                st.metric(
                    "Critical Alerts",
                    int(
                        critical_alerts
                    ),
                )

            with alert_col3:

                st.metric(
                    "Decision Threshold",
                    f"{DECISION_THRESHOLD:.2%}",
                )

            # ============================
            # Latest Maintenance Alert
            # ============================

            if high_risk_df.empty:

                st.success(
                    "No maintenance alerts detected "
                    "in the selected prediction history."
                )

            else:

                latest_alert = (
                    high_risk_df.iloc[0]
                )

                alert_probability = float(
                    latest_alert[
                        "failure_probability"
                    ]
                )

                severity = str(
                    latest_alert[
                        "severity"
                    ]
                )

                threshold_margin = float(
                    latest_alert[
                        "threshold_margin_pct"
                    ]
                )

                machine_type = str(
                    latest_alert.get(
                        "machine_type",
                        "Unknown",
                    )
                )

                alert_time = (
                    latest_alert.get(
                        "created_at"
                    )
                )

                if pd.notna(
                    alert_time
                ):

                    alert_time_text = (
                        alert_time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )

                else:

                    alert_time_text = (
                        "Unknown"
                    )

                if severity == "Critical":

                    st.error(
                        "Critical maintenance alert: "
                        "failure probability is substantially "
                        "above the decision threshold."
                    )

                else:

                    st.warning(
                        "High maintenance alert: "
                        "failure probability has exceeded "
                        "the decision threshold."
                    )

                st.markdown(
                    "#### Latest Maintenance Alert"
                )

                info1, info2, info3 = (
                    st.columns(3)
                )

                with info1:

                    st.metric(
                        "Failure Probability",
                        f"{alert_probability:.2%}",
                    )

                    st.write(
                        f"**Machine Type:** "
                        f"{machine_type}"
                    )

                with info2:

                    st.metric(
                        "Severity",
                        severity,
                    )

                    st.write(
                        f"**Prediction Time:** "
                        f"{alert_time_text}"
                    )

                with info3:

                    st.metric(
                        "Above Threshold",
                        f"{threshold_margin:.2f} pp",
                    )

                    st.write(
                        f"**Threshold:** "
                        f"{DECISION_THRESHOLD:.2%}"
                    )

                if severity == "Critical":

                    st.error(
                        "Recommended Action: Prioritise "
                        "immediate maintenance inspection "
                        "and assess the machine before "
                        "continued operation."
                    )

                else:

                    st.warning(
                        "Recommended Action: Schedule "
                        "a maintenance inspection and "
                        "review the machine before "
                        "continued operation."
                    )

            st.caption(
                "Severity is an operational prioritisation "
                "rule, not a model prediction. "
                f"Critical is currently defined as "
                f"{critical_threshold:.2%} or above."
            )

            # ============================
            # Maintenance Priority Queue
            # ============================

            st.divider()

            st.subheader(
                "Maintenance Priority Queue"
            )

            st.caption(
                "Threshold-exceeding predictions ranked "
                "by operational severity and failure probability."
            )

            if high_risk_df.empty:

                st.info(
                    "No machines currently require "
                    "maintenance prioritisation."
                )

            else:

                priority_df = (
                    high_risk_df.copy()
                )

                # Rank Critical above High.
                severity_rank = {
                    "Critical": 1,
                    "High": 2,
                }

                priority_df[
                    "severity_rank"
                ] = (
                    priority_df[
                        "severity"
                    ]
                    .map(
                        severity_rank
                    )
                    .fillna(99)
                )

                # Sort by severity first,
                # then highest failure probability,
                # then most recent prediction.
                sort_columns = [
                    "severity_rank",
                    "failure_probability",
                ]

                sort_ascending = [
                    True,
                    False,
                ]

                if (
                    "created_at"
                    in priority_df.columns
                ):

                    sort_columns.append(
                        "created_at"
                    )

                    sort_ascending.append(
                        False
                    )

                priority_df = (
                    priority_df.sort_values(
                        by=sort_columns,
                        ascending=sort_ascending,
                    )
                )

                # Add queue position.
                priority_df = (
                    priority_df.reset_index(
                        drop=True
                    )
                )

                priority_df[
                    "priority"
                ] = (
                    priority_df.index
                    + 1
                )

                # Format values for display.
                priority_df[
                    "failure_probability_pct"
                ] = (
                    priority_df[
                        "failure_probability"
                    ]
                    * 100
                ).round(2)

                priority_df[
                    "threshold_margin_pct"
                ] = (
                    pd.to_numeric(
                        priority_df[
                            "threshold_margin_pct"
                        ],
                        errors="coerce",
                    )
                    .round(2)
                )

                if (
                    "created_at"
                    in priority_df.columns
                ):

                    priority_df[
                        "created_at"
                    ] = (
                        pd.to_datetime(
                            priority_df[
                                "created_at"
                            ],
                            errors="coerce",
                        )
                        .dt.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )

                queue_columns = [
                    "priority",
                    "severity",
                    "machine_type",
                    "failure_probability_pct",
                    "threshold_margin_pct",
                    "created_at",
                ]

                available_queue_columns = [
                    column
                    for column in queue_columns
                    if column
                    in priority_df.columns
                ]

                queue_display_df = (
                    priority_df[
                        available_queue_columns
                    ].copy()
                )

                queue_display_df = (
                    queue_display_df.rename(
                        columns={
                            "priority":
                                "Priority",
                            "severity":
                                "Severity",
                            "machine_type":
                                "Machine Type",
                            "failure_probability_pct":
                                "Failure Probability (%)",
                            "threshold_margin_pct":
                                "Above Threshold (pp)",
                            "created_at":
                                "Prediction Time",
                        }
                    )
                )

                st.dataframe(
                    queue_display_df,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    f"{len(queue_display_df)} "
                    "maintenance alert(s) currently "
                    "in the priority queue."
                )

            # ============================
            # History Filters
            # ============================

            st.divider()

            st.subheader(
                "History Filters"
            )

            filter_col1, filter_col2 = (
                st.columns(2)
            )

            with filter_col1:

                decision_filter = st.selectbox(
                    "Decision",
                    options=[
                        "All",
                        "Normal Operation",
                        "Maintenance Required",
                    ],
                )

            with filter_col2:

                class_filter = st.selectbox(
                    "Predicted Class",
                    options=[
                        "All",
                        "No Failure",
                        "Failure",
                    ],
                )

            filtered_history_df = (
                history_df.copy()
            )

            if decision_filter != "All":

                filtered_history_df = (
                    filtered_history_df[
                        filtered_history_df[
                            "decision"
                        ].eq(
                            decision_filter
                        )
                    ]
                )

            if class_filter != "All":

                filtered_history_df = (
                    filtered_history_df[
                        filtered_history_df[
                            "predicted_class"
                        ].eq(
                            class_filter
                        )
                    ]
                )

            active_filters = []

            if decision_filter != "All":

                active_filters.append(
                    f"Decision: {decision_filter}"
                )

            if class_filter != "All":

                active_filters.append(
                    f"Predicted Class: "
                    f"{class_filter}"
                )

            if active_filters:

                st.caption(
                    " | ".join(
                        active_filters
                    )
                )

            else:

                st.caption(
                    "Showing all prediction records."
                )

            # ============================
            # Prediction History Records
            # ============================

            st.divider()

            st.subheader(
                "Prediction History Records"
            )

            st.caption(
                "Detailed prediction records based on "
                "the selected history filters."
            )

            display_columns = [
                "created_at",
                "machine_type",
                "predicted_class",
                "failure_probability",
                "decision",
                "air_temperature_k",
                "process_temperature_k",
                "rotational_speed_rpm",
                "torque_nm",
                "tool_wear_min",

            ]

            available_columns = [
                column
                for column in display_columns
                if column in filtered_history_df.columns
            ]

            display_history_df = (
                filtered_history_df[
                    available_columns
                ].copy()
            )

            # ============================
            # Sort and Format Date / Time
            # ============================

            if (
                "created_at"
                in display_history_df.columns
            ):

                display_history_df[
                    "created_at"
                ] = pd.to_datetime(
                    display_history_df[
                        "created_at"
                    ],
                    errors="coerce",
                )
                        # ============================
            # Sort and Format Date / Time
            # ============================

            if (
                "created_at"
                in display_history_df.columns
            ):

                display_history_df[
                    "created_at"
                ] = pd.to_datetime(
                    display_history_df[
                        "created_at"
                    ],
                    errors="coerce",
                )

                display_history_df = (
                    display_history_df.sort_values(
                        "created_at",
                        ascending=False,
                    )
                )

                display_history_df[
                    "created_at"
                ] = (
                    display_history_df[
                        "created_at"
                    ]
                    .dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

            # ============================
            # Format Failure Probability
            # ============================

            if (
                "failure_probability"
                in display_history_df.columns
            ):

                display_history_df[
                    "failure_probability"
                ] = (
                    pd.to_numeric(
                        display_history_df[
                            "failure_probability"
                        ],
                        errors="coerce",
                    )
                    * 100
                ).round(2)
                        # ============================
            # Rename Columns for Display
            # ============================

            display_history_df = (
                display_history_df.rename(
                    columns={
                        "created_at":
                            "Date / Time",
                        "machine_type":
                            "Machine Type",
                        "air_temperature_k":
                            "Air Temperature (K)",
                        "process_temperature_k":
                            "Process Temperature (K)",
                        "rotational_speed_rpm":
                            "Rotational Speed (RPM)",
                        "torque_nm":
                            "Torque (Nm)",
                        "tool_wear_min":
                            "Tool Wear (min)",
                        "predicted_class":
                            "Predicted Class",
                        "failure_probability":
                            "Failure Probability (%)",
                        "decision":
                            "Decision",
                    }
                )
            )

            # ============================
            # Display History Records
            # ============================

            if display_history_df.empty:

                st.info(
                    "No prediction records match "
                    "the selected filters."
                )

            else:

                st.dataframe(
                    display_history_df,
                    width="stretch",
                    hide_index=True,
                )

            st.caption(
                f"Showing "
                f"{len(display_history_df)} "
                f"of {len(history_df)} "
                "prediction record(s)."
            )

        else:

            st.info(
                "No prediction history is "
                "currently available."
            )

    else:

        st.warning(
            "Prediction history could not "
            f"be loaded "
            f"(HTTP "
            f"{history_response.status_code})."
        )

except requests.RequestException as error:

    st.warning(
        "Unable to connect to the prediction "
        "history service."
    )

    st.caption(
        str(error)
    )


# ============================
# Footer
# ============================

st.divider()

st.caption(
    "Industrial Predictive Maintenance Dashboard | "
    "Model: Tuned XGBoost | "
    "Explainability: SHAP | "
    f"Decision Threshold: "
    f"{DECISION_THRESHOLD:.4f}"
)
