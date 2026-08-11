import requests
import pandas as pd
import streamlit as st

from src.config import (
    API_URL,
    DECISION_THRESHOLD,
)


# ============================
# Page Configuration
# ============================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)


# ============================
# Session State
# ============================

if "api_result" not in st.session_state:
    st.session_state.api_result = None


def reset_dashboard():
    """
    Reset prediction results and dashboard state.
    """

    st.session_state.clear()


# ============================
# Header
# ============================

st.title(
    "⚙️ Industrial Predictive Maintenance"
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
    "Prediction"
)


predict_button = st.sidebar.button(
    "🚀 Predict Machine Failure",
    width="stretch",
)


st.sidebar.button(
    "🔄 Reset",
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
                timeout=10,
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
                    "🟢 No Failure"
                )

            else:

                st.error(
                    "🔴 Failure"
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
                    "🔴 High Failure Risk"
                )

            else:

                st.success(
                    "🟢 Low Failure Risk"
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
                    "✅ Normal Operation"
                )

            else:

                st.error(
                    "⚠️ Maintenance Required"
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
    "ℹ️ Model Information"
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
# Footer
# ============================

st.divider()

st.caption(
    f"Industrial Predictive Maintenance Dashboard | "
    f"Model: Tuned XGBoost | "
    f"Explainability: SHAP | "
    f"Decision Threshold: {DECISION_THRESHOLD:.4f}"
)