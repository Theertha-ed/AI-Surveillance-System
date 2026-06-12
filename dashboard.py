import streamlit as st
import pandas as pd
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Surveillance Control Room",
    layout="wide"
)

st.title("🛡️ AI Surveillance Control Room Dashboard")
st.caption("YOLOv8 + DeepSORT + Real-time Monitoring System")

log_file = "logs/detections.csv"


# ---------------- LOAD DATA ----------------
def load_data():
    try:
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            return pd.read_csv(log_file)
        else:
            return pd.DataFrame(columns=["Track_ID", "Object", "Event", "Confidence"])
    except Exception:
        return pd.DataFrame(columns=["Track_ID", "Object", "Event", "Confidence"])


df = load_data()


# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

st.sidebar.success("System Running")

st.sidebar.metric("Total Records", len(df))

st.sidebar.metric(
    "Unique Objects",
    df["Track_ID"].nunique() if "Track_ID" in df.columns else 0
)

st.sidebar.metric(
    "Intrusions",
    len(df[df["Event"] == "ZONE_INTRUSION"]) if "Event" in df.columns else 0
)


# ---------------- MAIN METRICS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📦 Total Detections")
    st.metric("", len(df))

with col2:
    st.markdown("### 👁️ Unique Objects")
    st.metric("", df["Track_ID"].nunique() if "Track_ID" in df.columns else 0)

with col3:
    st.markdown("### 🚨 Intrusions")
    intrusions = len(df[df["Event"] == "ZONE_INTRUSION"]) if "Event" in df.columns else 0
    st.metric("", intrusions)


st.divider()


# ---------------- OBJECT DISTRIBUTION ----------------
st.subheader("📊 Object Detection Distribution")

if "Object" in df.columns and not df.empty:
    st.bar_chart(df["Object"].value_counts())
else:
    st.warning("No detection data available yet.")


# ---------------- INTRUSION LOG ----------------
st.subheader("🚨 Intrusion Events")

if "Event" in df.columns and not df.empty:
    st.dataframe(
        df[df["Event"] == "ZONE_INTRUSION"],
        use_container_width=True
    )
else:
    st.info("No intrusion events detected.")


# ---------------- FULL LOG ----------------
st.subheader("📁 Full Activity Log")

st.dataframe(df.tail(50), use_container_width=True)


# ---------------- DOWNLOAD REPORT ----------------
st.subheader("📥 Download Report")

if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
    with open(log_file, "rb") as file:
        st.download_button(
            label="⬇️ Download Detection Report (CSV)",
            data=file,
            file_name="detections_report.csv",
            mime="text/csv"
        )
else:
    st.info("No data available to download.")


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI Surveillance System | YOLOv8 + DeepSORT + Streamlit")