import pandas as pd
import streamlit as st

df_metrics = pd.read_csv('zirconium-dioxide/results/metrics.csv')
st.title('zirconium-dioxide')
st.dataframe(df_metrics, use_container_width=True)