import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import detrend

st.set_page_config(page_title="Norma Euclidiana FaceMesh", layout="wide")

st.title("Análise das normas euclidianas dos pontos 3D")

arquivo = st.file_uploader("Carregue o arquivo CSV", type=["csv", "txt"])

if arquivo is not None:
    df = pd.read_csv(arquivo)

    st.subheader("Pré-visualização dos dados")
    st.dataframe(df.head())

    frame_col = df.columns[0]
    tempo_col = df.columns[1]

    coord_cols = df.columns[2:]

    if len(coord_cols) % 3 != 0:
        st.error("O número de colunas de coordenadas não é múltiplo de 3.")
        st.stop()

    n_pontos = len(coord_cols) // 3

    st.success(f"Foram identificados {n_pontos} pontos tridimensionais.")

    tempo = df[tempo_col]

    normas = pd.DataFrame()
    normas[frame_col] = df[frame_col]
    normas[tempo_col] = tempo

    for i in range(n_pontos):
        x = detrend(df.iloc[:, 2 + 3*i].values)
        y = detrend(df.iloc[:, 2 + 3*i + 1].values)
        z = detrend(df.iloc[:, 2 + 3*i + 2].values)

    normas[f"Pt{i}_norma"] = np.sqrt(x**2 + y**2 + z**2)
    
    st.subheader("Selecionar pontos para visualização")

    pontos = [f"Pt{i}_norma" for i in range(n_pontos)]

    pontos_selecionados = st.multiselect(
        "Escolha os pontos",
        pontos,
        default=["Pt0_norma"]
    )

    usar_tempo = st.radio(
        "Eixo X",
        ["Tempo", "Frame"],
        horizontal=True
    )

    eixo_x = tempo_col if usar_tempo == "Tempo" else frame_col

    if pontos_selecionados:
        fig = go.Figure()

        for ponto in pontos_selecionados:
            fig.add_trace(
                go.Scatter(
                    x=normas[eixo_x],
                    y=normas[ponto],
                    mode="lines",
                    name=ponto
                )
            )

        fig.update_layout(
            height=650,
            xaxis_title=eixo_x,
            yaxis_title="Norma euclidiana",
            margin=dict(l=40, r=20, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabela das normas calculadas")
    st.dataframe(normas)

    csv = normas.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Baixar normas em CSV",
        data=csv,
        file_name="normas_euclidianas_pontos.csv",
        mime="text/csv"
    )
