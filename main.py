import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import detrend

st.set_page_config(
    page_title="Norma Euclidiana FaceMesh",
    layout="wide"
)

st.title("Análise das normas euclidianas dos pontos 3D")

arquivo = st.file_uploader(
    "Carregue o arquivo principal CSV",
    type=["csv", "txt"]
)

arquivo_parametros = st.file_uploader(
    "Carregue o arquivo de parâmetros normativos",
    type=["csv", "txt"]
)

if arquivo is not None and arquivo_parametros is not None:

    # =========================
    # LEITURA DOS ARQUIVOS
    # =========================

    df = pd.read_csv(
        arquivo,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    parametros = pd.read_csv(
        arquivo_parametros,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    # =========================
    # LIMPEZA DOS NOMES
    # =========================

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    parametros.columns = (
        parametros.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    st.subheader("Pré-visualização dos dados principais")
    st.dataframe(df.head())

    st.subheader("Pré-visualização dos parâmetros normativos")
    st.dataframe(parametros.head())

    st.subheader("Colunas detectadas no arquivo normativo")
    st.write(parametros.columns.tolist())

    # =========================
    # IDENTIFICAÇÃO DA COLUNA LEVEL
    # =========================

    if not any(col.lower() == "level" for col in parametros.columns):

        st.error(
            f"Coluna 'Level' não encontrada.\n\n"
            f"Colunas detectadas: {parametros.columns.tolist()}"
        )

        st.stop()

    level_col = [
        col for col in parametros.columns
        if col.lower() == "level"
    ][0]

    # =========================
    # IDENTIFICAÇÃO DAS COLUNAS
    # =========================

    frame_col = df.columns[0]
    tempo_col = df.columns[1]

    df[tempo_col] = pd.to_numeric(
        df[tempo_col],
        errors="coerce"
    )

    df = df.dropna(subset=[tempo_col])

    coord_cols = df.columns[2:]

    if len(coord_cols) % 3 != 0:

        st.error(
            "O número de colunas de coordenadas "
            "não é múltiplo de 3."
        )

        st.stop()

    n_pontos = len(coord_cols) // 3

    st.success(
        f"Foram identificados {n_pontos} pontos tridimensionais."
    )

    # =========================
    # CONFIGURAÇÕES
    # =========================

    unidade_tempo = st.radio(
        "Unidade da coluna de tempo",
        ["segundos", "milissegundos"],
        horizontal=True
    )

    janela_ms = st.number_input(
        "Janela para autozero (ms)",
        min_value=1,
        max_value=1000,
        value=50,
        step=1
    )

    usar_detrend = st.checkbox(
        "Aplicar detrend após autozero",
        value=True
    )

    # =========================
    # TEMPO
    # =========================

    tempo = df[tempo_col].astype(float).values

    if unidade_tempo == "segundos":
        janela_autozero = janela_ms / 1000
    else:
        janela_autozero = janela_ms

    tempo_inicial = tempo[0]

    idx_autozero = (
        tempo <= tempo_inicial + janela_autozero
    )

    if np.sum(idx_autozero) < 2:

        st.warning(
            "A janela de autozero possui menos "
            "de 2 amostras."
        )

        idx_autozero = np.arange(min(5, len(tempo)))

    # =========================
    # DATAFRAME DE NORMAS
    # =========================

    normas = pd.DataFrame()

    normas[frame_col] = df[frame_col].values
    normas[tempo_col] = tempo

    rms_pontos = []

    x_medios = []
    y_medios = []
    z_medios = []

    # =========================
    # PROCESSAMENTO
    # =========================

    for i in range(n_pontos):

        x_original = pd.to_numeric(
            df.iloc[:, 2 + 3*i],
            errors="coerce"
        ).values

        y_original = pd.to_numeric(
            df.iloc[:, 2 + 3*i + 1],
            errors="coerce"
        ).values

        z_original = pd.to_numeric(
            df.iloc[:, 2 + 3*i + 2],
            errors="coerce"
        ).values

        # =====================
        # AUTOZERO
        # =====================

        x_zero = np.nanmean(
            x_original[idx_autozero]
        )

        y_zero = np.nanmean(
            y_original[idx_autozero]
        )

        z_zero = np.nanmean(
            z_original[idx_autozero]
        )

        x_autozero = x_original - x_zero
        y_autozero = y_original - y_zero
        z_autozero = z_original - z_zero

        # =====================
        # INTERPOLAÇÃO
        # =====================

        x_autozero = (
            pd.Series(x_autozero)
            .interpolate()
            .bfill()
            .ffill()
            .values
        )

        y_autozero = (
            pd.Series(y_autozero)
            .interpolate()
            .bfill()
            .ffill()
            .values
        )

        z_autozero = (
            pd.Series(z_autozero)
            .interpolate()
            .bfill()
            .ffill()
            .values
        )

        # =====================
        # DETREND OPCIONAL
        # =====================

        if usar_detrend:

            x_proc = detrend(x_autozero)
            y_proc = detrend(y_autozero)
            z_proc = detrend(z_autozero)

        else:

            x_proc = x_autozero
            y_proc = y_autozero
            z_proc = z_autozero

        # =====================
        # NORMA EUCLIDIANA
        # =====================

        norma = np.sqrt(
            x_proc**2 +
            y_proc**2 +
            z_proc**2
        )

        normas[f"Pt{i}_norma"] = norma

        # =====================
        # RMS
        # =====================

        rms = np.sqrt(np.mean(norma**2))

        rms_pontos.append(rms)

        x_medios.append(np.nanmean(x_original))
        y_medios.append(np.nanmean(y_original))
        z_medios.append(np.nanmean(z_original))

    # =========================
    # DATAFRAME RMS
    # =========================

    rms_df = pd.DataFrame({

        "ponto":
            [f"Pt{i}" for i in range(n_pontos)],

        "x_medio":
            x_medios,

        "y_medio":
            y_medios,

        "z_medio":
            z_medios,

        "RMS_norma":
            rms_pontos
    })

    # =========================
    # COMPARAÇÃO NORMATIVA
    # =========================

    st.subheader(
        "Comparação com limites normativos"
    )

    classes_desvio = []
    descricoes_desvio = []

    for i in range(n_pontos):

        ponto_col = f"Pt{i}_norma"

        rms_valor = rms_pontos[i]

        if ponto_col not in parametros.columns:

            st.warning(
                f"{ponto_col} não encontrado "
                f"no arquivo normativo."
            )

            classes_desvio.append(0)
            descricoes_desvio.append("Sem referência")

            continue

        try:

            lim_2sd = float(
                parametros.loc[
                    parametros[level_col]
                    .astype(str)
                    .str.strip()
                    == "Mean + 2 SD",

                    ponto_col
                ].values[0]
            )

            lim_3sd = float(
                parametros.loc[
                    parametros[level_col]
                    .astype(str)
                    .str.strip()
                    == "Mean + 3 SD",

                    ponto_col
                ].values[0]
            )

            lim_4sd = float(
                parametros.loc[
                    parametros[level_col]
                    .astype(str)
                    .str.strip()
                    == "Mean + 4 SD",

                    ponto_col
                ].values[0]
            )

            lim_5sd = float(
                parametros.loc[
                    parametros[level_col]
                    .astype(str)
                    .str.strip()
                    == "Mean + 5 SD",

                    ponto_col
                ].values[0]
            )

        except:

            classes_desvio.append(0)
            descricoes_desvio.append("Erro")

            continue

        # =====================
        # CLASSIFICAÇÃO
        # =====================

        if rms_valor < lim_2sd:

            classe = 0
            descricao = "< 2 DP"

        elif rms_valor < lim_3sd:

            classe = 1
            descricao = "≥ 2 DP"

        elif rms_valor < lim_4sd:

            classe = 2
            descricao = "≥ 3 DP"

        elif rms_valor < lim_5sd:

            classe = 3
            descricao = "≥ 4 DP"

        else:

            classe = 4
            descricao = "≥ 5 DP"

        classes_desvio.append(classe)
        descricoes_desvio.append(descricao)

    rms_df["classe_desvio"] = classes_desvio

    rms_df["interpretação"] = descricoes_desvio

    # =========================
    # MÁSCARA
    # =========================

    st.subheader(
        "Máscara colorida por desvio"
    )

    mostrar_numeros = st.checkbox(
        "Mostrar número dos pontos",
        value=False
    )

    texto_pontos = (
        [str(i) for i in range(n_pontos)]
        if mostrar_numeros
        else None
    )

    modo = (
        "markers+text"
        if mostrar_numeros
        else "markers"
    )

    corescale_custom = [

        [0.00, "blue"],
        [0.25, "green"],
        [0.50, "yellow"],
        [0.75, "orange"],
        [1.00, "red"]
    ]

    fig_mask = go.Figure()

    fig_mask.add_trace(

        go.Scatter(

            x=rms_df["x_medio"],

            y=rms_df["y_medio"],

            mode=modo,

            marker=dict(

                size=10,

                color=rms_df["classe_desvio"],

                cmin=0,
                cmax=4,

                colorscale=corescale_custom,

                colorbar=dict(

                    title="Desvio",

                    tickvals=[0,1,2,3,4],

                    ticktext=[
                        "<2DP",
                        "≥2DP",
                        "≥3DP",
                        "≥4DP",
                        "≥5DP"
                    ]
                ),

                showscale=True
            ),

            text=texto_pontos,

            textposition="top center",

            customdata=np.stack(
                [
                    rms_df["ponto"],
                    rms_df["RMS_norma"],
                    rms_df["interpretação"]
                ],
                axis=-1
            ),

            hovertemplate=(
                "Ponto: %{customdata[0]}<br>"
                "RMS: %{customdata[1]:.6f}<br>"
                "Classe: %{customdata[2]}<br>"
                "X: %{x:.4f}<br>"
                "Y: %{y:.4f}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_mask.update_layout(

        height=750,

        xaxis_title="X médio",

        yaxis_title="Y médio",

        yaxis=dict(
            autorange="reversed"
        ),

        margin=dict(
            l=40,
            r=20,
            t=40,
            b=40
        )
    )

    fig_mask.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    st.plotly_chart(
        fig_mask,
        use_container_width=True
    )

    # =========================
    # RESUMO
    # =========================

    st.subheader(
        "Resumo das classes"
    )

    resumo_classes = (

        rms_df["interpretação"]

        .value_counts()

        .reset_index()
    )

    resumo_classes.columns = [
        "Classe",
        "Número de pontos"
    ]

    st.dataframe(resumo_classes)

    # =========================
    # VISUALIZAÇÃO TEMPORAL
    # =========================

    st.subheader(
        "Visualização temporal"
    )

    pontos = [
        f"Pt{i}_norma"
        for i in range(n_pontos)
    ]

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

    eixo_x = (
        tempo_col
        if usar_tempo == "Tempo"
        else frame_col
    )

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

        if usar_detrend:

            y_label = (
                "Norma após autozero + detrend"
            )

        else:

            y_label = (
                "Norma após autozero"
            )

        fig.update_layout(

            height=650,

            xaxis_title=eixo_x,

            yaxis_title=y_label,

            margin=dict(
                l=40,
                r=20,
                t=40,
                b=40
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =========================
    # TABELAS
    # =========================

    st.subheader(
        "Ranking dos pontos"
    )

    rms_df_ordenado = (
        rms_df
        .sort_values(
            "RMS_norma",
            ascending=False
        )
    )

    st.dataframe(rms_df_ordenado)

    st.subheader(
        "Tabela das normas"
    )

    st.dataframe(normas)

    # =========================
    # DOWNLOADS
    # =========================

    csv_normas = (
        normas
        .to_csv(index=False)
        .encode("utf-8")
    )

    csv_rms = (
        rms_df_ordenado
        .to_csv(index=False)
        .encode("utf-8")
    )

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(

            label="Baixar normas",

            data=csv_normas,

            file_name="normas.csv",

            mime="text/csv"
        )

    with col2:

        st.download_button(

            label="Baixar RMS",

            data=csv_rms,

            file_name="rms.csv",

            mime="text/csv"
        )

else:

    st.info(
        "Carregue o arquivo principal "
        "e o arquivo normativo."
    )
