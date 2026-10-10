import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import detrend

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Análise Facial com Procrustes e Eventos Transitórios",
    layout="wide"
)

st.title("Análise dos deslocamentos faciais 2D")
st.caption(
    "Alinhamento de Procrustes + análise normativa sustentada + "
    "detecção de contrações transitórias/tiques."
)

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def ler_csv_flexivel(arquivo):
    """Lê CSV/TXT tentando inferir separador e removendo BOM/espaços."""
    df = pd.read_csv(
        arquivo,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )
    return df


def interpolar_coluna(valores):
    """Interpola NaNs e preenche extremidades."""
    s = pd.Series(valores, dtype=float)
    return (
        s.interpolate(limit_direction="both")
        .bfill()
        .ffill()
        .to_numpy()
    )


def alinhar_similaridade_2d(configuracao, referencia, indices_ancora):
    """
    Alinha uma configuração 2D à referência removendo:
    - translação
    - rotação
    - escala isotrópica

    Retorna:
    - configuração alinhada
    - fator de escala
    """
    movel = configuracao[indices_ancora]
    fixa = referencia[indices_ancora]

    centro_movel = np.mean(movel, axis=0)
    centro_fixa = np.mean(fixa, axis=0)

    movel_c = movel - centro_movel
    fixa_c = fixa - centro_fixa

    norma_movel = np.linalg.norm(movel_c)
    norma_fixa = np.linalg.norm(fixa_c)

    if norma_movel <= np.finfo(float).eps or norma_fixa <= np.finfo(float).eps:
        return configuracao - centro_movel + centro_fixa, np.nan

    h = movel_c.T @ fixa_c
    u, _, vt = np.linalg.svd(h)
    rotacao = u @ vt

    # Evita reflexão.
    if np.linalg.det(rotacao) < 0:
        vt[-1, :] *= -1
        rotacao = u @ vt

    movel_rot = movel_c @ rotacao

    denominador = np.sum(movel_c ** 2)
    if denominador <= np.finfo(float).eps:
        escala = np.nan
        alinhada = configuracao - centro_movel + centro_fixa
    else:
        escala = np.sum(movel_rot * fixa_c) / denominador
        alinhada = (
            (configuracao - centro_movel)
            @ rotacao
            * escala
            + centro_fixa
        )

    return alinhada, escala


def linha_normativa(parametros, level_col, rotulo, ponto_col):
    """Busca um valor normativo específico."""
    mask = (
        parametros[level_col]
        .astype(str)
        .str.strip()
        .str.lower()
        == rotulo.strip().lower()
    )

    valores = parametros.loc[mask, ponto_col]

    if len(valores) == 0:
        return np.nan

    return pd.to_numeric(valores.iloc[0], errors="coerce")


def construir_parametros_por_ponto(parametros, level_col, n_pontos):
    """
    Extrai média e níveis normativos de cada landmark.
    Também estima SD a partir de Mean + 2 SD.
    """
    linhas = []

    for i in range(n_pontos):
        ponto_col = f"Pt{i}_norma"

        if ponto_col not in parametros.columns:
            linhas.append({
                "ponto": i,
                "media": np.nan,
                "sd": np.nan,
                "lim_2sd": np.nan,
                "lim_3sd": np.nan,
                "lim_4sd": np.nan,
                "lim_5sd": np.nan,
            })
            continue

        media = linha_normativa(parametros, level_col, "Mean", ponto_col)
        lim_2 = linha_normativa(parametros, level_col, "Mean + 2 SD", ponto_col)
        lim_3 = linha_normativa(parametros, level_col, "Mean + 3 SD", ponto_col)
        lim_4 = linha_normativa(parametros, level_col, "Mean + 4 SD", ponto_col)
        lim_5 = linha_normativa(parametros, level_col, "Mean + 5 SD", ponto_col)

        if np.isfinite(media) and np.isfinite(lim_2) and lim_2 > media:
            sd = (lim_2 - media) / 2.0
        else:
            sd = np.nan

        linhas.append({
            "ponto": i,
            "media": media,
            "sd": sd,
            "lim_2sd": lim_2,
            "lim_3sd": lim_3,
            "lim_4sd": lim_4,
            "lim_5sd": lim_5,
        })

    return pd.DataFrame(linhas)


def classe_por_valor(valor, lim2, lim3, lim4, lim5):
    """
    0: < 2DP
    1: 2-3DP
    2: 3-4DP
    3: 4-5DP
    4: >=5DP
    """
    if not np.isfinite(valor):
        return 0

    if np.isfinite(lim5) and valor >= lim5:
        return 4
    if np.isfinite(lim4) and valor >= lim4:
        return 3
    if np.isfinite(lim3) and valor >= lim3:
        return 2
    if np.isfinite(lim2) and valor >= lim2:
        return 1

    return 0


def fechar_pequenos_gaps(mask, max_gap=0):
    """
    Preenche gaps False curtos entre duas regiões True.
    Ex.: True True False True True com max_gap=1 vira tudo True.
    """
    mask = np.asarray(mask, dtype=bool).copy()

    if max_gap <= 0 or mask.size == 0:
        return mask

    i = 0
    n = len(mask)

    while i < n:
        if mask[i]:
            i += 1
            continue

        inicio = i
        while i < n and not mask[i]:
            i += 1
        fim = i
        tamanho_gap = fim - inicio

        tem_true_antes = inicio > 0 and mask[inicio - 1]
        tem_true_depois = fim < n and mask[fim]

        if tem_true_antes and tem_true_depois and tamanho_gap <= max_gap:
            mask[inicio:fim] = True

    return mask


def encontrar_eventos(mask, min_frames=2):
    """
    Identifica sequências contínuas True com duração >= min_frames.
    Retorna pares (início, fim), ambos inclusivos.
    """
    mask = np.asarray(mask, dtype=bool)
    eventos = []

    i = 0
    n = len(mask)

    while i < n:
        if not mask[i]:
            i += 1
            continue

        inicio = i
        while i < n and mask[i]:
            i += 1
        fim = i - 1

        if fim - inicio + 1 >= min_frames:
            eventos.append((inicio, fim))

    return eventos


def vizinhos_k_mais_proximos(xy, k=6):
    """
    Para cada landmark, retorna os k landmarks mais próximos
    na geometria facial de referência.
    """
    xy = np.asarray(xy, dtype=float)
    n = len(xy)

    if n <= 1:
        return [np.array([], dtype=int) for _ in range(n)]

    k = max(1, min(int(k), n - 1))

    dif = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(dif ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)

    return [
        np.argsort(dist[i])[:k]
        for i in range(n)
    ]


def tempo_duracao_evento(tempo, inicio, fim):
    """
    Calcula duração aproximada do evento.
    Usa o dt mediano quando possível para incluir o último frame.
    """
    if len(tempo) < 2:
        return np.nan

    dt = np.nanmedian(np.diff(tempo))

    if not np.isfinite(dt):
        return np.nan

    return (tempo[fim] - tempo[inicio]) + dt


# ============================================================
# ENTRADA DOS DADOS
# ============================================================

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    arquivo = st.file_uploader(
        "Arquivo principal (CSV/TXT)",
        type=["csv", "txt"]
    )

with col_upload2:
    fonte_parametros = st.radio(
        "Fonte dos níveis normativos",
        [
            "Usar Parameters.csv do repositório",
            "Carregar arquivo normativo"
        ],
        horizontal=False
    )

    arquivo_parametros_upload = None

    if fonte_parametros == "Carregar arquivo normativo":
        arquivo_parametros_upload = st.file_uploader(
            "Arquivo normativo (CSV/TXT)",
            type=["csv", "txt"],
            key="parametros_upload"
        )


if arquivo is None:
    st.info("Carregue o arquivo principal para iniciar.")
    st.stop()


# ============================================================
# LEITURA
# ============================================================

try:
    df = ler_csv_flexivel(arquivo)
except Exception as e:
    st.error(f"Não foi possível ler o arquivo principal: {e}")
    st.stop()

try:
    if fonte_parametros == "Usar Parameters.csv do repositório":
        parametros = ler_csv_flexivel("Parameters.csv")
    else:
        if arquivo_parametros_upload is None:
            st.info("Carregue o arquivo normativo.")
            st.stop()
        parametros = ler_csv_flexivel(arquivo_parametros_upload)
except Exception as e:
    st.error(f"Não foi possível ler o arquivo normativo: {e}")
    st.stop()


# ============================================================
# VALIDAÇÃO
# ============================================================

if df.shape[1] < 5:
    st.error("O arquivo principal possui poucas colunas.")
    st.stop()

level_candidates = [
    col for col in parametros.columns
    if col.lower().strip() == "level"
]

if not level_candidates:
    st.error(
        "Coluna 'Level' não encontrada no arquivo normativo. "
        f"Colunas detectadas: {parametros.columns.tolist()[:20]}"
    )
    st.stop()

level_col = level_candidates[0]

frame_col = df.columns[0]
tempo_col = df.columns[1]

df[tempo_col] = pd.to_numeric(df[tempo_col], errors="coerce")
df = df.dropna(subset=[tempo_col]).reset_index(drop=True)

tempo = df[tempo_col].astype(float).to_numpy()

coord_cols = df.columns[2:]

if len(coord_cols) % 3 != 0:
    st.error(
        "O número de colunas de coordenadas após Frame e Tempo "
        "não é múltiplo de 3."
    )
    st.stop()

n_pontos = len(coord_cols) // 3

st.success(f"Foram identificados {n_pontos} pontos tridimensionais.")


# ============================================================
# CONFIGURAÇÃO DO PROCESSAMENTO
# ============================================================

st.sidebar.header("Processamento geométrico")

usar_procrustes = st.sidebar.checkbox(
    "Remover translação, rotação e escala global (Procrustes)",
    value=True,
    help=(
        "Alinha cada frame a uma face de referência e reduz "
        "movimentos rígidos da cabeça e efeito da distância face-câmera."
    )
)

unidade_tempo = st.sidebar.selectbox(
    "Unidade da coluna de tempo",
    ["segundos", "milissegundos"],
    index=0
)

janela_ms = st.sidebar.number_input(
    "Janela inicial de referência (ms)",
    min_value=100,
    value=1000,
    step=100
)

usar_detrend = st.sidebar.checkbox(
    "Aplicar detrend após o alinhamento",
    value=False
)

pontos_ancora_texto = st.sidebar.text_input(
    "Landmarks âncora (opcional)",
    value="",
    help=(
        "Índices separados por vírgula. Deixe vazio para usar todos. "
        "Para expressões faciais, landmarks estáveis podem melhorar "
        "o alinhamento."
    )
)


# ============================================================
# CONFIGURAÇÃO DOS EVENTOS TRANSITÓRIOS
# ============================================================

st.sidebar.header("Eventos transitórios")

limiar_evento_dp = st.sidebar.selectbox(
    "Limiar para iniciar evento",
    [2.0, 3.0, 4.0, 5.0],
    index=1,
    help="Sugestão inicial: 3 DP."
)

min_frames_evento = st.sidebar.number_input(
    "Mínimo de frames consecutivos",
    min_value=1,
    max_value=30,
    value=2,
    step=1
)

max_gap_frames = st.sidebar.number_input(
    "Gap máximo permitido dentro do evento (frames)",
    min_value=0,
    max_value=5,
    value=1,
    step=1
)

confirmar_espacialmente = st.sidebar.checkbox(
    "Exigir confirmação espacial",
    value=True,
    help=(
        "Um evento só é confirmado quando landmarks próximos "
        "também ultrapassam o limiar em algum momento do episódio."
    )
)

k_vizinhos = st.sidebar.number_input(
    "Número de vizinhos anatômicos aproximados",
    min_value=1,
    max_value=20,
    value=6,
    step=1,
    disabled=not confirmar_espacialmente
)

min_vizinhos_ativos = st.sidebar.number_input(
    "Mínimo de vizinhos simultaneamente ativos",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    disabled=not confirmar_espacialmente
)


# ============================================================
# CONSTRUÇÃO DAS COORDENADAS
# ============================================================

coordenadas = np.empty(
    (len(df), n_pontos, 3),
    dtype=float
)

for i in range(n_pontos):
    for eixo in range(3):
        valores = pd.to_numeric(
            df.iloc[:, 2 + 3 * i + eixo],
            errors="coerce"
        ).to_numpy()

        coordenadas[:, i, eixo] = interpolar_coluna(valores)


# ============================================================
# LANDMARKS ÂNCORA
# ============================================================

try:
    if pontos_ancora_texto.strip():
        indices_ancora = sorted({
            int(v.strip())
            for v in pontos_ancora_texto.split(",")
            if v.strip()
        })

        if len(indices_ancora) < 3:
            raise ValueError(
                "Informe pelo menos três landmarks de alinhamento."
            )

        if min(indices_ancora) < 0 or max(indices_ancora) >= n_pontos:
            raise ValueError(
                f"Os índices devem estar entre 0 e {n_pontos - 1}."
            )
    else:
        indices_ancora = list(range(n_pontos))

except ValueError as e:
    st.error(f"Landmarks de alinhamento inválidos: {e}")
    st.stop()


# ============================================================
# REFERÊNCIA FACIAL
# ============================================================

janela_referencia = (
    janela_ms / 1000.0
    if unidade_tempo == "segundos"
    else float(janela_ms)
)

tempo_inicial = tempo[0]
idx_referencia = tempo <= tempo_inicial + janela_referencia

if np.sum(idx_referencia) < 2:
    st.warning(
        "A janela de referência possui menos de 2 amostras. "
        "Serão usados os primeiros frames disponíveis."
    )
    idx_referencia = np.arange(min(5, len(tempo)))

referencia_xy = np.nanmean(
    coordenadas[idx_referencia, :, :2],
    axis=0
)

centro_ref = np.mean(
    referencia_xy[indices_ancora],
    axis=0
)

referencia_xy = referencia_xy - centro_ref

tamanho_ref = np.linalg.norm(
    referencia_xy[indices_ancora]
)

if tamanho_ref <= np.finfo(float).eps:
    st.error("Não foi possível construir uma referência facial válida.")
    st.stop()

referencia_xy = referencia_xy / tamanho_ref


# ============================================================
# PROCRUSTES
# ============================================================

coordenadas_alinhadas = coordenadas.copy()
escalas = np.ones(len(df), dtype=float)

if usar_procrustes:
    for frame in range(len(df)):
        alinhada, escala = alinhar_similaridade_2d(
            coordenadas[frame, :, :2],
            referencia_xy,
            indices_ancora
        )

        coordenadas_alinhadas[frame, :, :2] = alinhada
        escalas[frame] = escala


# ============================================================
# DISTÂNCIA FRAME-A-FRAME
# ============================================================

series_dict = {
    frame_col: df[frame_col].to_numpy(),
    tempo_col: tempo,
    "fator_escala_procrustes": escalas
}

rms_pontos = []

for i in range(n_pontos):
    x_proc = coordenadas_alinhadas[:, i, 0].copy()
    y_proc = coordenadas_alinhadas[:, i, 1].copy()

    if usar_detrend and len(x_proc) > 1:
        x_proc = detrend(x_proc)
        y_proc = detrend(y_proc)

    dx = np.diff(x_proc, prepend=x_proc[0])
    dy = np.diff(y_proc, prepend=y_proc[0])

    distancia_delta = np.sqrt(dx ** 2 + dy ** 2)

    series_dict[f"Pt{i}_norma"] = distancia_delta
    rms_pontos.append(
        np.sqrt(np.mean(distancia_delta ** 2))
    )

normas = pd.DataFrame(series_dict)

# Coordenadas para visualização da máscara.
x_ref = referencia_xy[:, 0]
y_ref = referencia_xy[:, 1]
z_medio = np.nanmean(coordenadas[:, :, 2], axis=0)


# ============================================================
# PARÂMETROS NORMATIVOS
# ============================================================

param_pontos = construir_parametros_por_ponto(
    parametros,
    level_col,
    n_pontos
)

if param_pontos["lim_2sd"].isna().all():
    st.error(
        "Não foi possível localizar níveis 'Mean + 2 SD' "
        "no arquivo normativo."
    )
    st.stop()


# ============================================================
# MATRIZ DE Z NORMATIVO APROXIMADO
# ============================================================

z_matrix = np.full(
    (len(df), n_pontos),
    np.nan,
    dtype=float
)

for i in range(n_pontos):
    serie = normas[f"Pt{i}_norma"].to_numpy(dtype=float)
    media = param_pontos.loc[i, "media"]
    sd = param_pontos.loc[i, "sd"]

    if np.isfinite(media) and np.isfinite(sd) and sd > 0:
        z_matrix[:, i] = (serie - media) / sd


# ============================================================
# ANÁLISE SUSTENTADA
# ============================================================

linhas_resumo = []

for i in range(n_pontos):
    serie = normas[f"Pt{i}_norma"].to_numpy(dtype=float)

    lim2 = param_pontos.loc[i, "lim_2sd"]
    lim3 = param_pontos.loc[i, "lim_3sd"]
    lim4 = param_pontos.loc[i, "lim_4sd"]
    lim5 = param_pontos.loc[i, "lim_5sd"]

    validos = np.isfinite(serie)
    s = serie[validos]
    n_total = len(s)

    if n_total == 0 or not np.isfinite(lim2):
        linhas_resumo.append({
            "ponto": f"Pt{i}",
            "x_medio": x_ref[i],
            "y_medio": y_ref[i],
            "z_medio": z_medio[i],
            "RMS_distancia_delta": rms_pontos[i],
            "classe_sustentada": 0,
            "interpretacao_sustentada": "Sem referência",
            "n_<2DP": np.nan,
            "n_2a3DP": np.nan,
            "n_3a4DP": np.nan,
            "n_4a5DP": np.nan,
            "n_>=5DP": np.nan,
            "%_<2DP": np.nan,
            "%_2a3DP": np.nan,
            "%_3a4DP": np.nan,
            "%_4a5DP": np.nan,
            "%_>=5DP": np.nan,
        })
        continue

    n_menor_2 = np.sum(s < lim2)

    n_2a3 = (
        np.sum((s >= lim2) & (s < lim3))
        if np.isfinite(lim3)
        else 0
    )

    n_3a4 = (
        np.sum((s >= lim3) & (s < lim4))
        if np.isfinite(lim3) and np.isfinite(lim4)
        else 0
    )

    n_4a5 = (
        np.sum((s >= lim4) & (s < lim5))
        if np.isfinite(lim4) and np.isfinite(lim5)
        else 0
    )

    n_maior_5 = (
        np.sum(s >= lim5)
        if np.isfinite(lim5)
        else 0
    )

    contagens_anormais = [
        n_2a3,
        n_3a4,
        n_4a5,
        n_maior_5
    ]

    if np.sum(contagens_anormais) == 0:
        classe_sust = 0
    else:
        classe_sust = int(
            np.argmax(contagens_anormais)
        ) + 1

    descricoes = [
        "Basal: nenhum valor ≥ 2 DP",
        "2–3 DP dominante",
        "3–4 DP dominante",
        "4–5 DP dominante",
        "≥ 5 DP dominante"
    ]

    linhas_resumo.append({
        "ponto": f"Pt{i}",
        "x_medio": x_ref[i],
        "y_medio": y_ref[i],
        "z_medio": z_medio[i],
        "RMS_distancia_delta": rms_pontos[i],
        "classe_sustentada": classe_sust,
        "interpretacao_sustentada": descricoes[classe_sust],
        "n_<2DP": n_menor_2,
        "n_2a3DP": n_2a3,
        "n_3a4DP": n_3a4,
        "n_4a5DP": n_4a5,
        "n_>=5DP": n_maior_5,
        "%_<2DP": 100 * n_menor_2 / n_total,
        "%_2a3DP": 100 * n_2a3 / n_total,
        "%_3a4DP": 100 * n_3a4 / n_total,
        "%_4a5DP": 100 * n_4a5 / n_total,
        "%_>=5DP": 100 * n_maior_5 / n_total,
    })

resumo_df = pd.DataFrame(linhas_resumo)


# ============================================================
# DETECÇÃO DE EVENTOS TRANSITÓRIOS
# ============================================================

vizinhos = vizinhos_k_mais_proximos(
    referencia_xy,
    k=int(k_vizinhos)
)

# Matriz booleana de atividade no limiar escolhido.
ativo_limiar = (
    np.isfinite(z_matrix)
    & (z_matrix >= float(limiar_evento_dp))
)

eventos_lista = []
resumo_eventos = []

for i in range(n_pontos):
    serie = normas[f"Pt{i}_norma"].to_numpy(dtype=float)
    zserie = z_matrix[:, i]

    mask = ativo_limiar[:, i]
    mask = fechar_pequenos_gaps(
        mask,
        max_gap=int(max_gap_frames)
    )

    eventos_brutos = encontrar_eventos(
        mask,
        min_frames=int(min_frames_evento)
    )

    eventos_confirmados = []

    for inicio, fim in eventos_brutos:
        confirmar = True
        max_vizinhos_simultaneos = 0

        if confirmar_espacialmente:
            idx_vizinhos = vizinhos[i]

            if len(idx_vizinhos) == 0:
                confirmar = False
            else:
                contagem_por_frame = np.sum(
                    ativo_limiar[
                        inicio:fim + 1,
                        :
                    ][:, idx_vizinhos],
                    axis=1
                )

                max_vizinhos_simultaneos = int(
                    np.max(contagem_por_frame)
                ) if len(contagem_por_frame) else 0

                confirmar = (
                    max_vizinhos_simultaneos
                    >= int(min_vizinhos_ativos)
                )

        if not confirmar:
            continue

        segmento = serie[inicio:fim + 1]
        zsegmento = zserie[inicio:fim + 1]

        if np.all(~np.isfinite(segmento)):
            continue

        idx_local_pico = int(
            np.nanargmax(segmento)
        )
        idx_pico = inicio + idx_local_pico

        pico_valor = serie[idx_pico]
        pico_z = (
            zserie[idx_pico]
            if np.isfinite(zserie[idx_pico])
            else np.nan
        )

        lim2 = param_pontos.loc[i, "lim_2sd"]
        lim3 = param_pontos.loc[i, "lim_3sd"]
        lim4 = param_pontos.loc[i, "lim_4sd"]
        lim5 = param_pontos.loc[i, "lim_5sd"]

        classe_pico = classe_por_valor(
            pico_valor,
            lim2,
            lim3,
            lim4,
            lim5
        )

        # Área acima do limiar normativo selecionado em unidades de DP*tempo.
        excesso_z = np.maximum(
            zsegmento - float(limiar_evento_dp),
            0
        )

        if len(tempo) > 1:
            tempo_seg = (
                tempo / 1000.0
                if unidade_tempo == "milissegundos"
                else tempo
            )

            area_excesso = np.trapezoid(
                np.nan_to_num(excesso_z, nan=0.0),
                x=tempo_seg[inicio:fim + 1]
            ) if fim > inicio else 0.0

            duracao = tempo_duracao_evento(
                tempo_seg,
                inicio,
                fim
            )
        else:
            area_excesso = np.nan
            duracao = np.nan

        evento = {
            "ponto": f"Pt{i}",
            "indice_ponto": i,
            "frame_inicio_idx": inicio,
            "frame_fim_idx": fim,
            "frame_pico_idx": idx_pico,
            "tempo_inicio": tempo[inicio],
            "tempo_fim": tempo[fim],
            "tempo_pico": tempo[idx_pico],
            "duracao": duracao,
            "n_frames": fim - inicio + 1,
            "pico_distancia": pico_valor,
            "pico_z_aprox": pico_z,
            "classe_pico": classe_pico,
            "area_excesso_DP_tempo": area_excesso,
            "max_vizinhos_simultaneos": max_vizinhos_simultaneos,
        }

        eventos_lista.append(evento)
        eventos_confirmados.append(evento)

    if eventos_confirmados:
        evento_max = max(
            eventos_confirmados,
            key=lambda e: (
                e["classe_pico"],
                e["pico_z_aprox"]
                if np.isfinite(e["pico_z_aprox"])
                else -np.inf
            )
        )

        classe_transitoria = int(
            evento_max["classe_pico"]
        )

        maior_pico_z = np.nanmax([
            e["pico_z_aprox"]
            for e in eventos_confirmados
        ])

        duracao_max = np.nanmax([
            e["duracao"]
            for e in eventos_confirmados
        ])

        area_total = np.nansum([
            e["area_excesso_DP_tempo"]
            for e in eventos_confirmados
        ])

        tempo_primeiro = min(
            e["tempo_inicio"]
            for e in eventos_confirmados
        )

    else:
        classe_transitoria = 0
        maior_pico_z = np.nan
        duracao_max = np.nan
        area_total = 0.0
        tempo_primeiro = np.nan

    resumo_eventos.append({
        "ponto": f"Pt{i}",
        "n_eventos_transitorios": len(eventos_confirmados),
        "classe_transitoria": classe_transitoria,
        "maior_pico_z_aprox": maior_pico_z,
        "duracao_maior_evento": duracao_max,
        "area_total_excesso_DP_tempo": area_total,
        "tempo_primeiro_evento": tempo_primeiro
    })

eventos_df = pd.DataFrame(eventos_lista)
resumo_eventos_df = pd.DataFrame(resumo_eventos)

resumo_df = resumo_df.merge(
    resumo_eventos_df,
    on="ponto",
    how="left"
)


# ============================================================
# LEGENDA / PALETA
# ============================================================

corescale_custom = [
    [0.00, "blue"],
    [0.25, "green"],
    [0.50, "yellow"],
    [0.75, "orange"],
    [1.00, "red"]
]

ticktext = [
    "Basal",
    "2–3 DP",
    "3–4 DP",
    "4–5 DP",
    "≥5 DP"
]


# ============================================================
# INFORMAÇÕES DO PROCESSAMENTO
# ============================================================

if usar_procrustes:
    st.info(
        "Os deslocamentos foram calculados após alinhamento de "
        "Procrustes 2D. O arquivo normativo deve ter sido produzido "
        "com o mesmo pipeline de processamento."
    )

col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.metric(
        "Landmarks",
        n_pontos
    )

with col_info2:
    st.metric(
        "Frames",
        len(df)
    )

with col_info3:
    st.metric(
        "Landmarks com evento",
        int(
            (resumo_df["n_eventos_transitorios"] > 0)
            .sum()
        )
    )

with col_info4:
    st.metric(
        "Eventos confirmados",
        len(eventos_df)
    )


# ============================================================
# OPÇÕES DE RÓTULO
# ============================================================

st.subheader("Configuração das máscaras")

col_rotulo1, col_rotulo2, col_rotulo3 = st.columns(3)

with col_rotulo1:
    mostrar_numeros = st.checkbox(
        "Mostrar número dos landmarks",
        value=False
    )

with col_rotulo2:
    mostrar_valores = st.checkbox(
        "Mostrar RMS ao lado dos landmarks",
        value=False
    )

with col_rotulo3:
    casas_decimais = st.number_input(
        "Casas decimais do RMS",
        min_value=2,
        max_value=8,
        value=5,
        step=1,
        disabled=not mostrar_valores
    )


def gerar_textos_pontos(df_resumo):
    textos = []

    for _, row in df_resumo.iterrows():
        partes = []

        if mostrar_numeros:
            partes.append(str(row["ponto"]))

        if mostrar_valores:
            partes.append(
                f'{row["RMS_distancia_delta"]:.{int(casas_decimais)}f}'
            )

        textos.append(" | ".join(partes))

    return (
        textos
        if (mostrar_numeros or mostrar_valores)
        else None
    )


texto_pontos = gerar_textos_pontos(resumo_df)

modo_pontos = (
    "markers+text"
    if texto_pontos is not None
    else "markers"
)


# ============================================================
# ABAS DAS MÁSCARAS
# ============================================================

tab_transitoria, tab_sustentada = st.tabs([
    "Contrações transitórias",
    "Alteração sustentada"
])


# ============================================================
# MÁSCARA TRANSITÓRIA
# ============================================================

with tab_transitoria:

    st.subheader(
        "Máscara de eventos transitórios"
    )

    st.write(
        "A cor representa a **maior intensidade normativa atingida "
        "por um evento transitório confirmado**, e não o estrato "
        "mais frequente durante toda a gravação."
    )

    fig_trans = go.Figure()

    custom_trans = np.column_stack([
        resumo_df["ponto"].astype(str),
        resumo_df["RMS_distancia_delta"].astype(float),
        resumo_df["n_eventos_transitorios"].astype(float),
        resumo_df["maior_pico_z_aprox"].astype(float),
        resumo_df["duracao_maior_evento"].astype(float),
        resumo_df["area_total_excesso_DP_tempo"].astype(float),
    ])

    fig_trans.add_trace(
        go.Scatter(
            x=resumo_df["x_medio"],
            y=resumo_df["y_medio"],
            mode=modo_pontos,
            marker=dict(
                size=10,
                color=resumo_df["classe_transitoria"],
                cmin=0,
                cmax=4,
                colorscale=corescale_custom,
                colorbar=dict(
                    title="Maior evento",
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=ticktext
                ),
                showscale=True
            ),
            text=texto_pontos,
            textposition="top center",
            customdata=custom_trans,
            hovertemplate=(
                "Ponto: %{customdata[0]}<br>"
                "RMS: %{customdata[1]:.6f}<br>"
                "Eventos confirmados: %{customdata[2]:.0f}<br>"
                "Maior pico (DP aprox.): %{customdata[3]:.2f}<br>"
                "Maior duração: %{customdata[4]:.3f}<br>"
                "Área total acima do limiar: %{customdata[5]:.4f}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_trans.update_layout(
        height=750,
        xaxis_title="X de referência",
        yaxis_title="Y de referência",
        yaxis=dict(autorange="reversed"),
        margin=dict(
            l=40,
            r=20,
            t=40,
            b=40
        )
    )

    fig_trans.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    st.plotly_chart(
        fig_trans,
        width="stretch"
    )

    st.caption(
        f"Evento = ≥ {limiar_evento_dp:.0f} DP por pelo menos "
        f"{int(min_frames_evento)} frame(s), "
        f"gap máximo = {int(max_gap_frames)} frame(s). "
        + (
            f"Confirmação espacial: ≥ {int(min_vizinhos_ativos)} "
            f"entre os {int(k_vizinhos)} vizinhos mais próximos."
            if confirmar_espacialmente
            else "Sem confirmação espacial."
        )
    )


# ============================================================
# MÁSCARA SUSTENTADA
# ============================================================

with tab_sustentada:

    st.subheader(
        "Máscara de alteração sustentada"
    )

    st.write(
        "A cor representa o estrato **mais frequente entre as "
        "amostras anormais** do landmark, preservando a lógica "
        "do código original."
    )

    custom_sust = np.column_stack([
        resumo_df["ponto"].astype(str),
        resumo_df["RMS_distancia_delta"].astype(float),
        resumo_df["interpretacao_sustentada"].astype(str),
        resumo_df["%_<2DP"].astype(float),
        resumo_df["%_2a3DP"].astype(float),
        resumo_df["%_3a4DP"].astype(float),
        resumo_df["%_4a5DP"].astype(float),
        resumo_df["%_>=5DP"].astype(float)
    ])

    fig_sust = go.Figure()

    fig_sust.add_trace(
        go.Scatter(
            x=resumo_df["x_medio"],
            y=resumo_df["y_medio"],
            mode=modo_pontos,
            marker=dict(
                size=10,
                color=resumo_df["classe_sustentada"],
                cmin=0,
                cmax=4,
                colorscale=corescale_custom,
                colorbar=dict(
                    title="Classificação",
                    tickvals=[0, 1, 2, 3, 4],
                    ticktext=ticktext
                ),
                showscale=True
            ),
            text=texto_pontos,
            textposition="top center",
            customdata=custom_sust,
            hovertemplate=(
                "Ponto: %{customdata[0]}<br>"
                "RMS: %{customdata[1]:.6f}<br>"
                "Classificação: %{customdata[2]}<br>"
                "% <2DP: %{customdata[3]:.1f}%<br>"
                "% 2–3DP: %{customdata[4]:.1f}%<br>"
                "% 3–4DP: %{customdata[5]:.1f}%<br>"
                "% 4–5DP: %{customdata[6]:.1f}%<br>"
                "% ≥5DP: %{customdata[7]:.1f}%<br>"
                "<extra></extra>"
            )
        )
    )

    fig_sust.update_layout(
        height=750,
        xaxis_title="X de referência",
        yaxis_title="Y de referência",
        yaxis=dict(autorange="reversed"),
        margin=dict(
            l=40,
            r=20,
            t=40,
            b=40
        )
    )

    fig_sust.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    st.plotly_chart(
        fig_sust,
        width="stretch"
    )


# ============================================================
# TABELA DE EVENTOS
# ============================================================

st.subheader("Eventos transitórios detectados")

if eventos_df.empty:
    st.info(
        "Nenhum evento transitório foi confirmado com os critérios atuais."
    )
else:
    eventos_exibir = eventos_df.sort_values(
        [
            "classe_pico",
            "pico_z_aprox"
        ],
        ascending=[False, False]
    ).reset_index(drop=True)

    st.dataframe(
        eventos_exibir,
        width="stretch"
    )


# ============================================================
# VISUALIZAÇÃO TEMPORAL
# ============================================================

st.subheader("Visualização temporal")

pontos_disponiveis = [
    f"Pt{i}_norma"
    for i in range(n_pontos)
]

pontos_selecionados = st.multiselect(
    "Escolha os landmarks",
    pontos_disponiveis,
    default=["Pt0_norma"]
)

col_temporal1, col_temporal2, col_temporal3 = st.columns(3)

with col_temporal1:
    usar_tempo = st.radio(
        "Eixo X",
        ["Tempo", "Frame"],
        horizontal=True
    )

with col_temporal2:
    mostrar_linhas_niveis = st.checkbox(
        "Mostrar níveis normativos",
        value=True
    )

with col_temporal3:
    mostrar_eventos_temporais = st.checkbox(
        "Destacar eventos transitórios",
        value=True
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

    # --------------------------------------------------------
    # NÍVEIS NORMATIVOS
    # --------------------------------------------------------

    if mostrar_linhas_niveis:

        if len(pontos_selecionados) == 1:
            ponto_ref = pontos_selecionados[0]

            try:
                idx = int(
                    ponto_ref
                    .replace("Pt", "")
                    .replace("_norma", "")
                )

                valores_niveis = [
                    (
                        param_pontos.loc[idx, "lim_2sd"],
                        "Mean + 2 SD"
                    ),
                    (
                        param_pontos.loc[idx, "lim_3sd"],
                        "Mean + 3 SD"
                    ),
                    (
                        param_pontos.loc[idx, "lim_4sd"],
                        "Mean + 4 SD"
                    ),
                    (
                        param_pontos.loc[idx, "lim_5sd"],
                        "Mean + 5 SD"
                    ),
                ]

                for valor, nome in valores_niveis:
                    if np.isfinite(valor):
                        fig.add_hline(
                            y=float(valor),
                            line_dash="dash",
                            annotation_text=nome,
                            annotation_position="top right"
                        )

            except Exception as e:
                st.warning(
                    f"Não foi possível adicionar os níveis normativos: {e}"
                )

        else:
            st.info(
                "As linhas normativas são exibidas quando apenas "
                "um landmark está selecionado, porque cada landmark "
                "possui limites próprios."
            )

    # --------------------------------------------------------
    # EVENTOS TRANSITÓRIOS NO TEMPO
    # --------------------------------------------------------

    if mostrar_eventos_temporais and not eventos_df.empty:

        for ponto in pontos_selecionados:
            idx = int(
                ponto
                .replace("Pt", "")
                .replace("_norma", "")
            )

            evs = eventos_df[
                eventos_df["indice_ponto"] == idx
            ]

            for _, ev in evs.iterrows():

                if usar_tempo == "Tempo":
                    x0 = ev["tempo_inicio"]
                    x1 = ev["tempo_fim"]
                else:
                    inicio = int(ev["frame_inicio_idx"])
                    fim = int(ev["frame_fim_idx"])

                    x0 = normas.iloc[inicio][frame_col]
                    x1 = normas.iloc[fim][frame_col]

                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    opacity=0.12,
                    line_width=0
                )

    y_label = (
        "Distância euclidiana frame-a-frame após Procrustes"
        if usar_procrustes
        else "Distância euclidiana frame-a-frame"
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
        width="stretch"
    )


# ============================================================
# RESUMOS
# ============================================================

st.subheader("Resumo por landmark")

resumo_ordenado = resumo_df.sort_values(
    [
        "classe_transitoria",
        "n_eventos_transitorios",
        "maior_pico_z_aprox"
    ],
    ascending=[False, False, False]
).reset_index(drop=True)

st.dataframe(
    resumo_ordenado,
    width="stretch"
)


# ============================================================
# DOWNLOADS
# ============================================================

st.subheader("Downloads")

csv_series = normas.to_csv(
    index=False
).encode("utf-8")

csv_resumo = resumo_ordenado.to_csv(
    index=False
).encode("utf-8")

if eventos_df.empty:
    csv_eventos = pd.DataFrame(
        columns=[
            "ponto",
            "indice_ponto",
            "tempo_inicio",
            "tempo_fim",
            "tempo_pico",
            "duracao",
            "n_frames",
            "pico_distancia",
            "pico_z_aprox",
            "classe_pico",
            "area_excesso_DP_tempo",
            "max_vizinhos_simultaneos"
        ]
    ).to_csv(index=False).encode("utf-8")
else:
    csv_eventos = eventos_df.to_csv(
        index=False
    ).encode("utf-8")

col_down1, col_down2, col_down3 = st.columns(3)

with col_down1:
    st.download_button(
        label="Baixar séries temporais",
        data=csv_series,
        file_name="distancias_delta.csv",
        mime="text/csv"
    )

with col_down2:
    st.download_button(
        label="Baixar resumo por landmark",
        data=csv_resumo,
        file_name="resumo_landmarks.csv",
        mime="text/csv"
    )

with col_down3:
    st.download_button(
        label="Baixar eventos transitórios",
        data=csv_eventos,
        file_name="eventos_transitorios.csv",
        mime="text/csv"
    )


# ============================================================
# NOTAS METODOLÓGICAS
# ============================================================

with st.expander("Como a detecção transitória funciona"):

    st.markdown(
        f"""
### 1. Alinhamento geométrico
Cada frame é alinhado à face de referência por transformação de
similaridade 2D, removendo translação, rotação e escala global.

### 2. Movimento local
Para cada landmark é calculada a distância euclidiana frame-a-frame:

\[
d_i(t)=\sqrt{{\Delta x_i(t)^2+\Delta y_i(t)^2}}
\]

### 3. Padronização normativa aproximada
O desvio-padrão é reconstruído a partir dos valores de
**Mean** e **Mean + 2 SD** do arquivo normativo:

\[
SD_i \\approx \\frac{{(Mean+2SD)_i-Mean_i}}{{2}}
\]

e:

\[
Z_i(t)=\\frac{{d_i(t)-Mean_i}}{{SD_i}}
\]

### 4. Evento candidato
Um evento começa quando o landmark fica em pelo menos
**{limiar_evento_dp:.0f} DP** durante pelo menos
**{int(min_frames_evento)} frame(s)**.

Gaps de até **{int(max_gap_frames)} frame(s)** podem ser fechados para
não fragmentar um único evento breve.

### 5. Confirmação espacial
{"É exigido que landmarks vizinhos também ultrapassem o limiar durante o episódio." if confirmar_espacialmente else "A confirmação espacial está desativada."}

### 6. Cor da máscara transitória
A cor é definida pelo **maior estrato normativo atingido pelo pico
de um evento confirmado**:

- azul: nenhum evento confirmado;
- verde: pico entre 2 e 3 DP;
- amarelo: pico entre 3 e 4 DP;
- laranja: pico entre 4 e 5 DP;
- vermelho: pico ≥ 5 DP.

Assim, um tique breve não é diluído pela longa duração de períodos
sem movimento.
"""
    )
