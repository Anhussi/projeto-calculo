"""
Painel SIR — interface Streamlit
================================
Três seções no menu esquerdo: Simulação, Surto e Ajuda.
Espaçamento uniforme, logs em PDF e modo interativo estilo Plague Inc.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import comparar_metodos, r0_basico, simular_sir
from relatorio import pdf_laboratorio, pdf_surto
from surto import (
    PESOS,
    TRAITS,
    comprar_trait,
    iniciar_em,
    mundo_from_session,
    mundo_novo,
    mundo_to_session,
    resumo as resumo_surto,
    tick,
)

# ---------------------------------------------------------------------------
# Tema
# ---------------------------------------------------------------------------

C_BG = "#0B1220"
C_PANEL = "#121A2B"
C_PANEL2 = "#182236"
C_LINE = "rgba(255,255,255,0.10)"
C_TEXT = "#E8EEF6"
C_MUTED = "#8B9BB4"
C_TEAL = "#2EC4B6"
C_TEAL_DIM = "#1A8F85"
C_INF = "#E85D4C"
C_REC = "#5B8DEF"
C_S = "#2EC4B6"
C_WARN = "#E9C46A"
GAP = "1.1rem"

PRESETS = {
    "contida": dict(N=100_000, I0=80, beta=0.14, gamma=0.22, t_final=120, h=0.25),
    "gripe": dict(N=100_000, I0=50, beta=0.36, gamma=0.20, t_final=160, h=0.25),
    "sarampo": dict(N=100_000, I0=20, beta=1.80, gamma=0.14, t_final=80, h=0.1),
}

st.set_page_config(
    page_title="Painel SIR",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {C_TEXT};
}}
.stApp {{
    background: radial-gradient(1200px 560px at 0% 0%, #132033 0%, {C_BG} 52%);
}}
#MainMenu, footer, div[data-testid="stAppToolbar"],
[data-testid="stToolbarActions"], .stDeployButton,
[data-testid="stStatusWidget"], [data-testid="stHeaderActionElements"],
div[data-testid="stDecoration"] {{
    visibility: hidden; display: none;
}}
/* Cabeçalho some com a barra aberta; com ela fechada, sobra só o botão Menu. */
header[data-testid="stHeader"], [data-testid="stAppHeader"] {{
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    border: none !important;
    overflow: visible !important;
}}
header[data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"]),
[data-testid="stAppHeader"]:has([data-testid="stExpandSidebarButton"]) {{
    height: 3.4rem !important;
    min-height: 3.4rem !important;
}}
header [data-testid="stToolbar"],
[data-testid="stHeader"] [data-testid="stToolbar"],
div[data-testid="stToolbar"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
    height: auto !important;
    overflow: visible !important;
}}
[data-testid="stSidebarCollapseButton"] {{
    visibility: visible !important;
    opacity: 1 !important;
    display: inline-flex !important;
    color: {C_TEAL} !important;
}}
[data-testid="stExpandSidebarButton"] {{
    visibility: visible !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.4rem !important;
    position: fixed !important;
    top: 0.85rem !important;
    left: 0.85rem !important;
    z-index: 100000 !important;
    width: auto !important;
    min-width: 2.5rem !important;
    height: 2.5rem !important;
    padding: 0 0.85rem 0 0.7rem !important;
    background: {C_PANEL2} !important;
    color: {C_TEAL} !important;
    border: 1px solid {C_TEAL} !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}}
[data-testid="stExpandSidebarButton"]::after {{
    content: "Menu";
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: {C_TEAL};
}}

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stSidebar"],
[data-testid="stSidebarContent"], [data-testid="stDataFrame"] {{
    scrollbar-width: thin;
    scrollbar-color: {C_TEAL} {C_PANEL};
}}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: {C_BG}; }}
::-webkit-scrollbar-thumb {{
    background: {C_TEAL};
    border-radius: 8px;
    border: 2px solid {C_BG};
}}
[data-testid="stSidebar"] ::-webkit-scrollbar-track {{ background: {C_PANEL}; }}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{ border-color: {C_PANEL}; }}

[data-testid="stSidebar"] {{
    background: {C_PANEL};
    border-right: 1px solid {C_LINE};
}}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    padding: 1rem 0.95rem 1.4rem 0.95rem;
}}
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.75rem !important;
}}
[data-testid="stSidebar"] label {{ color: {C_TEXT} !important; }}

.block-container {{
    padding: 1.35rem 1.7rem 2rem 1.7rem !important;
    max-width: 1280px;
}}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] {{
    gap: {GAP} !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
    padding: 0 !important;
}}
[data-testid="stHorizontalBlock"] {{
    gap: {GAP} !important;
    align-items: stretch !important;
}}
[data-testid="column"] {{
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}}
[data-testid="column"] > div {{
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}}
[data-testid="column"] [data-testid="stVerticalBlock"] {{
    gap: {GAP} !important;
}}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {{
    margin: 0;
}}
[data-testid="stElementContainer"] {{
    margin: 0 !important;
}}
[data-testid="stMarkdownContainer"] {{
    width: 100%;
}}
div[data-testid="stAlert"] {{
    padding: 0.9rem 1.05rem !important;
}}
[data-testid="stPlotlyChart"] {{
    margin: 0 !important;
}}
div[data-testid="stSlider"] {{
    padding-top: 0.15rem !important;
    padding-bottom: 0.15rem !important;
}}
div[data-testid="stNumberInput"] {{
    padding-bottom: 0 !important;
}}
[data-testid="stMetric"] {{
    background: {C_PANEL};
    border: 1px solid {C_LINE};
    border-radius: 12px;
    padding: 0.85rem 1rem;
}}

.kicker {{
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.16em;
    text-transform: uppercase; color: {C_TEAL}; margin: 0 0 0.4rem;
}}
.title {{
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.85rem; font-weight: 600; letter-spacing: -0.02em;
    margin: 0 0 0.45rem; color: {C_TEXT};
}}
.lede {{
    color: {C_MUTED}; font-size: 0.95rem; line-height: 1.5;
    margin: 0; max-width: 42rem;
}}
.stack {{
    display: flex;
    flex-direction: column;
    gap: {GAP};
}}
.eq-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: {GAP};
    align-items: stretch;
    flex: 1;
}}
.eq-row > .eq {{ height: 100%; }}
@media (max-width: 1100px) {{
    .eq-row {{ grid-template-columns: 1fr; }}
}}

.kpi-row, .insp-nums, .help-grid {{
    display: grid;
    gap: {GAP};
    align-items: stretch;
    width: 100%;
}}
.kpi-row {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.insp-nums {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.help-grid {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
}}
.help-grid .span2 {{ grid-column: 1 / -1; }}
@media (max-width: 1100px) {{
    .kpi-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .insp-nums, .help-grid {{ grid-template-columns: 1fr; }}
    .help-grid .span2 {{ grid-column: auto; }}
}}

.card {{
    background: {C_PANEL};
    border: 1px solid {C_LINE};
    border-radius: 12px;
    padding: 1.15rem 1.2rem;
    min-height: 108px;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
}}
.card .lbl {{
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: {C_MUTED}; margin: 0;
}}
.card .val {{
    font-family: 'IBM Plex Serif', serif; font-size: 1.7rem; font-weight: 600;
    line-height: 1.15; margin: 0;
}}
.card .hint {{ color: {C_MUTED}; font-size: 0.78rem; margin: 0; }}
.card p {{ color: {C_MUTED}; font-size: 0.92rem; line-height: 1.55; margin: 0; }}
.card h3 {{
    font-family: 'IBM Plex Serif', serif; font-size: 1.08rem;
    margin: 0; color: {C_TEXT};
}}
.eq {{
    background: {C_PANEL2};
    border: 1px solid {C_LINE};
    border-radius: 10px;
    padding: 0.95rem 1.1rem;
    font-family: 'IBM Plex Sans', ui-monospace, monospace;
    font-size: 0.95rem;
    line-height: 1.75;
    color: {C_TEXT};
}}
.eq b {{ color: {C_TEAL}; font-weight: 600; }}
.eq .note {{ color: {C_MUTED}; font-size: 0.82rem; display: block; margin-top: 0.15rem; }}

.r0-box {{
    background: {C_PANEL2}; border: 1px solid {C_LINE};
    border-radius: 10px; padding: 0.9rem 1rem;
}}
.r0-box .n {{
    font-family: 'IBM Plex Serif', serif; font-size: 1.55rem;
    color: {C_TEAL}; font-weight: 600; margin: 0.15rem 0;
}}
.r0-bar {{ height: 6px; background: {C_LINE}; border-radius: 99px; overflow: hidden; margin-top: 0.45rem; }}
.r0-bar > span {{ display: block; height: 100%; border-radius: 99px; }}
.bar-lbl {{
    display: flex; justify-content: space-between; font-size: 0.72rem;
    color: {C_MUTED}; margin: 0.55rem 0 0.2rem;
}}

.news {{
    background: {C_PANEL2}; border-left: 3px solid {C_TEAL};
    border-radius: 0 8px 8px 0; padding: 0.55rem 0.75rem;
    color: {C_MUTED}; font-size: 0.82rem; margin: 0 0 0.45rem;
}}
.flow {{
    display: flex; gap: 0.5rem; flex-wrap: wrap;
    align-items: center; margin-top: auto;
}}
.flow .box {{
    border-radius: 10px; padding: 0.55rem 0.8rem;
    font-weight: 700; font-size: 0.8rem;
}}

div.stButton > button {{
    background: {C_PANEL2} !important; color: {C_TEXT} !important;
    border: 1px solid {C_LINE} !important; font-weight: 600 !important;
    border-radius: 8px !important; min-height: 2.5rem;
}}
div.stButton > button:hover {{ border-color: {C_TEAL} !important; color: {C_TEAL} !important; }}
div.stButton > button[kind="primary"] {{
    background: {C_TEAL} !important; color: #06221f !important; border: none !important;
}}
button[data-baseweb="tab"] {{ color: {C_MUTED} !important; font-weight: 600 !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {C_TEAL} !important; }}

[data-testid="stSidebar"] [role="radiogroup"] {{
    gap: 0.45rem !important;
    display: flex !important;
    flex-direction: column !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    background: {C_PANEL2};
    border: 1px solid {C_LINE};
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    margin: 0 !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    border-color: {C_TEAL};
    background: rgba(46, 196, 182, 0.12);
}}
[data-testid="stCaption"] {{ color: {C_MUTED}; }}
hr {{ border-color: {C_LINE} !important; margin: 0.2rem 0 !important; }}
</style>
""",
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C_PANEL2,
    font=dict(family="IBM Plex Sans, sans-serif", size=12, color=C_TEXT),
    margin=dict(l=44, r=16, t=28, b=44),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
)
PLOTLY_CFG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def _layout(**extra) -> dict:
    cfg = dict(PLOT_LAYOUT)
    cfg.update(extra)
    if "xaxis" in extra:
        cfg["xaxis"] = {**dict(gridcolor=C_LINE, zeroline=False, color=C_MUTED), **extra["xaxis"]}
    if "yaxis" in extra:
        cfg["yaxis"] = {**dict(gridcolor=C_LINE, zeroline=False, color=C_MUTED), **extra["yaxis"]}
    return cfg


def carregar_preset(nome: str) -> None:
    for k, v in PRESETS[nome].items():
        st.session_state[k] = v
    registrar(f"Cenário carregado: {nome}")


def interpolar(res, t: float):
    ts = np.asarray(res.tempos, dtype=float)
    t = float(np.clip(t, ts[0], ts[-1]))
    return (
        float(np.interp(t, ts, res.S)),
        float(np.interp(t, ts, res.I)),
        float(np.interp(t, ts, res.R)),
    )


def registrar(evento: str) -> None:
    st.session_state.diario.append(
        {"quando": datetime.now().strftime("%H:%M:%S"), "evento": evento}
    )


def get_mundo():
    try:
        return mundo_from_session(st.session_state.mundo)
    except Exception:
        m = mundo_novo()
        st.session_state.mundo = mundo_to_session(m)
        return m


def save_mundo(m) -> None:
    st.session_state.mundo = mundo_to_session(m)


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------

if "diario" not in st.session_state:
    st.session_state.diario = []
if "N" not in st.session_state:
    st.session_state.pagina = "simulacao"
    st.session_state.reproduzir = False
    st.session_state.mundo = mundo_to_session(mundo_novo())
    st.session_state.semente = "sp"
    st.session_state.metodo = "rk4"
    st.session_state.comparar = True
    st.session_state.dia_insp = 44.5
    st.session_state.ver_s = True
    st.session_state.ver_i = True
    st.session_state.ver_r = True
    st.session_state.ver_euler = True
    st.session_state.modo_area = False
    carregar_preset("gripe")
if "pagina" not in st.session_state:
    st.session_state.pagina = "simulacao"
if "mundo" not in st.session_state or "cura" not in st.session_state.get("mundo", {}):
    st.session_state.mundo = mundo_to_session(mundo_novo())
if "cidade_insp" not in st.session_state:
    st.session_state.cidade_insp = "sp"

# ---------------------------------------------------------------------------
# Sidebar — navegação + conteúdo da seção
# ---------------------------------------------------------------------------

with st.sidebar:
    st.radio(
        "Seção",
        options=["simulacao", "surto", "ajuda"],
        format_func=lambda x: {"simulacao": "Simulação", "surto": "Surto", "ajuda": "Ajuda"}[x],
        key="pagina",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.15rem'></div>", unsafe_allow_html=True)

    pagina = st.session_state.pagina

    if pagina == "simulacao":
        st.caption("Controles da epidemia · atualiza na hora")
        st.button("Contida · R0 < 1", width="stretch", on_click=carregar_preset, args=("contida",))
        st.button("Gripe · R0 ≈ 1,8", width="stretch", on_click=carregar_preset, args=("gripe",))
        st.button("Sarampo · R0 ≈ 12", width="stretch", on_click=carregar_preset, args=("sarampo",))

        N = st.number_input("População N", min_value=100, max_value=10_000_000, step=1_000, key="N")
        if st.session_state.I0 >= int(N):
            st.session_state.I0 = max(1, int(N) - 1)
        I0 = st.number_input(
            "Infectados iniciais I(0)",
            min_value=1,
            max_value=int(min(100_000, max(int(N) - 1, 1))),
            step=1,
            key="I0",
        )
        beta = st.slider("Transmissão β (1/dia)", 0.02, 2.00, step=0.01, key="beta")
        gamma = st.slider("Recuperação γ (1/dia)", 0.02, 1.00, step=0.01, key="gamma")
        t_final = st.slider("Horizonte (dias)", 20, 400, step=10, key="t_final")
        h = st.select_slider("Passo h (dias)", options=[0.05, 0.1, 0.25, 0.5, 1.0], key="h")
        metodo = st.selectbox(
            "Método",
            ["rk4", "euler"],
            format_func=lambda m: "Runge-Kutta 4" if m == "rk4" else "Euler explícito",
            key="metodo",
        )
        comparar = st.checkbox("Comparar Euler × RK4", key="comparar")

        try:
            r0_live = r0_basico(float(beta), float(gamma))
            status = "a doença cresce" if r0_live > 1 else "a doença some sozinha"
            cor = C_INF if r0_live > 1 else C_TEAL
            st.markdown(
                f"""<div class="r0-box">
                <div class="lbl" style="color:{C_MUTED};font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;">R0 agora</div>
                <div class="n">{r0_live:.2f}</div>
                <div style="color:{C_MUTED};font-size:0.8rem;">Cada doente infecta ~{r0_live:.1f} pessoas · {status}</div>
                <div class="r0-bar"><span style="width:{min(100, r0_live/5*100):.1f}%;background:{cor}"></span></div>
                </div>""",
                unsafe_allow_html=True,
            )
        except Exception:
            r0_live = None
            st.warning("γ inválido.")

    elif pagina == "surto":
        st.caption("Você é o vírus. Evolua, espalhe, acompanhe o mapa.")
        m = get_mundo()
        rs = resumo_surto(m)
        st.markdown(
            f"""<div class="r0-box">
            <div style="color:{C_MUTED};font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;">DNA / dia {rs['t']:.0f}</div>
            <div class="n">{rs['dna']}</div>
            <div style="color:{C_MUTED};font-size:0.8rem;">R0 = {rs['R0']:.2f} · ataque {rs['ataque']:.1f}% · {rs['cidades_atingidas']}/8 cidades</div>
            <div class="bar-lbl"><span>Cura</span><span>{rs.get('cura', 0):.0f}%</span></div>
            <div class="r0-bar"><span style="width:{rs.get('cura', 0):.1f}%;background:{C_REC}"></span></div>
            <div class="bar-lbl"><span>Gravidade</span><span>{rs.get('gravidade', 0):.0f}</span></div>
            <div class="r0-bar"><span style="width:{rs.get('gravidade', 0):.1f}%;background:{C_INF}"></span></div>
            </div>""",
            unsafe_allow_html=True,
        )
        if m.origem is None:
            st.selectbox(
                "Cidade-semente",
                options=[c.id for c in m.cidades.values()],
                format_func=lambda i: m.cidades[i].nome,
                key="semente",
            )
            if st.button("Soltar o vírus", width="stretch", type="primary"):
                iniciar_em(m, st.session_state.semente)
                save_mundo(m)
                registrar(f"Surto iniciado em {m.cidades[m.origem].nome}")
                st.rerun()
        else:
            a, b, c, d = st.columns(4)
            if a.button("+1", width="stretch"):
                tick(m, 1)
                save_mundo(m)
                st.rerun()
            if b.button("+7", width="stretch"):
                tick(m, 7)
                save_mundo(m)
                st.rerun()
            if c.button("+30", width="stretch"):
                tick(m, 30)
                save_mundo(m)
                st.rerun()
            if d.button("Novo", width="stretch"):
                st.session_state.mundo = mundo_to_session(mundo_novo())
                st.session_state.reproduzir = False
                registrar("Surto reiniciado")
                st.rerun()
            st.toggle("Reproduzir", key="reproduzir", help="Avança 1 dia a cada recarregamento.")

        st.markdown("**Evoluções**")
        grupo_atual = None
        for tr in TRAITS:
            if tr["grupo"] != grupo_atual:
                grupo_atual = tr["grupo"]
                st.caption(grupo_atual)
            ja = tr["id"] in m.traits
            rotulo = f"{'✓  ' if ja else ''}{tr['nome']}  ·  {tr['custo']}"
            if ja:
                st.caption(rotulo + "  ·  ativo")
            elif st.button(
                rotulo,
                width="stretch",
                disabled=m.origem is None or m.dna < tr["custo"] or bool(m.fim),
                help=tr["desc"],
            ):
                if comprar_trait(m, tr["id"]):
                    save_mundo(m)
                    registrar(f"Evolução comprada: {tr['nome']}")
                    st.rerun()

        st.markdown("**Boletim**")
        for n in m.news[:5]:
            st.markdown(f'<div class="news">{n}</div>', unsafe_allow_html=True)

    else:
        st.caption("História, fórmula e o motor numérico estão à direita.")
        st.markdown(
            """
**Três filas**

1. ainda podem pegar (S)  
2. doentes agora (I)  
3. já passaram (R)  

**A conta**

R0 = β ÷ γ  

*grudenta* ÷ *rápido sara* = *quantos cada doente infecta*

O computador não tem o dia 47 de cabeça: ele anda um passo *h* com RK4 e filma os três grupos.
"""
        )

# ---------------------------------------------------------------------------
# Figuras do laboratório
# ---------------------------------------------------------------------------


def figura_series(res, extra, t_mark, s_m, i_m, r_m, ver, area: bool) -> go.Figure:
    fig = go.Figure()
    series = [
        (res.S, C_S, "Ainda podem pegar", ver["s"], s_m),
        (res.I, C_INF, "Doentes agora", ver["i"], i_m),
        (res.R, C_REC, "Já passaram", ver["r"], r_m),
    ]
    for ys, cor, nome, vis, _m in series:
        if not vis:
            continue
        kw = dict(
            x=res.tempos, y=ys, name=nome,
            hovertemplate="%{y:,.0f}<extra>" + nome + "</extra>",
        )
        if area:
            fig.add_trace(go.Scatter(**kw, stackgroup="sir", mode="lines", line=dict(color=cor, width=1.4)))
        else:
            fig.add_trace(go.Scatter(**kw, line=dict(color=cor, width=2.5)))
    if extra is not None and ver["euler"]:
        fig.add_trace(
            go.Scatter(
                x=extra.tempos, y=extra.I, name="Doentes (Euler)",
                line=dict(color=C_WARN, width=1.7, dash="dash"),
                hovertemplate="%{y:,.0f}<extra>Euler</extra>",
            )
        )
    fig.add_vline(x=res.t_pico, line_width=1, line_dash="dot", line_color=C_MUTED,
                  annotation_text=f"pico · dia {res.t_pico:.1f}", annotation_font_color=C_MUTED)
    fig.add_vline(x=t_mark, line_width=1.6, line_color=C_TEAL)
    if not area:
        fig.add_trace(
            go.Scatter(
                x=[t_mark, t_mark, t_mark], y=[s_m, i_m, r_m], mode="markers",
                marker=dict(size=9, color=[C_S, C_INF, C_REC], line=dict(width=1, color="white")),
                showlegend=False, hoverinfo="skip",
            )
        )
    fig.update_layout(
        **_layout(
            height=360,
            xaxis=dict(
                title="Dias",
                rangeslider=dict(visible=True, thickness=0.07, bgcolor=C_PANEL, bordercolor=C_LINE),
            ),
            yaxis=dict(title="Pessoas"),
        )
    )
    return fig


def figura_fase(res, s_m, i_m) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.S, y=res.I, mode="lines", name="caminho", line=dict(color=C_INF, width=2.3)))
    fig.add_trace(go.Scatter(x=[res.S[0]], y=[res.I[0]], mode="markers", name="início", marker=dict(size=9, color=C_TEAL)))
    fig.add_trace(go.Scatter(x=[res.S_final], y=[res.I_final], mode="markers", name="fim", marker=dict(size=9, color=C_REC)))
    fig.add_trace(go.Scatter(x=[s_m], y=[i_m], mode="markers", name="agora", marker=dict(size=11, color=C_WARN, symbol="x")))
    if res.R0 and res.R0 > 0:
        fig.add_vline(x=res.N / res.R0, line_width=1, line_dash="dash", line_color=C_TEAL,
                      annotation_text="limiar", annotation_font_color=C_TEAL)
    fig.update_layout(**_layout(height=280, hovermode="closest",
                                xaxis=dict(title="Ainda podem pegar"), yaxis=dict(title="Doentes agora")))
    return fig


def figura_erro(res, t_mark: float) -> go.Figure:
    hist = res.historico
    passo = max(1, len(hist) // 500)
    amostra = hist[::passo]
    ts = [s.t for s in amostra if s.erro_local is not None]
    errs = [s.erro_local for s in amostra if s.erro_local is not None]
    fig = go.Figure()
    if ts:
        fig.add_trace(go.Scatter(x=ts, y=errs, mode="lines", line=dict(color=C_WARN, width=2),
                                 fill="tozeroy", fillcolor="rgba(233,196,106,0.12)",
                                 hovertemplate="t=%{x:.1f}<br>%{y:.2e}<extra></extra>"))
    fig.add_vline(x=t_mark, line_width=1.5, line_color=C_TEAL)
    fig.update_layout(**_layout(height=280, showlegend=False,
                                xaxis=dict(title="Dias"),
                                yaxis=dict(title="Diferença Euler × RK4", type="log")))
    return fig


STATUS_COR = {
    "saudavel": "#8B9BB4",
    "infectada": "#E9C46A",
    "epidemia": "#E07A5F",
    "crise": "#E85D4C",
    "recuperada": "#5B8DEF",
}
STATUS_NOME = {
    "saudavel": "sem casos",
    "infectada": "casos locais",
    "epidemia": "epidemia",
    "crise": "crise",
    "recuperada": "onda passou",
}


def figura_mapa(m) -> go.Figure:
    fig = go.Figure()
    for a in m.cidades.values():
        if a.I < 15:
            continue
        for b in m.cidades.values():
            w = PESOS[(a.id, b.id)]
            if w < 0.07:
                continue
            op = min(0.45, 0.08 + 0.35 * w * min(1.0, a.I / 80_000))
            fig.add_trace(
                go.Scattergeo(
                    lat=[a.lat, b.lat], lon=[a.lon, b.lon],
                    mode="lines",
                    line=dict(width=0.7 + 2.4 * w, color=f"rgba(232,93,76,{op:.2f})"),
                    hoverinfo="skip", showlegend=False,
                )
            )
    lats, lons, sizes, colors, texts, names = [], [], [], [], [], []
    for c in m.cidades.values():
        stt = c.status
        lats.append(c.lat)
        lons.append(c.lon)
        names.append(c.nome)
        sizes.append(16 + 20 * (c.N / 12_300_000) ** 0.5)
        colors.append(STATUS_COR[stt])
        texts.append(
            f"<b>{c.nome}</b> · {STATUS_NOME[stt]}<br>"
            f"Doentes: {c.I:,.0f}<br>"
            f"Já passaram: {c.R:,.0f}<br>"
            f"Ainda podem pegar: {c.S:,.0f}<br>"
            f"Agora: {c.prevalencia:.2f}%"
        )
    fig.add_trace(
        go.Scattergeo(
            lat=lats, lon=lons, text=names, hovertext=texts, hoverinfo="text",
            marker=dict(
                size=sizes, color=colors,
                line=dict(width=1.4, color="white"),
            ),
            showlegend=False,
        )
    )
    fig.update_geos(
        projection_type="mercator",
        showland=True, landcolor="#152033",
        showocean=True, oceancolor="#0B1220",
        showlakes=False, showrivers=False,
        showcountries=True, countrycolor="#2A3B55",
        showcoastlines=True, coastlinecolor="#2A3B55",
        bgcolor="rgba(0,0,0,0)",
        center=dict(lat=-15.0, lon=-50.0),
        lataxis_range=[-33.5, 2.5],
        lonaxis_range=[-74, -33],
        resolution=50,
        fitbounds=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=390,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family="IBM Plex Sans, sans-serif", color=C_TEXT),
    )
    return fig


def figura_nacional(m) -> go.Figure:
    fig = go.Figure()
    hist = m.historico
    if hist:
        ts = [p["t"] for p in hist]
        fig.add_trace(go.Scatter(
            x=ts, y=[p["I"] for p in hist], name="Doentes",
            line=dict(color=C_INF, width=2.3),
            hovertemplate="dia %{x:.0f}<br>%{y:,.0f}<extra>Doentes</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=ts, y=[p["R"] for p in hist], name="Já passaram",
            line=dict(color=C_REC, width=2),
            hovertemplate="dia %{x:.0f}<br>%{y:,.0f}<extra>Já passaram</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=ts, y=[p.get("cura", 0) for p in hist], name="Cura %",
            yaxis="y2",
            line=dict(color=C_TEAL, width=1.6, dash="dot"),
            hovertemplate="dia %{x:.0f}<br>%{y:.0f}%<extra>Cura</extra>",
        ))
    layout = _layout(
        height=280, showlegend=True,
        xaxis=dict(title="Dias"),
        yaxis=dict(title="Pessoas"),
        margin=dict(l=44, r=44, t=16, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    layout["yaxis2"] = dict(
        title="Cura %", overlaying="y", side="right", range=[0, 100],
        gridcolor="rgba(0,0,0,0)", color=C_TEAL, zeroline=False,
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------

pagina = st.session_state.pagina

if pagina == "simulacao":
    kwargs = dict(
        beta=float(st.session_state.beta),
        gamma=float(st.session_state.gamma),
        N=float(st.session_state.N),
        I0=float(min(st.session_state.I0, st.session_state.N - 1)),
        t_final=float(st.session_state.t_final),
        h=float(st.session_state.h),
        max_iter=200_000,
        comparar_erro_local=True,
        parar_se_extinta=False,
    )
    if st.session_state.comparar:
        r_euler, r_rk4 = comparar_metodos(**kwargs)
        r = r_rk4 if st.session_state.metodo == "rk4" else r_euler
        outro = r_euler if st.session_state.metodo == "rk4" else r_rk4
        if outro is not None and not outro.sucesso:
            outro = None
    else:
        r = simular_sir(metodo=st.session_state.metodo, **kwargs)
        outro = None
        r_euler = r_rk4 = None

    if not r.sucesso:
        st.error(r.mensagem)
        st.stop()

    if st.session_state.dia_insp > float(st.session_state.t_final):
        st.session_state.dia_insp = float(st.session_state.t_final)

    rebanho = f"precisa imunizar {r.limiar_rebanho*100:.0f}%" if r.limiar_rebanho else "abaixo do limiar"
    st.markdown(
        f"""<div class="stack">
        <div>
          <div class="kicker">Laboratório</div>
          <div class="title">Painel SIR</div>
          <p class="lede">Arraste os controles à esquerda. O pico, o dia e o tamanho da onda mudam na hora.</p>
        </div>
        <div class="kpi-row">
          <div class="card"><p class="lbl">R0</p><p class="val">{r.R0:.2f}</p><p class="hint">{rebanho}</p></div>
          <div class="card"><p class="lbl">Pico de doentes</p><p class="val">{r.pico_I:,.0f}</p><p class="hint">{100*r.pico_I/r.N:.2f}% da cidade</p></div>
          <div class="card"><p class="lbl">Dia do pico</p><p class="val">{r.t_pico:.1f}</p><p class="hint">{r.metodo} · h = {st.session_state.h:g}</p></div>
          <div class="card"><p class="lbl">Já passaram no fim</p><p class="val">{100*r.ataque_final:.1f}%</p><p class="hint">{r.R_final:,.0f} pessoas</p></div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    c_sl, c_btn = st.columns((3.4, 1), vertical_alignment="bottom")
    with c_sl:
        dia = st.slider("Olhar o dia…", 0.0, float(st.session_state.t_final), step=0.5, key="dia_insp")
    with c_btn:
        if st.button("Ir ao pico", width="stretch"):
            st.session_state.dia_insp = float(round(r.t_pico, 1))
            st.rerun()

    s_m, i_m, r_m = interpolar(r, float(dia))
    st.markdown(
        f"""<div class="insp-nums">
          <div class="card"><p class="lbl">Ainda podem pegar</p><p class="val" style="color:{C_S}">{s_m:,.0f}</p></div>
          <div class="card"><p class="lbl">Doentes agora</p><p class="val" style="color:{C_INF}">{i_m:,.0f}</p></div>
          <div class="card"><p class="lbl">Já passaram</p><p class="val" style="color:{C_REC}">{r_m:,.0f}</p></div>
        </div>""",
        unsafe_allow_html=True,
    )

    v1, v2, v3, v4, v5 = st.columns(5, vertical_alignment="center")
    ver_s = v1.checkbox("Podem pegar", key="ver_s")
    ver_i = v2.checkbox("Doentes", key="ver_i")
    ver_r = v3.checkbox("Já passaram", key="ver_r")
    ver_eu = v4.checkbox("Euler", key="ver_euler", disabled=outro is None)
    area = v5.toggle("Empilhar", key="modo_area")
    ver = {"s": ver_s, "i": ver_i, "r": ver_r, "euler": ver_eu and outro is not None}

    st.plotly_chart(figura_series(r, outro, float(dia), s_m, i_m, r_m, ver, area),
                    width="stretch", config=PLOTLY_CFG)

    f1, f2 = st.columns(2, vertical_alignment="top")
    with f1:
        st.caption("Caminho da epidemia (quem ainda pode pegar × quem está doente)")
        st.plotly_chart(figura_fase(r, s_m, i_m), width="stretch", config=PLOTLY_CFG)
    with f2:
        st.caption("Onde o atalho Euler erra mais, comparado ao RK4")
        st.plotly_chart(figura_erro(r, float(dia)), width="stretch", config=PLOTLY_CFG)

    tab1, tab2, tab3 = st.tabs(["Passos", "Euler × RK4", "Diário"])
    hist = r.historico
    if len(hist) <= 40:
        amostra = hist
    else:
        idxs = list(range(8)) + list(range(len(hist)//2 - 6, len(hist)//2 + 6)) + list(range(len(hist) - 8, len(hist)))
        amostra = [hist[i] for i in sorted(set(i for i in idxs if 0 <= i < len(hist)))]

    with tab1:
        df = pd.DataFrame(
            [{
                "n": s.n, "t (d)": round(s.t, 3), "S": round(s.S, 1),
                "I": round(s.I, 1), "R": round(s.R, 1),
                "|S+I+R−N|": f"{s.conservacao:.2e}",
                "erro": f"{s.erro_local:.2e}" if s.erro_local is not None else "—",
            } for s in amostra]
        )
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(f"{r.metodo} · {r.n_passos} passos · drift {r.drift_max:.2e}")

    with tab2:
        if outro is None:
            r_e, r_rk = comparar_metodos(**kwargs)
        elif st.session_state.metodo == "rk4":
            r_rk, r_e = r, outro
        else:
            r_e, r_rk = r, outro
        a, b, c = st.columns(3)
        a.metric("Pico Euler", f"{r_e.pico_I:,.0f}")
        b.metric("Pico RK4", f"{r_rk.pico_I:,.0f}")
        c.metric("Diferença", f"{abs(r_e.pico_I - r_rk.pico_I):,.0f}")
        st.caption("Passo menor → diferença menor. Euler com passo grosso atrasa o pico.")

    with tab3:
        if st.button("Anotar cenário atual no diário"):
            registrar(
                f"Lab N={r.N:.0f} β={r.beta:.2f} γ={r.gamma:.2f} R0={r.R0:.2f} "
                f"pico={r.pico_I:.0f} dia={r.t_pico:.1f}"
            )
            st.rerun()
        if st.session_state.diario:
            st.dataframe(pd.DataFrame(st.session_state.diario), width="stretch", hide_index=True)
        else:
            st.caption("O diário ainda está vazio.")

        amostra_pdf = [
            {"n": s.n, "t": s.t, "S": s.S, "I": s.I, "R": s.R,
             "erro": f"{s.erro_local:.2e}" if s.erro_local is not None else "-"}
            for s in amostra
        ]
        comp = None
        if outro is not None:
            if st.session_state.metodo == "rk4":
                re_, rk_ = outro, r
            else:
                re_, rk_ = r, outro
            comp = {
                "pico_euler": re_.pico_I, "pico_rk4": rk_.pico_I,
                "delta_pico": abs(re_.pico_I - rk_.pico_I),
                "delta_dia": abs(re_.t_pico - rk_.t_pico),
            }
        pdf = pdf_laboratorio(
            params=dict(N=r.N, I0=kwargs["I0"], beta=r.beta, gamma=r.gamma,
                        t_final=kwargs["t_final"], h=kwargs["h"], metodo=r.metodo),
            kpis=dict(R0=r.R0, pico_I=r.pico_I, t_pico=r.t_pico, ataque_final=r.ataque_final,
                      n_passos=r.n_passos, drift_max=r.drift_max),
            historico_amostra=amostra_pdf,
            comparacao=comp,
            diario=st.session_state.diario,
        )
        st.download_button(
            "Baixar relatório PDF",
            data=pdf,
            file_name=f"painel_sir_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            width="stretch",
            type="primary",
        )

elif pagina == "surto":
    m = get_mundo()
    if st.session_state.get("reproduzir") and m.origem and not m.fim:
        tick(m, 1.0)
        save_mundo(m)
        time.sleep(0.4)
        st.rerun()

    rs = resumo_surto(m)
    st.markdown(
        f"""<div class="stack">
        <div>
          <div class="kicker">Modo surto</div>
          <div class="title">O vírus contra o Brasil</div>
          <p class="lede">Você é o patógeno. Solte o vírus, gaste DNA em sintomas e corra contra a cura. Cada capital é um SIR; aviões e ônibus levam o contágio.</p>
        </div>
        <div class="kpi-row">
          <div class="card"><p class="lbl">Dia</p><p class="val">{rs['t']:.0f}</p><p class="hint">R0 {rs['R0']:.2f}</p></div>
          <div class="card"><p class="lbl">Doentes agora</p><p class="val">{rs['I']:,.0f}</p><p class="hint">em 8 capitais</p></div>
          <div class="card"><p class="lbl">Já passaram</p><p class="val">{rs['ataque']:.1f}%</p><p class="hint">ataque nacional</p></div>
          <div class="card"><p class="lbl">Cura / gravidade</p><p class="val">{rs.get('cura', 0):.0f}%</p><p class="hint">gravidade {rs.get('gravidade', 0):.0f} · DNA {rs['dna']}</p></div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if m.fim == "vitoria":
        st.success("O país foi saturado. Baixe o PDF e registre a campanha.")
    elif m.fim == "cura":
        st.warning("A cura ou o isolamento ganharam. Evolua mais cedo e tente de novo.")

    if m.origem and not m.fim:
        t1, t2, t3, t4 = st.columns(4, vertical_alignment="center")
        if t1.button("+1 dia", width="stretch"):
            tick(m, 1)
            save_mundo(m)
            st.rerun()
        if t2.button("+7 dias", width="stretch"):
            tick(m, 7)
            save_mundo(m)
            st.rerun()
        if t3.button("+30 dias", width="stretch"):
            tick(m, 30)
            save_mundo(m)
            st.rerun()
        if t4.button("Reiniciar surto", width="stretch"):
            st.session_state.mundo = mundo_to_session(mundo_novo())
            st.session_state.reproduzir = False
            registrar("Surto reiniciado")
            st.rerun()
    elif m.origem is None:
        st.info("Na barra da esquerda, escolha a cidade-semente e aperte **Soltar o vírus**.")

    mapa, lista = st.columns((1.55, 1), vertical_alignment="top")
    with mapa:
        st.plotly_chart(figura_mapa(m), width="stretch", config={"displayModeBar": False})
        st.caption("Cinza: sem casos · amarelo: local · laranja: epidemia · vermelho: crise · azul: a onda já passou. Linhas: rotas ativas.")
    with lista:
        rows = []
        for c in sorted(m.cidades.values(), key=lambda x: -x.I):
            rows.append({
                "Cidade": c.nome,
                "Situação": STATUS_NOME[c.status],
                "Doentes": int(c.I),
                "Já passaram": int(c.R),
                "% agora": round(c.prevalencia, 2),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=360)
        ids = list(m.cidades.keys())
        st.selectbox(
            "Inspecionar cidade",
            options=ids,
            format_func=lambda i: m.cidades[i].nome,
            key="cidade_insp",
        )
        cid = m.cidades[st.session_state.cidade_insp]
        st.markdown(
            f"""<div class="card" style="min-height:auto">
              <p class="lbl">{cid.nome} · {STATUS_NOME[cid.status]}</p>
              <p class="hint" style="margin-top:0.45rem">Ainda podem pegar {cid.S:,.0f} · doentes {cid.I:,.0f} · já passaram {cid.R:,.0f}</p>
              <div class="bar-lbl"><span>Doentes agora</span><span>{cid.prevalencia:.2f}%</span></div>
              <div class="r0-bar"><span style="width:{min(100, cid.prevalencia * 6):.1f}%;background:{STATUS_COR[cid.status]}"></span></div>
              <div class="bar-lbl"><span>Já tiveram contato</span><span>{cid.ataque:.1f}%</span></div>
              <div class="r0-bar"><span style="width:{min(100, cid.ataque):.1f}%;background:{C_REC}"></span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    curva, boletim = st.columns((1.55, 1), vertical_alignment="top")
    with curva:
        st.caption("Onda nacional — doentes, recuperados e avanço da cura")
        st.plotly_chart(figura_nacional(m), width="stretch", config=PLOTLY_CFG)
    with boletim:
        st.caption("Boletim e medidas do governo")
        if m.medidas:
            nomes = []
            for md in m.medidas:
                if md == "alerta":
                    nomes.append("alerta nacional")
                elif md == "aeroportos":
                    nomes.append("triagem nos aeroportos")
                elif md == "campanha":
                    nomes.append("campanha de higiene")
                elif md.startswith("lockdown_"):
                    nomes.append("restrição em " + m.cidades[md.split("_", 1)[1]].nome)
            st.markdown(
                "<p class='hint'>" + " · ".join(nomes) + "</p>",
                unsafe_allow_html=True,
            )
        for n in m.news[:6]:
            st.markdown(f'<div class="news">{n}</div>', unsafe_allow_html=True)

    cidades_pdf = [
        {"nome": c.nome, "I": c.I, "S": c.S, "R": c.R, "prev": c.prevalencia}
        for c in m.cidades.values()
    ]
    pdf_s = pdf_surto(rs, m.log, cidades_pdf)
    st.download_button(
        "Baixar log do surto em PDF",
        data=pdf_s,
        file_name=f"surto_{datetime.now():%Y%m%d_%H%M}.pdf",
        mime="application/pdf",
        width="stretch",
        type="primary",
    )

else:
    st.markdown(
        f"""<div class="stack">
<div>
  <div class="kicker">Ajuda</div>
  <div class="title">Como o painel calcula a epidemia</div>
  <p class="lede">Primeiro a história. Depois a fórmula que o motor integra a cada passo, sem biblioteca de EDO.</p>
</div>
<div class="help-grid">
  <div class="card">
    <h3>Os dois gêmeos gripados</h3>
    <p>Uma escola tem 1.000 alunos. Segunda-feira: dois gêmeos voltam de um casamento resfriados.
    Na terça, a melhor amiga tosse. Na quarta, metade da sala falta. Na sexta, os gêmeos já estão bem
    e voltam — mas agora quem está doente são os outros.</p>
    <p>Três filas na chamada: quem ainda não pegou, quem está na cama, quem já ficou bom.
    Ninguém é apagado da lista. É só isso o modelo SIR.</p>
    <div class="flow">
      <div class="box" style="background:#14352f;color:{C_S}">podem pegar S</div>
      <span style="color:{C_MUTED}">gruda →</span>
      <div class="box" style="background:#3a1d1a;color:{C_INF}">doentes I</div>
      <span style="color:{C_MUTED}">melhoram →</span>
      <div class="box" style="background:#1a2744;color:{C_REC}">já passaram R</div>
    </div>
  </div>
  <div class="card">
    <h3>A conta, como volume da rádio</h3>
    <p>Cada aluno doente, num dia, conversa de perto com 3 pessoas.
    Se a gripe pega em 12% desses encontros, isso é o <b>β</b> — o quão grudenta ela é.</p>
    <p>Se a pessoa fica doente uns 5 dias, isso é <b>1/γ</b> — o quão lento ela sara.</p>
    <p>Multiplica: 3 × 0,12 × 5 = <b>1,8</b>. Esse 1,8 é o <b>R0</b>: cada doente, no começo,
    passa a doença para quase dois colegas.</p>
    <p>R0 maior que 1: a turma vira dominó. R0 menor que 1: o assunto morre no recreio.</p>
  </div>
  <div class="card span2">
    <h3>A fórmula por trás do sistema</h3>
    <p>O motor integra o SIR de Kermack–McKendrick (1927). Não existe fórmula fechada para S(t), I(t), R(t):
    por isso cada quadro do gráfico é um passo numérico.</p>
    <div class="eq-row">
      <div class="eq">
        dS/dt = −β · S · I / N<br>
        dI/dt = &nbsp;β · S · I / N − γ · I<br>
        dR/dt = γ · I
        <span class="note">S = ainda podem pegar · I = doentes agora · R = já passaram · N = população</span>
      </div>
      <div class="eq">
        R0 = β / γ<br>
        limiar de rebanho = 1 − 1/R0<br>
        S + I + R = N
        <span class="note">R0 &gt; 1 a onda cresce. R0 &lt; 1 ela murcha. As três filas sempre somam a escola inteira.</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h3>Como o sistema funciona</h3>
    <p>O programa guarda os três números (S, I, R) e o relógio t. A cada passo de tamanho <b>h</b> (em dias)
    ele pergunta à fórmula: “o quanto cada grupo muda neste instantinho?” e atualiza o estado.</p>
    <p><b>RK4</b> faz quatro sondagens no intervalo e mistura o resultado — o filme padrão do laboratório e do Surto.</p>
    <p><b>Euler</b> dá um passo só, mais tosco. Com h grosso ele atrasa o pico. O painel mostra a diferença |RK4 − Euler|.</p>
    <p>No Surto, cada capital é um SIR próprio. Aviões e ônibus acoplam as cidades: uma fração dos doentes de A encontra
    quem ainda pode pegar em B. Evoluções mudam β, γ e a viagem; o governo e a cura puxam o outro lado.</p>
  </div>
  <div class="card">
    <h3>O passo de Runge-Kutta 4</h3>
    <p>Se y = (S, I, R) e f(t, y) é a fórmula acima, um quadro do filme é:</p>
    <div class="eq">
      k1 = f(t, y)<br>
      k2 = f(t + h/2, y + h/2 · k1)<br>
      k3 = f(t + h/2, y + h/2 · k2)<br>
      k4 = f(t + h, y + h · k3)<br>
      y<sub>n+1</sub> = y + (h/6) · (k1 + 2k2 + 2k3 + k4)
    </div>
    <p>Euler seria só y + h · k1. Tudo isso está em <b>engine.py</b>, escrito na mão: sem SciPy para integrar ou achar raiz.</p>
  </div>
  <div class="card">
    <h3>O que cada tela faz</h3>
    <p><b>Simulação</b> — você mexe em β, γ, N e h. O gráfico mostra a onda inteira, o caminho S×I e o erro do atalho Euler. O PDF guarda parâmetros, passos e o diário.</p>
    <p><b>Surto</b> — o mesmo motor em oito capitais. Você joga de vírus: escolhe a semente, gasta DNA, avança o calendário. O mapa muda de cor, o governo reage, a cura pesquisa.</p>
    <p><b>Teste rápido:</b> Contida (R0 &lt; 1, a onda nem começa) → Sarampo (pico cedo e alto) → Surto em São Paulo, Aerossol, +30 dias.</p>
  </div>
  <div class="card">
    <h3>O que o laboratório está medindo</h3>
    <p><b>Pico</b> é o dia em que I é máximo — o pior momento para o hospital.</p>
    <p><b>Ataque</b> é a fração que já passou pela doença no fim: (N − S) / N.</p>
    <p><b>Drift</b> é |S+I+R − N|. Tem de ficar perto de zero: se crescer, o passo numérico está vazando gente.</p>
    <p>Arraste o dia, empilhe as cores, compare Euler × RK4. Se as três cores somam a população, a conta fechou.</p>
  </div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
