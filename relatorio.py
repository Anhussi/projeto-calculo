"""Gera relatórios PDF do laboratório e do modo Surto (logs inclusos)."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional

from fpdf import FPDF

FONT_REG = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


class RelatorioSIR(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=16)
        if FONT_REG.exists():
            self.add_font("UI", "", str(FONT_REG))
            self.add_font("UI", "B", str(FONT_BLD if FONT_BLD.exists() else FONT_REG))
            self.font_ok = True
        else:
            self.font_ok = False

    def _f(self, style: str = "", size: int = 11) -> None:
        family = "UI" if self.font_ok else "Helvetica"
        self.set_font(family, style, size)

    def header(self) -> None:
        self._f("B", 12)
        self.set_text_color(26, 143, 133)
        self.cell(0, 8, "Painel SIR  |  relatorio de simulacao", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(46, 196, 182)
        self.set_line_width(0.4)
        y = self.get_y()
        self.line(10, y, 200, y)
        self.ln(6)
        self.set_text_color(30, 30, 30)

    def footer(self) -> None:
        self.set_y(-12)
        self._f("", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            8,
            f"Motor numerico proprio (Euler / RK4)  ·  {datetime.now():%d/%m/%Y %H:%M}  ·  p. {self.page_no()}",
            align="C",
        )

    def titulo(self, texto: str) -> None:
        self._f("B", 14)
        self.set_text_color(11, 18, 32)
        self.multi_cell(0, 8, texto)
        self.ln(2)

    def secao(self, texto: str) -> None:
        self.ln(2)
        self._f("B", 12)
        self.set_text_color(26, 143, 133)
        self.cell(0, 7, texto, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(30, 30, 30)
        self.ln(1)

    def corpo(self, texto: str) -> None:
        self._f("", 10)
        self.multi_cell(0, 5.5, texto)
        self.ln(1)

    def kv(self, pares: Iterable[tuple[str, str]]) -> None:
        self._f("", 10)
        for k, v in pares:
            self._f("B", 10)
            self.cell(55, 6, k)
            self._f("", 10)
            self.cell(0, 6, str(v), new_x="LMARGIN", new_y="NEXT")

    def tabela(self, cab: List[str], linhas: List[List[str]], larguras: Optional[List[float]] = None) -> None:
        if not larguras:
            larguras = [190 / len(cab)] * len(cab)
        self._f("B", 8)
        self.set_fill_color(18, 26, 43)
        self.set_text_color(232, 238, 246)
        for i, c in enumerate(cab):
            self.cell(larguras[i], 7, c, border=0, fill=True)
        self.ln()
        self.set_text_color(30, 30, 30)
        self._f("", 8)
        fill = False
        for row in linhas:
            if fill:
                self.set_fill_color(236, 244, 242)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(larguras[i], 6, str(cell), border=0, fill=True)
            self.ln()
            fill = not fill
        self.ln(2)


def _pdf_bytes(pdf: RelatorioSIR) -> bytes:
    gerado = pdf.output()
    if isinstance(gerado, (bytes, bytearray)):
        return bytes(gerado)
    out = BytesIO()
    pdf.output(out)
    return out.getvalue()


def pdf_laboratorio(
    *,
    params: dict,
    kpis: dict,
    historico_amostra: List[dict],
    comparacao: Optional[dict],
    diario: List[dict],
) -> bytes:
    pdf = RelatorioSIR()
    pdf.add_page()
    pdf.titulo("Laboratorio SIR")
    pdf.corpo(
        "Registro automatico da simulacao. O motor integra as EDOs do modelo "
        "SIR com Euler e Runge-Kutta 4, sem bibliotecas que resolvam a EDO."
    )
    pdf.secao("Parametros")
    pdf.kv(
        [
            ("Populacao N", f"{params.get('N', 0):,.0f}"),
            ("Infectados I(0)", f"{params.get('I0', 0):,.0f}"),
            ("beta (1/dia)", f"{params.get('beta', 0):.3f}"),
            ("gamma (1/dia)", f"{params.get('gamma', 0):.3f}"),
            ("Horizonte (dias)", f"{params.get('t_final', 0):.0f}"),
            ("Passo h", f"{params.get('h', 0):g}"),
            ("Metodo", str(params.get("metodo", "")).upper()),
        ]
    )
    pdf.secao("Resultados")
    pdf.kv(
        [
            ("R0", f"{kpis.get('R0', 0):.3f}" if kpis.get("R0") is not None else "-"),
            ("Pico de infectados", f"{kpis.get('pico_I', 0):,.1f}"),
            ("Dia do pico", f"{kpis.get('t_pico', 0):.2f}"),
            ("Ataque final", f"{100 * kpis.get('ataque_final', 0):.2f}%"),
            ("Passos", f"{kpis.get('n_passos', 0)}"),
            ("Drift max |S+I+R-N|", f"{kpis.get('drift_max', 0):.2e}"),
        ]
    )
    if comparacao:
        pdf.secao("Euler x RK4")
        pdf.kv(
            [
                ("Pico Euler", f"{comparacao['pico_euler']:,.1f}"),
                ("Pico RK4", f"{comparacao['pico_rk4']:,.1f}"),
                ("Delta pico", f"{comparacao['delta_pico']:,.2f}"),
                ("Delta dia do pico", f"{comparacao['delta_dia']:.3f}"),
            ]
        )
    if historico_amostra:
        pdf.secao("Amostra de passos")
        pdf.tabela(
            ["n", "t (d)", "S", "I", "R", "erro local"],
            [
                [
                    str(s.get("n", "")),
                    f"{s.get('t', 0):.2f}",
                    f"{s.get('S', 0):,.0f}",
                    f"{s.get('I', 0):,.0f}",
                    f"{s.get('R', 0):,.0f}",
                    s.get("erro", "-"),
                ]
                for s in historico_amostra
            ],
            [18, 28, 36, 36, 36, 36],
        )
    if diario:
        pdf.secao("Diario / logs")
        pdf.tabela(
            ["quando", "evento"],
            [[d.get("quando", ""), d.get("evento", "")[:90]] for d in diario[-30:]],
            [40, 150],
        )
    return _pdf_bytes(pdf)


def pdf_surto(resumo: dict, log: List[dict], cidades: List[dict]) -> bytes:
    pdf = RelatorioSIR()
    pdf.add_page()
    pdf.titulo("Modo Surto — log da epidemia")
    pdf.corpo(
        "Cada cidade e um SIR integrado com RK4. Viagens acoplam as capitais. "
        "Evolucoes alteram beta, gamma e a taxa de deslocamento."
    )
    pdf.secao("Estado")
    pdf.kv(
        [
            ("Dia", f"{resumo.get('t', 0):.1f}"),
            ("R0", f"{resumo.get('R0', 0):.2f}" if resumo.get("R0") is not None else "-"),
            ("beta", f"{resumo.get('beta', 0):.3f}"),
            ("gamma", f"{resumo.get('gamma', 0):.3f}"),
            ("Infectados agora", f"{resumo.get('I', 0):,.0f}"),
            ("Ataque global", f"{resumo.get('ataque', 0):.2f}%"),
            ("DNA restante", str(resumo.get("dna", 0))),
            ("Cura", f"{resumo.get('cura', 0):.1f}%"),
            ("Gravidade", f"{resumo.get('gravidade', 0):.0f}"),
            ("Cidades atingidas", str(resumo.get("cidades_atingidas", 0))),
            ("Origem", str(resumo.get("origem") or "-")),
            ("Desfecho", str(resumo.get("fim") or "em curso")),
            ("Evolucoes", ", ".join(resumo.get("traits") or []) or "-"),
            ("Medidas", ", ".join(resumo.get("medidas") or []) or "-"),
        ]
    )
    pdf.secao("Cidades")
    pdf.tabela(
        ["cidade", "I", "S", "R", "preval. %"],
        [
            [
                c["nome"],
                f"{c['I']:,.0f}",
                f"{c['S']:,.0f}",
                f"{c['R']:,.0f}",
                f"{c['prev']:.2f}",
            ]
            for c in cidades
        ],
        [42, 37, 37, 37, 37],
    )
    pdf.secao("Log de eventos")
    pdf.tabela(
        ["dia", "evento"],
        [[f"{e.get('t', 0):.1f}", str(e.get("evento", ""))[:90]] for e in log],
        [22, 168],
    )
    return _pdf_bytes(pdf)
