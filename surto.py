"""
Modo Surto — simulação interativa estilo Plague Inc.
Cada cidade é um SIR próprio, integrado com RK4 do engine.
Viagens acoplam as cidades. Cura e medidas de governo competem com o vírus.
Sem SciPy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine import derivada_sir, passo_rk4, r0_basico

CIDADES_BASE: List[dict] = [
    {"id": "manaus", "nome": "Manaus", "lat": -3.12, "lon": -60.02, "N": 2_200_000},
    {"id": "recife", "nome": "Recife", "lat": -8.05, "lon": -34.88, "N": 1_650_000},
    {"id": "salvador", "nome": "Salvador", "lat": -12.97, "lon": -38.50, "N": 2_400_000},
    {"id": "brasilia", "nome": "Brasília", "lat": -15.78, "lon": -47.93, "N": 3_050_000},
    {"id": "bh", "nome": "Belo Horizonte", "lat": -19.92, "lon": -43.94, "N": 2_500_000},
    {"id": "rj", "nome": "Rio de Janeiro", "lat": -22.91, "lon": -43.17, "N": 6_700_000},
    {"id": "sp", "nome": "São Paulo", "lat": -23.55, "lon": -46.63, "N": 12_300_000},
    {"id": "poa", "nome": "Porto Alegre", "lat": -30.03, "lon": -51.23, "N": 1_480_000},
]

TRAITS: List[dict] = [
    {
        "id": "aperto",
        "nome": "Aperto de mão",
        "grupo": "Transmissão",
        "custo": 4,
        "dbeta": 0.05,
        "dgamma": 0.0,
        "dviagem": 0.0,
        "grav": 4,
        "desc": "Contágio no cumprimento e no transporte lotado.",
    },
    {
        "id": "tosse",
        "nome": "Tosse seca",
        "grupo": "Transmissão",
        "custo": 6,
        "dbeta": 0.08,
        "dgamma": 0.0,
        "dviagem": 0.0,
        "grav": 10,
        "desc": "Gotículas no ar de escritórios e escolas.",
    },
    {
        "id": "aerossol",
        "nome": "Aerossol",
        "grupo": "Transmissão",
        "custo": 11,
        "dbeta": 0.14,
        "dgamma": 0.0,
        "dviagem": 0.0,
        "grav": 14,
        "desc": "Permanece em ambientes fechados por horas.",
    },
    {
        "id": "nausea",
        "nome": "Náusea",
        "grupo": "Sintomas",
        "custo": 5,
        "dbeta": 0.03,
        "dgamma": 0.0,
        "dviagem": 0.0,
        "grav": 8,
        "desc": "As pessoas percebem que algo está errado. A pesquisa da cura acelera um pouco.",
    },
    {
        "id": "febre",
        "nome": "Febre longa",
        "grupo": "Resistência",
        "custo": 5,
        "dbeta": 0.0,
        "dgamma": -0.02,
        "dviagem": 0.0,
        "grav": 12,
        "desc": "As pessoas demoram mais para melhorar.",
    },
    {
        "id": "evasao",
        "nome": "Evasão imune",
        "grupo": "Resistência",
        "custo": 9,
        "dbeta": 0.0,
        "dgamma": -0.04,
        "dviagem": 0.0,
        "grav": 8,
        "desc": "O corpo leva ainda mais tempo para se livrar. A cura anda mais devagar.",
    },
    {
        "id": "pneumonia",
        "nome": "Pneumonia",
        "grupo": "Sintomas",
        "custo": 12,
        "dbeta": 0.06,
        "dgamma": -0.03,
        "dviagem": -0.008,
        "grav": 28,
        "desc": "Casos graves. O vírus se espalha, mas hospitais disparam o alerta.",
    },
    {
        "id": "hubs",
        "nome": "Rotas aéreas",
        "grupo": "Mobilidade",
        "custo": 8,
        "dbeta": 0.0,
        "dgamma": 0.0,
        "dviagem": 0.035,
        "grav": 2,
        "desc": "O vírus pega carona em aeroportos e ônibus.",
    },
]

GRAVIDADE_TRAIT = {tr["id"]: tr.get("grav", 0) for tr in TRAITS}


def _peso_viagem(a: dict, b: dict) -> float:
    if a["id"] == b["id"]:
        return 0.0
    dlat = a["lat"] - b["lat"]
    dlon = a["lon"] - b["lon"]
    dist = (dlat * dlat + dlon * dlon) ** 0.5
    hub = 1.6 if a["id"] in ("sp", "rj", "brasilia") or b["id"] in ("sp", "rj", "brasilia") else 1.0
    return hub / (1.0 + dist)


PESOS = {
    (a["id"], b["id"]): _peso_viagem(a, b)
    for a in CIDADES_BASE
    for b in CIDADES_BASE
}


@dataclass
class Cidade:
    id: str
    nome: str
    lat: float
    lon: float
    N: float
    S: float
    I: float
    R: float
    detectada: bool = False

    @property
    def prevalencia(self) -> float:
        return 100.0 * self.I / self.N

    @property
    def ataque(self) -> float:
        return 100.0 * (self.N - self.S) / self.N

    @property
    def status(self) -> str:
        if self.I < 1 and self.R < 1:
            return "saudavel"
        if self.prevalencia >= 8:
            return "crise"
        if self.prevalencia >= 2:
            return "epidemia"
        if self.I >= 1:
            return "infectada"
        return "recuperada"


@dataclass
class Mundo:
    t: float = 0.0
    dna: int = 16
    traits: List[str] = field(default_factory=list)
    cidades: Dict[str, Cidade] = field(default_factory=dict)
    news: List[str] = field(default_factory=list)
    log: List[dict] = field(default_factory=list)
    origem: Optional[str] = None
    fim: Optional[str] = None
    infectadas: int = 0
    ataque_global: float = 0.0
    dna_ganho_ataque: int = 0
    cura: float = 0.0
    medidas: List[str] = field(default_factory=list)
    historico: List[dict] = field(default_factory=list)

    def beta(self) -> float:
        b = 0.40
        for tr in TRAITS:
            if tr["id"] in self.traits:
                b += tr["dbeta"]
        if "alerta" in self.medidas:
            b *= 0.97
        if "campanha" in self.medidas:
            b *= 0.92
        return max(0.02, b)

    def gamma(self) -> float:
        g = 0.16
        for tr in TRAITS:
            if tr["id"] in self.traits:
                g += tr["dgamma"]
        g *= 1.0 + 1.6 * (self.cura / 100.0)
        return max(0.03, g)

    def viagem(self) -> float:
        v = 0.045
        for tr in TRAITS:
            if tr["id"] in self.traits:
                v += tr["dviagem"]
        if "alerta" in self.medidas:
            v *= 0.90
        if "aeroportos" in self.medidas:
            v *= 0.70
        grav = self.gravidade()
        v *= max(0.40, 1.0 - 0.40 * (grav / 100.0))
        return max(0.004, v)

    def gravidade(self) -> float:
        g = 8.0
        for tid in self.traits:
            g += GRAVIDADE_TRAIT.get(tid, 0)
        return min(100.0, g)

    def totais(self) -> Tuple[float, float, float, float]:
        S = sum(c.S for c in self.cidades.values())
        I = sum(c.I for c in self.cidades.values())
        R = sum(c.R for c in self.cidades.values())
        N = sum(c.N for c in self.cidades.values())
        return S, I, R, N


def mundo_novo() -> Mundo:
    cidades = {}
    for raw in CIDADES_BASE:
        cidades[raw["id"]] = Cidade(
            id=raw["id"],
            nome=raw["nome"],
            lat=raw["lat"],
            lon=raw["lon"],
            N=float(raw["N"]),
            S=float(raw["N"]),
            I=0.0,
            R=0.0,
        )
    m = Mundo(cidades=cidades)
    m.news.append("Silêncio nos boletins. Nenhum caso confirmado.")
    m.log.append({"t": 0.0, "evento": "Partida iniciada. Escolha a cidade-semente."})
    return m


def _registrar(m: Mundo, texto: str) -> None:
    m.news.insert(0, texto)
    m.news = m.news[:10]
    m.log.append({"t": round(m.t, 2), "evento": texto})


def iniciar_em(m: Mundo, cidade_id: str, sementes: float = 40.0) -> None:
    if m.origem is not None or m.fim:
        return
    c = m.cidades[cidade_id]
    semente = min(sementes, c.S)
    c.S -= semente
    c.I += semente
    c.detectada = True
    m.origem = cidade_id
    m.infectadas = 1
    _S, I, R, N = m.totais()
    m.historico.append({"t": 0.0, "I": I, "R": R, "S": _S, "cura": 0.0})
    _registrar(m, f"Paciente zero em {c.nome}. {semente:.0f} pessoas infectadas.")


def comprar_trait(m: Mundo, trait_id: str) -> bool:
    if m.fim or trait_id in m.traits:
        return False
    tr = next((x for x in TRAITS if x["id"] == trait_id), None)
    if tr is None or m.dna < tr["custo"]:
        return False
    m.dna -= tr["custo"]
    m.traits.append(trait_id)
    _registrar(m, f"Evolução: {tr['nome']} (−{tr['custo']} DNA).")
    return True


def _passo_cidade(c: Cidade, beta: float, gamma: float, h: float, lockdown: bool) -> None:
    y = np.array([c.S, c.I, c.R], dtype=float)
    beta_c = beta * (0.62 if lockdown else 1.0)

    def f(t: float, estado: np.ndarray) -> np.ndarray:
        return derivada_sir(t, estado, beta=beta_c, gamma=gamma, N=c.N)

    y = passo_rk4(f, 0.0, y, h)
    y = np.maximum(y, 0.0)
    massa = float(np.sum(y))
    if massa > 0:
        y *= c.N / massa
    c.S, c.I, c.R = float(y[0]), float(y[1]), float(y[2])


def _atualizar_medidas(m: Mundo) -> None:
    n_det = sum(1 for c in m.cidades.values() if c.detectada)
    if n_det >= 2 and "alerta" not in m.medidas:
        m.medidas.append("alerta")
        _registrar(m, "Ministério da Saúde emite alerta nacional. Viagens caem um pouco.")
    if n_det >= 5 and "aeroportos" not in m.medidas:
        m.medidas.append("aeroportos")
        _registrar(m, "Aeroportos reforçam triagem. Menos passageiros contaminados embarcam.")
    if m.ataque_global >= 8 and "campanha" not in m.medidas:
        m.medidas.append("campanha")
        _registrar(m, "Campanha de higiene nas capitais. Transmissão um pouco menor.")
    for c in m.cidades.values():
        chave = f"lockdown_{c.id}"
        if c.prevalencia > 8 and chave not in m.medidas:
            m.medidas.append(chave)
            _registrar(m, f"{c.nome} restringe aglomerações. O contágio local desacelera.")


def _atualizar_cura(m: Mundo, passo: float) -> None:
    n_det = sum(1 for c in m.cidades.values() if c.detectada)
    if n_det <= 0 or m.cura >= 100:
        return
    grav = m.gravidade()
    pesquisa = (0.10 + 0.055 * n_det + 0.035 * (grav / 25.0)) * passo
    if "evasao" in m.traits:
        pesquisa *= 0.52
    if "aerossol" in m.traits:
        pesquisa *= 0.88
    if "nausea" in m.traits:
        pesquisa *= 1.12
    if "pneumonia" in m.traits:
        pesquisa *= 1.18
    m.cura = min(100.0, m.cura + pesquisa)
    if m.cura >= 100:
        _registrar(m, "A cura chegou aos hospitais. As pessoas passam a melhorar bem mais rápido.")


def tick(m: Mundo, dias: float = 1.0, h: float = 0.25) -> None:
    if m.fim or m.origem is None:
        return
    restante = float(dias)
    ultimo_registro = m.historico[-1]["t"] if m.historico else -1.0
    while restante > 1e-9:
        passo = min(h, restante)
        beta, gamma, viagem = m.beta(), m.gamma(), m.viagem()
        for c in m.cidades.values():
            if c.I > 0.05 or c.R > 0:
                _passo_cidade(c, beta, gamma, passo, f"lockdown_{c.id}" in m.medidas)

        fluxos = []
        for a in m.cidades.values():
            if a.I < 1:
                continue
            for b in m.cidades.values():
                w = PESOS[(a.id, b.id)]
                if w <= 0:
                    continue
                novos = viagem * passo * a.I * (b.S / b.N) * w
                if novos >= 0.05:
                    fluxos.append((b.id, novos))
        agreg: Dict[str, float] = {}
        for cid, n in fluxos:
            agreg[cid] = agreg.get(cid, 0.0) + n
        for cid, n in agreg.items():
            b = m.cidades[cid]
            n = min(n, b.S)
            if n < 0.05:
                continue
            b.S -= n
            b.I += n
            if not b.detectada and b.I >= 5:
                b.detectada = True
                m.infectadas += 1
                m.dna += 3
                _registrar(m, f"{b.nome} confirma transmissão comunitária. +3 DNA.")

        _atualizar_cura(m, passo)
        m.t += passo
        restante -= passo

        if m.t - ultimo_registro >= 0.95 or restante <= 1e-9:
            _S, I, R, N = m.totais()
            m.historico.append({"t": m.t, "I": I, "R": R, "S": _S, "cura": m.cura})
            ultimo_registro = m.t

    _S, I, R, N = m.totais()
    ataque = 100.0 * (N - _S) / N
    ganho = int(ataque) - m.dna_ganho_ataque
    if ganho > 0:
        m.dna += ganho
        m.dna_ganho_ataque = int(ataque)
    m.ataque_global = ataque
    _atualizar_medidas(m)

    if I < 0.8 and m.t > 12:
        if ataque >= 70:
            m.fim = "vitoria"
            _registrar(m, "A onda passou: a maior parte do país já foi exposta.")
        else:
            m.fim = "cura"
            _registrar(m, "O surto morreu antes de se espalhar. R0 efetivo baixo demais.")
    elif ataque >= 85 and I > 100:
        m.fim = "vitoria"
        _registrar(m, "Colapso nacional: mais de 85% já tiveram contato com o vírus.")
    elif m.cura >= 100 and I < 80 and m.t > 18:
        m.fim = "cura"
        _registrar(m, "Com a cura nas ruas, o surto perde força e some.")


def resumo(m: Mundo) -> dict:
    S, I, R, N = m.totais()
    try:
        r0 = r0_basico(m.beta(), m.gamma())
    except Exception:
        r0 = None
    return {
        "t": m.t,
        "R0": r0,
        "beta": m.beta(),
        "gamma": m.gamma(),
        "S": S,
        "I": I,
        "R": R,
        "N": N,
        "ataque": m.ataque_global,
        "dna": m.dna,
        "cidades_atingidas": sum(1 for c in m.cidades.values() if c.I > 1 or c.R > 1),
        "origem": m.origem,
        "fim": m.fim,
        "traits": list(m.traits),
        "cura": m.cura,
        "gravidade": m.gravidade(),
        "medidas": list(m.medidas),
        "viagem": m.viagem(),
    }


def mundo_from_session(raw: dict) -> Mundo:
    cidades = {k: Cidade(**v) if not isinstance(v, Cidade) else v for k, v in raw["cidades"].items()}
    return Mundo(
        t=raw["t"],
        dna=raw["dna"],
        traits=list(raw["traits"]),
        cidades=cidades,
        news=list(raw["news"]),
        log=list(raw["log"]),
        origem=raw["origem"],
        fim=raw["fim"],
        infectadas=raw["infectadas"],
        ataque_global=raw["ataque_global"],
        dna_ganho_ataque=raw["dna_ganho_ataque"],
        cura=float(raw.get("cura") or 0.0),
        medidas=list(raw.get("medidas") or []),
        historico=list(raw.get("historico") or []),
    )


def mundo_to_session(m: Mundo) -> dict:
    return {
        "t": m.t,
        "dna": m.dna,
        "traits": m.traits,
        "cidades": {k: asdict(v) for k, v in m.cidades.items()},
        "news": m.news,
        "log": m.log,
        "origem": m.origem,
        "fim": m.fim,
        "infectadas": m.infectadas,
        "ataque_global": m.ataque_global,
        "dna_ganho_ataque": m.dna_ganho_ataque,
        "cura": m.cura,
        "medidas": m.medidas,
        "historico": m.historico,
    }
