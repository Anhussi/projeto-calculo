"""
Motor numérico próprio — Painel SIR
===================================
Integra o sistema epidemiológico SIR (Kermack–McKendrick, 1927)
com métodos de EDO implementados do zero:

  • Euler explícito
  • Runge-Kutta de 4ª ordem (RK4)

O modelo não possui solução explícita no tempo: a órbita no plano
S–I envolve a função W de Lambert, mas a série temporal S(t), I(t),
R(t) exige integração numérica. Por isso RK4 é o motor da aplicação.

Proibido: scipy.integrate, odeint, solve_ivp e equivalentes.
NumPy é usado apenas como contêiner de arrays e operações básicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


State = np.ndarray  # [S, I, R]
DerivFn = Callable[[float, State], State]


@dataclass
class StepRecord:
    """Um passo da integração (tabela de convergência / erro)."""

    n: int
    t: float
    S: float
    I: float
    R: float
    conservacao: float
    erro_local: Optional[float] = None


@dataclass
class SimulationResult:
    """Resultado completo de uma simulação SIR."""

    sucesso: bool
    mensagem: str
    metodo: str
    beta: float
    gamma: float
    N: float
    R0: Optional[float]
    limiar_rebanho: Optional[float]
    t_final_efetivo: float
    pico_I: float
    t_pico: float
    ataque_final: float
    S_final: float
    I_final: float
    R_final: float
    drift_max: float
    tempos: List[float] = field(default_factory=list)
    S: List[float] = field(default_factory=list)
    I: List[float] = field(default_factory=list)
    R: List[float] = field(default_factory=list)
    historico: List[StepRecord] = field(default_factory=list)
    erro_medio_local: Optional[float] = None
    n_passos: int = 0
    extinta: bool = False


class EngineError(Exception):
    """Falha controlada do motor numérico."""

    def __init__(self, codigo: str, mensagem: str) -> None:
        self.codigo = codigo
        self.mensagem = mensagem
        super().__init__(f"[{codigo}] {mensagem}")


# ---------------------------------------------------------------------------
# Modelo SIR
# ---------------------------------------------------------------------------

def derivada_sir(
    t: float,
    estado: State,
    beta: float,
    gamma: float,
    N: float,
) -> State:
    """
    Sistema autônomo de EDOs do modelo SIR clássico.

        dS/dt = -β · S · I / N
        dI/dt =  β · S · I / N − γ · I
        dR/dt =  γ · I

    Conservação: dS/dt + dI/dt + dR/dt = 0  ⇒  S+I+R = N (constante).
    """
    S, I, _R = float(estado[0]), float(estado[1]), float(estado[2])
    if N <= 0:
        raise EngineError("N_INVALIDO", "A população N deve ser maior que zero.")

    infeccao = beta * S * I / N
    recuperacao = gamma * I
    return np.array([-infeccao, infeccao - recuperacao, recuperacao], dtype=float)


def r0_basico(beta: float, gamma: float) -> float:
    """Número de reprodução básico R0 = β/γ. Indefinido se γ = 0."""
    if gamma == 0:
        raise EngineError(
            "R0_INDEFINIDO",
            "gamma = 0 torna R0 indefinido (recuperação nula).",
        )
    return beta / gamma


def limiar_imunidade_rebanho(R0: float) -> float:
    """Fração mínima a imunizar para R_efetivo < 1: 1 − 1/R0 (0 se R0 ≤ 1)."""
    if R0 <= 1.0:
        return 0.0
    return 1.0 - 1.0 / R0


def validar_parametros(
    beta: float,
    gamma: float,
    N: float,
    I0: float,
    R0_init: float,
    h: float,
    t_final: float,
) -> None:
    if beta < 0:
        raise EngineError("BETA_INVALIDO", "A taxa de transmissão beta não pode ser negativa.")
    if gamma < 0:
        raise EngineError("GAMMA_INVALIDO", "A taxa de recuperação gamma não pode ser negativa.")
    if gamma == 0:
        raise EngineError(
            "R0_INDEFINIDO",
            "gamma = 0 torna R0 indefinido e impede a recuperação.",
        )
    if N <= 0:
        raise EngineError("N_INVALIDO", "A população N deve ser maior que zero.")
    if I0 < 0 or R0_init < 0:
        raise EngineError(
            "COMPARTIMENTO_INVALIDO",
            "I0 e R(0) não podem ser negativos.",
        )
    if I0 + R0_init > N:
        raise EngineError(
            "COMPARTIMENTO_INVALIDO",
            "I0 + R(0) não pode exceder a população N.",
        )
    if h <= 0:
        raise EngineError("PASSO_INVALIDO", "O passo h deve ser positivo.")
    if t_final <= 0:
        raise EngineError("HORIZONTE_INVALIDO", "O horizonte t_final deve ser positivo.")


# ---------------------------------------------------------------------------
# Métodos numéricos (implementados do zero)
# ---------------------------------------------------------------------------

def passo_euler(f: DerivFn, t: float, y: State, h: float) -> State:
    """
    Um passo do método de Euler explícito.

        y_{n+1} = y_n + h · f(t_n, y_n)

    Erro local O(h²), global O(h). Barato, mas atrasa o pico e pode
    produzir compartimentos negativos se h for grande demais.
    """
    if h <= 0:
        raise EngineError("PASSO_INVALIDO", "O passo h deve ser positivo.")
    return y + h * f(t, y)


def passo_rk4(f: DerivFn, t: float, y: State, h: float) -> State:
    """
    Um passo do método de Runge-Kutta de 4ª ordem (RK4).

        k1 = f(t,       y)
        k2 = f(t + h/2, y + h/2 · k1)
        k3 = f(t + h/2, y + h/2 · k2)
        k4 = f(t + h,   y + h   · k3)
        y_{n+1} = y + (h/6) · (k1 + 2 k2 + 2 k3 + k4)

    Erro local O(h⁵), global O(h⁴). Adequado para o SIR, que é suave
    mas não-linear (o termo S·I).
    """
    if h <= 0:
        raise EngineError("PASSO_INVALIDO", "O passo h deve ser positivo.")
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _escolher_passo(metodo: str) -> Callable[[DerivFn, float, State, float], State]:
    chave = metodo.strip().lower()
    if chave in ("euler", "e"):
        return passo_euler
    if chave in ("rk4", "runge-kutta", "runge_kutta", "rk"):
        return passo_rk4
    raise EngineError(
        "METODO_DESCONHECIDO",
        f"Método '{metodo}' não suportado. Use 'euler' ou 'rk4'.",
    )


def _resultado_vazio(metodo: str, mensagem: str, **kwargs) -> SimulationResult:
    return SimulationResult(
        sucesso=False,
        mensagem=mensagem,
        metodo=metodo,
        beta=kwargs.get("beta", 0.0),
        gamma=kwargs.get("gamma", 0.0),
        N=kwargs.get("N", 0.0),
        R0=None,
        limiar_rebanho=None,
        t_final_efetivo=0.0,
        pico_I=0.0,
        t_pico=0.0,
        ataque_final=0.0,
        S_final=0.0,
        I_final=0.0,
        R_final=0.0,
        drift_max=0.0,
    )


# ---------------------------------------------------------------------------
# Integração
# ---------------------------------------------------------------------------

def simular_sir(
    beta: float,
    gamma: float,
    N: float,
    I0: float,
    *,
    R0_init: float = 0.0,
    t_final: float = 160.0,
    h: float = 0.25,
    metodo: str = "rk4",
    tol: float = 1e-6,
    tol_extincao: float = 1e-8,
    max_iter: int = 200_000,
    comparar_erro_local: bool = True,
    parar_se_extinta: bool = True,
) -> SimulationResult:
    """
    Integra o SIR até t_final (ou até a epidemia se extinguir).

    Critérios de parada
    -------------------
    • t ≥ t_final
    • I abaixo de `tol_extincao` e já em decaimento (epidemia extinta)
    • número máximo de passos (`max_iter`)

    Tratamento de falhas
    --------------------
    • parâmetros inválidos (beta, gamma, N, I0, h, t_final)
    • gamma = 0 (R0 indefinido)
    • divergência (NaN / inf / compartimento muito negativo)
    • estouro de iterações
    """
    nome = metodo.strip().upper()
    if nome in ("RUNGE-KUTTA", "RUNGE_KUTTA", "RK"):
        nome = "RK4"
    elif nome == "E":
        nome = "EULER"

    try:
        validar_parametros(beta, gamma, N, I0, R0_init, h, t_final)
        passo = _escolher_passo(metodo)
        R0 = r0_basico(beta, gamma)
        rebanho = limiar_imunidade_rebanho(R0)
        S0 = N - I0 - R0_init
        if S0 < 0:
            raise EngineError(
                "COMPARTIMENTO_INVALIDO",
                "S(0) = N − I0 − R(0) resultou negativo.",
            )

        def f(t: float, estado: State) -> State:
            return derivada_sir(t, estado, beta=beta, gamma=gamma, N=N)

        y = np.array([S0, I0, R0_init], dtype=float)
        t = 0.0
        historico: List[StepRecord] = []
        tempos = [0.0]
        Ss = [float(y[0])]
        Is = [float(y[1])]
        Rs = [float(y[2])]
        erros: List[float] = []

        pico_I = float(y[1])
        t_pico = 0.0
        drift_max = 0.0
        atingiu_pico = False

        historico.append(
            StepRecord(
                n=0,
                t=0.0,
                S=float(y[0]),
                I=float(y[1]),
                R=float(y[2]),
                conservacao=abs(float(y[0] + y[1] + y[2]) - N),
                erro_local=None,
            )
        )

        n = 0
        while t < t_final - 0.5 * h and n < max_iter:
            n += 1
            h_uso = min(h, t_final - t)
            if h_uso <= 0:
                break

            y_novo = passo(f, t, y, h_uso)

            erro_local: Optional[float] = None
            if comparar_erro_local:
                y_rk = y_novo if passo is passo_rk4 else passo_rk4(f, t, y, h_uso)
                y_eu = y_novo if passo is passo_euler else passo_euler(f, t, y, h_uso)
                diff = y_rk - y_eu
                erro_local = float(np.sqrt(np.sum(diff * diff)))
                erros.append(erro_local)

            t_novo = t + h_uso

            if not np.all(np.isfinite(y_novo)):
                return SimulationResult(
                    sucesso=False,
                    mensagem=(
                        "A integração divergiu (NaN ou infinito). "
                        "Reduza o passo h ou revise os parâmetros."
                    ),
                    metodo=nome,
                    beta=beta,
                    gamma=gamma,
                    N=N,
                    R0=R0,
                    limiar_rebanho=rebanho,
                    t_final_efetivo=t,
                    pico_I=pico_I,
                    t_pico=t_pico,
                    ataque_final=float(y[2]) / N,
                    S_final=float(y[0]),
                    I_final=float(y[1]),
                    R_final=float(y[2]),
                    drift_max=drift_max,
                    tempos=tempos,
                    S=Ss,
                    I=Is,
                    R=Rs,
                    historico=historico,
                    erro_medio_local=float(np.mean(erros)) if erros else None,
                    n_passos=n - 1,
                )

            # Euler com h grande pode produzir valores negativos leves
            if np.any(y_novo < -1e-6 * N):
                return SimulationResult(
                    sucesso=False,
                    mensagem=(
                        "Compartimento negativo: o passo h é grande demais "
                        "para este método. Reduza h ou use RK4."
                    ),
                    metodo=nome,
                    beta=beta,
                    gamma=gamma,
                    N=N,
                    R0=R0,
                    limiar_rebanho=rebanho,
                    t_final_efetivo=t,
                    pico_I=pico_I,
                    t_pico=t_pico,
                    ataque_final=float(max(y[2], 0.0)) / N,
                    S_final=float(y[0]),
                    I_final=float(y[1]),
                    R_final=float(y[2]),
                    drift_max=drift_max,
                    tempos=tempos,
                    S=Ss,
                    I=Is,
                    R=Rs,
                    historico=historico,
                    erro_medio_local=float(np.mean(erros)) if erros else None,
                    n_passos=n - 1,
                )

            # Clipagem apenas de negativos numéricos irrelevantes (não altera a ordem do método)
            y_novo = np.where(y_novo >= -1e-12 * N, np.maximum(y_novo, 0.0), y_novo)
            drift = abs(float(np.sum(y_novo)) - N)
            drift_max = max(drift_max, drift)

            y = y_novo
            t = t_novo

            if float(y[1]) > pico_I + tol:
                pico_I = float(y[1])
                t_pico = t
                atingiu_pico = True
            elif float(y[1]) < pico_I - tol:
                atingiu_pico = True

            tempos.append(t)
            Ss.append(float(y[0]))
            Is.append(float(y[1]))
            Rs.append(float(y[2]))
            historico.append(
                StepRecord(
                    n=n,
                    t=t,
                    S=float(y[0]),
                    I=float(y[1]),
                    R=float(y[2]),
                    conservacao=drift,
                    erro_local=erro_local,
                )
            )

            i_atual = float(y[1])
            em_decaimento = atingiu_pico and i_atual <= pico_I
            if (
                parar_se_extinta
                and i_atual < max(tol_extincao, tol * N)
                and em_decaimento
                and t > h
            ):
                return SimulationResult(
                    sucesso=True,
                    mensagem="Simulação concluída: epidemia extinta antes do horizonte.",
                    metodo=nome,
                    beta=beta,
                    gamma=gamma,
                    N=N,
                    R0=R0,
                    limiar_rebanho=rebanho,
                    t_final_efetivo=t,
                    pico_I=pico_I,
                    t_pico=t_pico,
                    ataque_final=float(y[2]) / N,
                    S_final=float(y[0]),
                    I_final=float(y[1]),
                    R_final=float(y[2]),
                    drift_max=drift_max,
                    tempos=tempos,
                    S=Ss,
                    I=Is,
                    R=Rs,
                    historico=historico,
                    erro_medio_local=float(np.mean(erros)) if erros else None,
                    n_passos=n,
                    extinta=True,
                )

        if n >= max_iter and t < t_final - 0.5 * h:
            return SimulationResult(
                sucesso=False,
                mensagem=(
                    f"Número máximo de iterações ({max_iter}) atingido "
                    "antes do horizonte. Aumente max_iter ou o passo h."
                ),
                metodo=nome,
                beta=beta,
                gamma=gamma,
                N=N,
                R0=R0,
                limiar_rebanho=rebanho,
                t_final_efetivo=t,
                pico_I=pico_I,
                t_pico=t_pico,
                ataque_final=float(y[2]) / N,
                S_final=float(y[0]),
                I_final=float(y[1]),
                R_final=float(y[2]),
                drift_max=drift_max,
                tempos=tempos,
                S=Ss,
                I=Is,
                R=Rs,
                historico=historico,
                erro_medio_local=float(np.mean(erros)) if erros else None,
                n_passos=n,
            )

        return SimulationResult(
            sucesso=True,
            mensagem="Simulação concluída até o horizonte.",
            metodo=nome,
            beta=beta,
            gamma=gamma,
            N=N,
            R0=R0,
            limiar_rebanho=rebanho,
            t_final_efetivo=t,
            pico_I=pico_I,
            t_pico=t_pico,
            ataque_final=float(y[2]) / N,
            S_final=float(y[0]),
            I_final=float(y[1]),
            R_final=float(y[2]),
            drift_max=drift_max,
            tempos=tempos,
            S=Ss,
            I=Is,
            R=Rs,
            historico=historico,
            erro_medio_local=float(np.mean(erros)) if erros else None,
            n_passos=n,
            extinta=float(y[1]) < max(tol_extincao, tol * N),
        )

    except EngineError as exc:
        return _resultado_vazio(nome, str(exc), beta=beta, gamma=gamma, N=N)


def comparar_metodos(
    beta: float,
    gamma: float,
    N: float,
    I0: float,
    **kwargs,
) -> Tuple[SimulationResult, SimulationResult]:
    """Roda o mesmo cenário com Euler e RK4."""
    r_euler = simular_sir(beta, gamma, N, I0, metodo="euler", **kwargs)
    r_rk4 = simular_sir(beta, gamma, N, I0, metodo="rk4", **kwargs)
    return r_euler, r_rk4


def crescimento_exponencial_inicial(
    t: float,
    I0: float,
    beta: float,
    gamma: float,
    S0: float,
    N: float,
) -> float:
    """
    Aproximação analítica da fase inicial: S(t) ≈ S0 ⇒

        I(t) ≈ I0 · exp((β S0 / N − γ) t)

    Válida somente enquanto I ≪ N e S permanece próximo de S0.
    Usada SOMENTE em testes — nunca como motor da aplicação.
    """
    taxa = beta * S0 / N - gamma
    return I0 * float(np.exp(taxa * t))
