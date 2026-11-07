"""
Testes de validação do motor numérico — Painel SIR
=================================================
A aproximação exponencial da fase inicial e a referência RK4 com
passo fino são usadas SOMENTE aqui, nunca como motor da aplicação.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine import (
    comparar_metodos,
    crescimento_exponencial_inicial,
    simular_sir,
)


def quase_igual(a: float, b: float, tol: float = 1e-2) -> bool:
    return abs(a - b) <= tol


def teste_1_conservacao() -> None:
    """Caso 1 — S+I+R permanece igual a N em todo o histórico."""
    print("=" * 60)
    print("TESTE 1: Conservacao S+I+R = N")
    N = 100_000.0
    r = simular_sir(
        beta=0.36,
        gamma=0.20,
        N=N,
        I0=50.0,
        t_final=160.0,
        h=0.25,
        metodo="rk4",
        parar_se_extinta=False,
    )
    assert r.sucesso, r.mensagem
    assert r.drift_max < 1e-6, f"drift_max={r.drift_max}"
    for s in r.historico:
        massa = s.S + s.I + s.R
        assert abs(massa - N) < 1e-6, f"t={s.t}: S+I+R={massa}"
    print(f"  Passos            : {r.n_passos}")
    print(f"  Drift maximo      : {r.drift_max:.3e}")
    print(f"  S+I+R final       : {r.S_final + r.I_final + r.R_final:.6f}")
    print("  [OK] PASSOU")


def teste_2_fase_inicial_exponencial() -> None:
    """
    Caso 2 — Com I0/N pequeno e horizonte curto, I(t) do RK4
    aproxima I0 exp((beta S0/N - gamma) t) com erro relativo < 2%.
    """
    print("=" * 60)
    print("TESTE 2: Fase inicial x exponencial analitica")
    N = 1_000_000.0
    I0 = 10.0
    S0 = N - I0
    beta, gamma = 0.40, 0.20
    t_final = 5.0
    r = simular_sir(
        beta=beta,
        gamma=gamma,
        N=N,
        I0=I0,
        t_final=t_final,
        h=0.01,
        metodo="rk4",
        parar_se_extinta=False,
        comparar_erro_local=False,
    )
    assert r.sucesso, r.mensagem
    I_num = r.I[-1]
    I_ref = crescimento_exponencial_inicial(t_final, I0, beta, gamma, S0, N)
    erro_rel = abs(I_num - I_ref) / I_ref
    assert erro_rel < 0.02, f"erro relativo={erro_rel:.4f} (num={I_num}, ref={I_ref})"
    print(f"  I analitico (t=5) : {I_ref:.4f}")
    print(f"  I RK4             : {I_num:.4f}")
    print(f"  Erro relativo     : {erro_rel * 100:.3f} %")
    print("  [OK] PASSOU")


def teste_3_r0_menor_que_um() -> None:
    """Caso 3 — R0 < 1: I decresce (epidemia nao decola)."""
    print("=" * 60)
    print("TESTE 3: R0 < 1, I decresce monotonicamente")
    r = simular_sir(
        beta=0.10,
        gamma=0.25,
        N=50_000.0,
        I0=200.0,
        t_final=80.0,
        h=0.1,
        metodo="rk4",
        parar_se_extinta=False,
    )
    assert r.sucesso, r.mensagem
    assert r.R0 is not None and r.R0 < 1.0
    for a, b in zip(r.I, r.I[1:]):
        assert b <= a + 1e-8, "I cresceu com R0 < 1"
    assert r.pico_I <= r.I[0] + 1e-8
    print(f"  R0                : {r.R0:.3f}")
    print(f"  I(0)              : {r.I[0]:.2f}")
    print(f"  I final           : {r.I_final:.2f}")
    print(f"  Pico I            : {r.pico_I:.2f} (deve coincidir com I0)")
    print("  [OK] PASSOU")


def teste_4_tratamento_de_falhas() -> None:
    """Caso 4 — entradas invalidas rejeitadas com mensagem clara."""
    print("=" * 60)
    print("TESTE 4: Tratamento de falhas")

    r_n = simular_sir(0.3, 0.2, N=0, I0=1)
    assert not r_n.sucesso

    r_g = simular_sir(0.3, gamma=0.0, N=1000, I0=1)
    assert not r_g.sucesso
    assert "R0" in r_g.mensagem.upper() or "gamma" in r_g.mensagem.lower()

    r_h = simular_sir(0.3, 0.2, N=1000, I0=1, h=-0.1)
    assert not r_h.sucesso

    r_i = simular_sir(0.3, 0.2, N=100, I0=150)
    assert not r_i.sucesso

    print("  N = 0        -> rejeitado")
    print("  gamma = 0    -> rejeitado (R0 indefinido)")
    print("  h < 0        -> rejeitado")
    print("  I0 > N       -> rejeitado")
    print("  [OK] PASSOU")


def teste_5_rk4_mais_preciso_que_euler() -> None:
    """
    Caso 5 — Com h grosseiro, o erro do pico de I do Euler e maior
    ou igual ao do RK4, medido contra RK4 com passo 10x menor.
    """
    print("=" * 60)
    print("TESTE 5: RK4 mais preciso que Euler (h grosseiro)")
    params = dict(beta=0.45, gamma=0.15, N=20_000.0, I0=40.0, t_final=90.0)
    ref = simular_sir(
        **params,
        h=0.05,
        metodo="rk4",
        comparar_erro_local=False,
        parar_se_extinta=False,
    )
    r_e, r_r = comparar_metodos(
        **params,
        h=0.5,
        comparar_erro_local=False,
        parar_se_extinta=False,
    )
    assert ref.sucesso and r_e.sucesso and r_r.sucesso, (
        ref.mensagem,
        r_e.mensagem,
        r_r.mensagem,
    )
    err_e = abs(r_e.pico_I - ref.pico_I)
    err_r = abs(r_r.pico_I - ref.pico_I)
    assert err_r <= err_e + 1e-9, (
        f"Esperava erro RK4 ({err_r}) <= erro Euler ({err_e})"
    )
    print(f"  Pico referencia (h=0.05): {ref.pico_I:.4f}")
    print(f"  Pico Euler    (h=0.50)  : {r_e.pico_I:.4f}  |erro|={err_e:.4f}")
    print(f"  Pico RK4      (h=0.50)  : {r_r.pico_I:.4f}  |erro|={err_r:.4f}")
    print("  [OK] PASSOU")


def main() -> int:
    testes = [
        teste_1_conservacao,
        teste_2_fase_inicial_exponencial,
        teste_3_r0_menor_que_um,
        teste_4_tratamento_de_falhas,
        teste_5_rk4_mais_preciso_que_euler,
    ]
    falhas = 0
    for t in testes:
        try:
            t()
        except AssertionError as exc:
            falhas += 1
            print(f"  [X] FALHOU: {exc}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"  [X] ERRO: {exc}")
    print("=" * 60)
    if falhas == 0:
        print(f"Todos os {len(testes)} testes passaram!")
        return 0
    print(f"{falhas} teste(s) falharam.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
