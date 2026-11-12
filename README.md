# Painel SIR

Laboratório numérico para o modelo epidemiológico **SIR** (Kermack–McKendrick, 1927). O motor de integração é implementado do zero; a interface só coleta parâmetros e apresenta resultados.

Projeto Integrador — Cálculo Numérico.

---

## 1. O problema real

Uma secretaria de saúde precisa de ordem de grandeza para três perguntas:

1. Qual o **pico** de infectados simultâneos?
2. Em que **dia** ele ocorre?
3. Que fração da população será atingida (**ataque final**)?

O modelo compartimental clássico é

```
dS/dt = −β S I / N
dI/dt =  β S I / N − γ I
dR/dt =  γ I
```

com `S + I + R = N` constante. Não existe fórmula fechada para `S(t)`, `I(t)`, `R(t)`. A órbita no plano S–I envolve a função W de Lambert; a série temporal exige um método numérico para EDOs.

O número de reprodução básico `R₀ = β/γ` decide o regime: se `R₀ > 1` a epidemia decola; se `R₀ < 1` ela regride. O limiar de imunidade de rebanho é `1 − 1/R₀`.

---

## 2. Método numérico

| Método | Ordem global | Papel |
|--------|----------------|-------|
| Euler explícito | O(h) | comparação / didática |
| Runge-Kutta 4 | O(h⁴) | motor principal |

RK4 equilibra precisão e clareza: quatro avaliações da derivada por passo, erro local O(h⁵). O SIR é suave e não-linear (termo `S·I`); Euler com passo grosso **atrasa o pico** e distorce a amplitude.

**Critérios de parada**

- horizonte `t_final`
- epidemia extinta (`I` abaixo da tolerância, já em decaimento)
- número máximo de iterações

**Falhas tratadas:** `β`, `γ`, `N`, `I0` inválidos; `γ = 0` (R₀ indefinido); `h ≤ 0`; divergência (NaN/inf); compartimento muito negativo; estouro de iterações.

**Limitações do modelo e do método**

- População homogênea, `β` constante, sem natalidade, mortalidade extra, quarentena ou idade.
- Euler instável / negativo com `h` grande.
- Passo fixo: o trecho íngreme do pico poderia se beneficiar de passo adaptativo (fora do escopo).
- Útil para ordem de grandeza, não para previsão operacional.

---

## 3. Estrutura

```
projeto_calculo/
├── engine.py          # SIR + Euler + RK4 (sem UI)
├── app.py             # dashboard Streamlit
├── tests.py           # validação × referência analítica / passo fino
├── requirements.txt
└── README.md
```

`engine.py` não chama `scipy.integrate`, `odeint`, `solve_ivp` nem equivalentes. A exponencial da fase inicial e a referência com passo fino existem **somente** em `tests.py`.

---

## 4. Como executar

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python tests.py
streamlit run app.py
```

Interface em `http://localhost:8501`.

---

## 5. Como usar o painel

1. Ajuste população, infectados iniciais, `β` e `γ`.
2. Escolha horizonte e passo `h`.
3. Clique **Simular epidemia**.
4. Leia R₀, pico, dia do pico e ataque final.
5. Confira a série temporal, o plano de fase S×I, o erro local e a aba Euler × RK4.

Cenário-base: gripe-like, `R₀ ≈ 1,8` (`β = 0,36`, `γ = 0,20`), `N = 100 000`, `I(0) = 50`.

---

## 6. Validação

| Caso | O que verifica |
|------|----------------|
| 1 | Conservação: `|S+I+R−N| < 10⁻⁶` em todo o histórico |
| 2 | Fase inicial: RK4 ≈ `I₀ exp((β S₀/N − γ) t)` (erro relativo < 2%) |
| 3 | `R₀ < 1`: `I(t)` decresce (epidemia não decola) |
| 4 | `N = 0`, `γ = 0`, `h < 0`, `I0 > N` rejeitados |
| 5 | Com `h` grosso, erro do pico de Euler ≥ erro do RK4 vs. RK4 com passo 10× menor |

---

## 7. Prints

Após rodar o app, substitua por capturas reais:

1. Dashboard com KPIs e curvas S/I/R
2. Plano de fase com a reta `S = N/R₀`
3. Gráfico de erro local e tabela de passos
4. Aba Euler × RK4

---

## 8. Apresentação (~10 min)

Deck web no visual do painel, com prints reais e o mapeamento dos requisitos:

```bash
# na pasta do projeto
python -m http.server 8765 --bind 127.0.0.1 --directory apresentacao
```

Abra `http://127.0.0.1:8765`. Setas ou espaço avançam; **F** entra em tela cheia.

1. **Problema** — pico, dia, ataque; SIR sem solução no tempo.
2. **Método** — Euler e RK4 em `engine.py`; `R₀ = β/γ`.
3. **Laboratório** — print do dashboard, fase e erro Euler × RK4.
4. **Como funciona** — fórmulas da aba Ajuda.
5. **Validação** — 5 testes vs referência analítica / passo fino.
6. **Limites** — homogeneidade, `β` constante, passo fixo.

---

## 9. Divisão de tarefas (modelo)

| Integrante | Responsabilidade |
|------------|------------------|
| A | `engine.py` (EDOs, Euler, RK4, falhas) |
| B | `tests.py` e análise de erro |
| C | `app.py` (dashboard, gráficos) |
| D | README, prints, apresentação |

Ajuste os nomes conforme o grupo.

---

## 10. Bibliotecas

| Uso | Biblioteca |
|-----|------------|
| Interface | Streamlit |
| Gráficos | Plotly |
| Tabelas | Pandas |
| Arrays / ops básicas | NumPy |
| **Proibido na engine** | `scipy.integrate`, `odeint`, `solve_ivp` e equivalentes |

---

## 11. Publicar

O Vercel **não** executa `streamlit run app.py`. Ele procura em `app.py` uma variável `app` / `application` / `handler` de Flask, FastAPI ou Django. Por isso o erro `PYTHON_ENTRYPOINT_NOT_FOUND`.

Este repositório está configurado para o Vercel servir o painel **no navegador** (Stlite / Pyodide): mesma interface, sem servidor Python. A primeira abertura baixa o runtime e pode levar ~30 s.

Na Vercel: Framework Preset **Other**, e deixe o `vercel.json` mandar no build. A apresentação fica em `/apresentacao`.

Para o app com servidor Streamlit de verdade (mais rápido depois do load):

1. https://share.streamlit.io
2. New app → `Anhussi/projeto-calculo` → arquivo `app.py`
