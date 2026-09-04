# Investigación SSRN/académica — trading algorítmico, breakout, drawdown, prop firms
**Fecha:** 2026-09-03 noche → 2026-09-04. **Hecho mientras el usuario dormía**, a pedido explícito.
**Método:** búsquedas dirigidas en SSRN (papers.ssrn.com) + arXiv + fuentes técnicas MT5, vía WebSearch
(SSRN bloqueó el fetch directo de PDFs con 403, así que los hallazgos son de abstracts/resúmenes
indexados, no de los papers completos línea por línea — donde importa, lo marco explícitamente).

**Nada de esto se implementó.** Es solo insumo para decidir juntos qué probar, siguiendo la regla de
la sección 20 de CLAUDE.md ("evidencia externa + evidencia propia combinadas, no se cambia código a
ciegas"). Ningún parámetro del bot fue tocado.

---

## 0. Resumen ejecutivo (lo más importante primero)

1. **Hallazgo casi calcado al problema que tenemos hoy**: un paper (Rounce, 2026) documenta que una
   estrategia de breakout FX en vivo divergió del backtest porque el trailing-stop se rechazaba en
   MetaTrader 5 en producción (modificación de SL rechazada cuando el nivel objetivo queda detrás del
   precio actual) — es el mismo género de problema que el `Retcode 10016` que llevamos 2 días sin poder
   validar porque el guard de drawdown bloquea todo intento de orden. **Acción sin riesgo que se puede
   hacer ahora mismo, sin esperar a que el guard se libere**: leer `SYMBOL_TRADE_STOPS_LEVEL` y
   `SYMBOL_TRADE_FREEZE_LEVEL` de cada símbolo vía la API de MT5 (es solo lectura, no dispara una orden)
   para saber de antemano si el ATR×0.3 va a chocar con el mínimo del broker antes de gastar el primer
   intento real en descubrirlo. Ver sección 3.

2. **Alerta que vale la pena cruzar con nuestros propios datos**: un paper (Costa, 2026) sobre EURUSD,
   GBPJPY, USDCAD, USDJPY, AUDUSD y Gold (2016-2026, ~3,800 breakouts) encuentra que los majors FX
   **invalidan/barren liquidez en más del 75% de los breakouts** (mean-reversion, no continuación),
   mientras que **Gold sí muestra continuación direccional real**. Nuestra config actual tiene 7 pares
   forex + XAUUSD + NAS100/US30. Si esto se sostiene, XAUUSD y los índices podrían tener una tasa de
   acierto estructuralmente mejor que los majors puros para una estrategia de breakout — vale la pena
   revisar el desglose por símbolo en `memory/scan_stats.json`/scores históricos cuando haya suficiente
   muestra, no solo confiar en la literatura. Ver sección 2.

3. **Bandera sobre el propio proceso de reoptimización**: `ATR_MULT_SL=0.3` / `RR_MULT=25.0` /
   `DONCHIAN_N=1` se fijaron tras corregir un bug de paridad D1/H4 la sesión pasada. La literatura de
   backtest-overfitting (Bailey & López de Prado — Deflated Sharpe Ratio, Probability of Backtest
   Overfitting) documenta exactamente este patrón de riesgo: "arreglo un bug → reoptimizo → el backtest
   vuelve a verse bien" puede producir una config que luce válida en el histórico pero está sobreajustada
   al ruido de esa reoptimización puntual, no a una ventaja real. No es una acusación de que esté mal —
   es una razón concreta para, cuando haya más trades reales, pedir explícitamente un chequeo de
   out-of-sample/PBO antes de tratar esta config como definitiva. Ver sección 6.

4. **Confirmación externa de una decisión ya tomada**: no existe validación académica real de "Smart
   Money Concepts" / order blocks como metodología institucional formal — coincide con la auditoría
   propia de la sección 16 de CLAUDE.md que ya desactivó Lunar/Elliott/Chaos por falta de evidencia.
   No cambia nada, pero refuerza que la política del proyecto (nunca añadir un agente/concepto sin
   evidencia de backtest real) está alineada con el consenso externo. Ver sección 7.

5. **Contexto realista sobre prop firms**: pass rates públicos en challenges de 2 fases (FTMO y
   comparables) están en el rango 8-15%, y solo ~7% de todos los que compran un challenge llegan a
   cobrar un payout alguna vez (dataset FPFX Technology, ~300K cuentas). Esto no es un paper académico
   per sé (son cifras autoreportadas por las firmas, no auditadas independientemente), pero sirve como
   ancla de expectativas realista. Ver sección 8.

6. **DONCHIAN_N=1 es atípico frente a la literatura clásica** (Turtle System usa N=20/55, la mayoría de
   papers de time-series momentum operan en horizontes de días-a-meses, no un breakout de 1 solo bar).
   Esto no invalida la config — el bot llegó ahí por una razón concreta documentada (bug de paridad) —
   pero si en el futuro se explora volver a un N mayor, hay bibliografía de respaldo real para ese
   experimento específico. Ver sección 1 y 6.

---

## 1. El paper más directamente comparable a nuestra estrategia

**"Evaluating the Performance of a Donchian Channel Breakout Strategy with ATR-Based Risk Management"**
— Nitish Poluri, SSRN (feb 2026). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6272239

Evalúa exactamente el mismo tipo de sistema que corre el bot: Donchian breakout + filtro de régimen de
volatilidad basado en ATR + risk management basado en ATR. El abstract confirma que el filtro de
volatilidad y los controles de riesgo "son importantes en sistemas de tendencia que siguen mercados
cripto" — no pude extraer del resumen indexado los números exactos de N del canal, múltiplo de ATR, o
win rate/profit factor específicos (SSRN bloqueó el PDF con 403 y el snippet de búsqueda no los trae).
**Pendiente**: si se quiere el detalle numérico completo, habría que crear cuenta gratuita en SSRN y
descargar el PDF manualmente — no es algo que pueda hacer yo sin login. Vale la pena que el usuario lo
revise directamente, es la comparación más cercana a nuestra propia arquitectura que encontré.

Contraste con la literatura "clásica" de breakout (Turtle Trading / regla de las 4 semanas de Donchian):
esos sistemas usan N=20 (entrada) / N=10-55 (salida), operando en marcos de días. Nuestro
`DONCHIAN_N=1` es conceptualmente distinto — más cerca de un disparador de momentum de una sola vela
que de un "breakout de canal" en el sentido clásico. Eso no lo hace inválido (el mercado y el timeframe
también son distintos), pero si algún día se quiere comparar contra la literatura de forma más limpia,
valdría la pena backtestear también N=5/N=10/N=20 como referencia, no solo confiar en que N=1 es el
óptimo porque fue lo que salió de la última reoptimización.

---

## 2. Forex breakouts: ¿tendencia real o trampa de liquidez?

**"The Illusion of Breakouts: Empirical Evidence of Institutional Liquidity Capture in Major Currency
Pairs"** — Rodrigo Costa, SSRN (2026). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6592020

- Muestra: EURUSD, GBPJPY, USDCAD, USDJPY, AUDUSD y Gold, 2016-2026 (~10 años), >3,800 breakouts
  mapeados contra un "rango institucional de 20 días".
- Hallazgo central: los majors FX **actúan primero como mecanismo de mean-reversion** — invalidan el
  breakout y barren la liquidez en **más del 75%** de los casos mapeados.
- Gold es la excepción: muestra "anomalía de flujo direccional macroeconómico", consolidando breakouts
  verdaderos con mayor frecuencia que los pares FX.

**Por qué importa para nosotros**: nuestra lista de pares (EURUSD, GBPUSD, XAUUSD, USDJPY, GBPJPY,
NAS100, US30, NZDUSD/AUDUSD) mezcla exactamente los dos regímenes que este paper distingue — majors FX
puros (donde el breakout, según este estudio, es mayormente una trampa) y Gold/índices (donde podría
haber continuación real). No es una recomendación de sacar pares — es una hipótesis concreta y
verificable con nuestros propios datos: cuando haya masa suficiente de trades reales, desglosar win
rate por símbolo y ver si XAUUSD/NAS100/US30 rinden estructuralmente mejor que EURUSD/GBPUSD/USDJPY/
GBPJPY en breakout. Si el patrón se sostiene con datos propios, ahí sí habría base para decidir algo.

Complementario: **"Liquidity-Driven Breakout Reliability: Why Price Moves Where Liquidity Is Missing"**
— Mittal & Choudhary, SSRN (2026). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5962358
Usa un modelo de supervivencia Kaplan-Meier: breakouts que entran en zonas de **bajo volumen/nodo de
liquidez** tienen >70% de probabilidad de continuar; los que entran en zonas de **alto volumen** se
comportan casi como aleatorios (~50/50). Conclusión del paper: lo que mueve el precio de forma
persistente es la *ausencia* de liquidez a vencer, no la agresividad del flujo de órdenes. Esto es un
filtro de entrada potencial (evaluar el volume profile/nodo de liquidez en el punto de breakout antes de
tomarlo) — coherente con el ya existente `POI zone filter` del bot (sección 3 de CLAUDE.md), que ya
filtra por proximidad a zonas de OB, pero no explícitamente por volumen bajo/alto en ese nivel. No lo
implementaría sin backtest — es una idea a explorar, no un cambio a hacer ahora.

---

## 3. El paralelo directo con nuestro bug no-validado (Retcode 10016)

**"What Survived Live Reconciliation: Auditing a Systematic FX Strategy"** — Kelvin Adjei Rounce, SSRN
(2026). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5805443

- El paper audita una estrategia de FX breakout por sesión, tras **3 meses de simulación
  live-paper-trading** que divergió materialmente del backtest original.
- Causa raíz encontrada: la regla de trailing-stop del backtest **era inejecutable en MT5 en vivo**
  — las modificaciones de SL eran rechazadas cuando el nivel objetivo quedaba detrás del precio actual
  (exactamente el tipo de rechazo que produce un `Retcode 10016`/stops inválidos en la vida real).
- Solución que encontraron: reemplazar el trailing-stop dinámico por un "close-only trail" (cerrar la
  posición manualmente en vez de mover el SL) — esto subió el trade ganador promedio simulado de $138 a
  $260 (+88%) una vez corregido.

**Esto es lo más accionable de toda la búsqueda, y no requiere que el guard de drawdown se libere
primero.** Es pura lectura vía API MT5, sin arriesgar nada:

```python
# Ejemplo de lo que se podría chequear ahora mismo, sin abrir ninguna orden:
import MetaTrader5 as mt5
for symbol in ["EURUSD","GBPUSD","XAUUSD","USDJPY","GBPJPY","NAS100","US30","NZDUSD"]:
    info = mt5.symbol_info(symbol)
    print(symbol, "stops_level=", info.trade_stops_level, "freeze_level=", info.trade_freeze_level)
```

Si `ATR(14) × 0.3` para algún símbolo da una distancia de SL en puntos **menor** que
`trade_stops_level` de ese símbolo en Axi Demo, ese símbolo va a rechazar la orden con 10016 la primera
vez que se intente — no por un bug de código, sino porque el SL calculado nunca fue lo bastante ancho
para ese broker/símbolo específico. Esto se puede saber HOY, sin esperar a que el guard libere, y sin
gastar el primer intento real "a ciegas" en descubrirlo. Lo dejo como tarea concreta para la próxima
sesión, no lo ejecuté yo mismo porque toca `metatrader_connector.py`/MT5 y las reglas del proyecto piden
no tocar MT5 con Python sin que el usuario esté presente y logueado en la GUI (sección 4 de CLAUDE.md) —
pero es una lectura, no una escritura, así que probablemente sea seguro hacerlo con el usuario presente
mañana.

Complementario (fuente técnica, no académica pero útil para lo mismo): la causa real de un `10016`
según documentación MQL5 son 3 cosas — (a) SL/TP más cerca del precio que `SYMBOL_TRADE_STOPS_LEVEL`,
(b) SL/TP del lado equivocado (BUY: SL debajo del Bid, TP arriba del Ask; SELL: al revés), o (c) dentro
de `SYMBOL_TRADE_FREEZE_LEVEL` (no se puede modificar un stop mientras el precio está a esa distancia).
El mecanismo de retry adaptativo que ya existe en `connectors/metatrader_connector.py` (líneas ~299-348,
de la sesión anterior) parece apuntar en la dirección correcta según esta descripción — pero sigue sin
validarse en vivo.

---

## 4. Base académica del trend-following / momentum (contexto, no acción directa)

- **"Time Series Momentum"** — Moskowitz, Ooi, Pedersen (el paper fundacional, AQR). Encuentra momentum
  persistente en futuros de equity index, divisas, commodities y bonos en horizontes de **1 a 12
  meses**, con reversión parcial en horizontes más largos. Un portafolio diversificado de estas
  estrategias entrega retorno anormal con poca exposición a factores estándar.
- **"Risk Adjusted Time Series Momentum" (RAMOM)** — Dudler, Gmuer, Malamud: normalizar el momentum por
  volatilidad pasada supera al momentum tradicional en casi todas las combinaciones de lookback/holding.
- **Crítica reciente**: un paper que examina 60 futuros líquidos encuentra evidencia *débil* tanto para
  momentum como para su opuesto (reversión) — es decir, el edge de trend-following no es universal ni
  obvio incluso en la literatura más reciente, hay desacuerdo real.
- **"The Science and Practice of Trend-following Systems"** — Artur Sepp: clasifica los sistemas de
  trend-following en tipos "Europeo", "Americano" y "Time Series Momentum"; da un marco práctico de
  diseño (no pude extraer los parámetros numéricos exactos del PDF bloqueado).

**Relevancia**: esta literatura valida la tendencia como fenómeno real, pero casi siempre en horizontes
de días-meses sobre futuros/CTAs, no en el timeframe H1/H4 con `DONCHIAN_N=1` que usa el bot. No es una
contradicción directa, pero sí significa que no podemos apoyarnos en "time series momentum está probado
académicamente" como justificación de la config actual sin más matices — el fenómeno probado y nuestra
implementación específica no son exactamente lo mismo.

---

## 5. Estacionalidad horaria: lo que dice la literatura vs. lo que dice nuestro propio backtest

- **Breedon & Ranaldo, "Intraday Patterns in FX Returns and Order Flow"**: evidencia de que las divisas
  tienden a depreciarse durante el horario de trading local de cada centro (Londres deprecia GBP durante
  horas de Londres, etc.), relacionado con flujo de órdenes de participantes locales.
- **Consenso práctico general** (no todos son papers académicos, algunos son fuentes de mercado): el
  overlap Londres-NY (~13:00-17:00 UTC) concentra >70% del volumen diario y es descrito habitualmente
  como "la mejor ventana".
- **Nuestro propio backtest de 16 años (sección 19 de CLAUDE.md) encuentra lo contrario para esas horas
  específicas**: 17:00-19:00 UTC están bloqueadas por WR=24-28%, de las peores del día; la hora 14 UTC
  también fue bloqueada explícitamente tras el backtest de 2026-07-26 (WR=29-36%, avg=-$35/trade, con
  más del doble de trades que cualquier otra hora — la peor combinación posible). Las horas que sí
  rinden en nuestros datos son 15, 16, 20, 21, 22, 23 UTC — una mezcla que no coincide limpiamente con
  "el overlap es lo mejor".

**Conclusión de esta sección, ya aplicando la regla de la sección 20 de CLAUDE.md**: esto es
exactamente un caso de "la literatura genérica dice una cosa, nuestro backtest verificado con datos
propios dice otra" — y la regla del proyecto es clara: se confía en el backtest propio, no en la
sabiduría convencional de blogs/guías de sesión. No hay ninguna acción que tomar aquí, es solo
documentar que la divergencia es consistente con lo que ya sabíamos, no una señal de alerta nueva.

---

## 6. Overfitting de backtest — relevante para la reoptimización reciente

- **"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"**
  — Bailey & López de Prado. Corrige el Sharpe Ratio por sesgo de selección bajo tests múltiples y
  no-normalidad de retornos.
- **"The Probability of Backtest Overfitting" (PBO)** — Bailey, Borwein, López de Prado, Zhu. Método de
  validación cruzada simétrica combinatoria (CSCV) para estimar la probabilidad de que un backtest esté
  sobreajustado.
- **"Evaluating Trading Strategies"** — Harvey & Liu: bajo tests múltiples, un t-stat de 2.0 (el umbral
  clásico de "significativo") ya no alcanza — proponen un umbral más estricto (~3.0) cuando se han
  probado muchas variantes de una estrategia, como ocurre naturalmente en cualquier proceso de
  reoptimización iterativa.

**Relevancia concreta**: `ATR_MULT_SL=0.3`/`RR_MULT=25.0`/`DONCHIAN_N=1` nacieron de corregir un bug de
paridad D1/H4 y volver a ajustar tras eso. Ese patrón — bug, fix, reoptimizar, el backtest luce bien de
nuevo — es exactamente el escenario que esta literatura advierte que puede producir una config que
parece ganadora en el histórico específico usado para reoptimizar, sin que eso implique una ventaja real
hacia adelante. No es una crítica al proceso (fue el correcto dado el bug real que había), es una razón
concreta para, una vez haya una muestra decente de trades reales o out-of-sample genuino, pedir
explícitamente un chequeo tipo PBO/deflated Sharpe antes de tratar esta config como "la definitiva" —
en vez de solo mirar si el P&L acumulado se ve bien.

---

## 7. Smart Money Concepts / Order Blocks — no hay validación académica real

Búsqueda específica confirma lo que ya sabíamos por la propia auditoría del proyecto: SMC **no es una
metodología formalmente reconocida ni divulgada por bancos, hedge funds o firmas prop** — no hay
estudios académicos públicos que cuantifiquen específicamente conceptos como "order block" u "FVG" tal
como los define la comunidad retail de SMC. Algunos de sus principios generales (liquidez, ejecución de
órdenes, desequilibrios oferta/demanda) sí tienen raíces en microestructura de mercado académica
legítima, pero el framework SMC en sí, tal como se popularizó en redes/cursos, no tiene el respaldo que
a veces se le atribuye.

Esto no cambia nada operativamente — coincide exactamente con la decisión ya tomada en la sección 16 de
CLAUDE.md (Lunar/Elliott/Chaos eliminados por falta de evidencia real, política de "nunca añadir un
agente sin evidencia de backtest real"). Es una confirmación externa de que esa política es la correcta,
no un hallazgo nuevo que requiera acción.

---

## 8. Realidad de las prop firms (contexto de expectativas, no un paper per sé)

- Pass rate público de FTMO (la firma más transparente del sector) para el challenge estándar de 2 fases:
  **9-10%** (incluyendo verificación completa, filtra a quienes pasan fase 1 solo por suerte).
- Rango público reportado por firmas en general: **8-15%** para evaluaciones de 2 fases — cifras
  autoreportadas, no auditadas independientemente, así que hay que tomarlas como orientativas.
- Dataset FPFX Technology (~300K cuentas, ~100K traders, 10 firmas, reportado por Finance Magnates,
  sept 2026): solo **~7% de todos los que compran un challenge** llegan a cobrar un payout alguna vez.
  De los que sí pasan y quedan fondeados, ~45% recibe al menos un payout.

No cambia la estrategia de nada — solo sirve como ancla realista: pasar el challenge es la parte
"fácil" relativa, mantener consistencia para cobrar payouts repetidos es donde la mayoría se cae. Esto
refuerza (externamente, no es nada nuevo) por qué la sección 11 de CLAUDE.md pone tanto peso en reglas
de consistencia y modo seguridad (stop diario al 60%, pausa tras 3 perdedores, etc.) — ese tipo de
disciplina es exactamente lo que separa al 7% del resto según este dato.

---

## 9. Position sizing bajo incertidumbre (relevante para muestras chicas tipo challenge)

- **Kelly criterion clásico**: tamaño óptimo de posición como fracción del capital según el edge
  estimado — teóricamente supera a cualquier otro sizing, pero requiere conocer la probabilidad de
  éxito real, algo que en la práctica nunca se sabe con certeza.
- **"Bayesian Kelly Criterion with Parameter Uncertainty"** — Sukhov: en vez de asumir que se conoce la
  probabilidad de éxito real, modela la incertidumbre de esa estimación explícitamente — relevante
  porque durante un challenge de prop firm el número de trades es chico, así que la probabilidad de
  éxito "medida" tiene mucho error de muestreo.
- **"Trade Sizing Techniques for Drawdown and Tail Risk Control"** — Strub: 3 algoritmos concretos de
  sizing para controlar drawdown máximo o tail risk (uno basado en volatilidad histórica, uno en
  Extreme Value Theory sobre VaR/CVaR, uno en EVT aplicado directamente a la distribución de drawdowns).
- **"Optimal Portfolio Strategy to Control Maximum Drawdown"** — Yang & Zhong: estrategia de trading
  discreta que controla directamente el drawdown máximo dentro de un nivel objetivo mientras maximiza
  la tasa de crecimiento de largo plazo del portafolio.

**Relevancia para la conversación pendiente sobre el guard de drawdown**: estos papers no dicen "el
guard actual está mal" — dicen que existe un marco matemático real detrás de la idea de "frenar antes
del límite duro" (que es exactamente lo que hace nuestro guard al 70% del límite). Si en algún momento
se decide ajustar el guard específicamente para validación en demo (la conversación que quedó pendiente
en `SESION_ACTUAL.md`), esta es la bibliografía de referencia para hacerlo de forma principiada en vez
de simplemente subir un número — pero eso sigue siendo una decisión del usuario, no algo que yo vaya a
tocar unilateralmente.

---

## 10. Lista de lecturas pendientes (si se quiere profundizar con cuenta SSRN)

Todos bloquearon el PDF completo vía fetch automatizado (403) — solo tengo el abstract indexado. Quedan
como pendientes si el usuario quiere leerlos completos (requiere cuenta gratuita SSRN):

- Poluri (2026) — Donchian+ATR — **el más prioritario, es casi nuestra misma arquitectura**.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6272239
- Rounce (2026) — Live reconciliation FX breakout — **el más accionable ahora mismo**.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5805443
- Costa (2026) — Illusion of Breakouts (majors vs Gold).
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6592020
- Howard (2026) — Stop Distance / Exit Methodology, E-mini S&P 500 — relevante para RR_MULT=25.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6350238
- Mittal & Choudhary (2026) — Liquidity-Driven Breakout Reliability (Kaplan-Meier).
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5962358
- Sepp — Science and Practice of Trend-following Systems.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167787
- Wang & Gangwar (2025) — Optimizing Intraday Breakout Strategies on NSE (block-based eval).
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5198458

---

## 11. Tareas concretas sugeridas para cuando el usuario esté presente (ninguna ejecutada por mí)

1. **Sin riesgo, se puede hacer aunque el guard siga bloqueado**: leer `trade_stops_level` y
   `trade_freeze_level` vía MT5 API para los 8 símbolos activos y compararlo contra la distancia de SL
   que produce `ATR(14) × 0.3` en cada uno ahora mismo — para saber de antemano si algún símbolo va a
   rechazar por 10016 antes de gastar el primer intento real descubriéndolo (sección 3).
2. Cuando haya masa de trades reales: desglosar win rate/profit factor por símbolo y comparar
   FX majors puros vs XAUUSD/índices, para verificar si el patrón de Costa (majors mean-revierten,
   Gold/índices continúan) se sostiene con nuestros propios datos (sección 2).
3. Cuando se acumulen más trades: pedir un chequeo tipo deflated-Sharpe/PBO sobre
   ATR_MULT_SL=0.3/RR_MULT=25.0 antes de tratarlos como definitivos, dado que nacieron de una
   reoptimización post-bug (sección 6).
4. Conversación pendiente (ya documentada en SESION_ACTUAL.md): si se ajusta el guard de drawdown para
   validación en demo, usar el marco de Strub/Yang&Zhong como referencia en vez de un ajuste ad-hoc
   (sección 9) — decisión del usuario, no mía.

Ninguna de estas 4 tareas fue ejecutada esta noche — son la lista para la próxima sesión con el usuario
presente, en el orden en que probablemente convenga abordarlas.
