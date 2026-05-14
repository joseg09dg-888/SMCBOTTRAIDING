# core/agent_health_check.py
"""
AgentHealthCheck — verifica que los 21 agentes del sistema puedan importarse y
estén disponibles. Diseñado para TDD: primero los tests, luego esta implementación.
"""
import importlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AgentStatus:
    name: str
    module: str          # e.g. "agents.signal_agent"
    class_name: str      # e.g. "SignalAgent"
    is_initialized: bool
    has_valid_output: bool
    signals_generated: int
    accuracy: Optional[float]   # 0.0–1.0, None if not applicable
    last_check: datetime
    error: Optional[str] = None


@dataclass
class HealthReport:
    timestamp: datetime
    total_agents: int
    healthy_agents: int
    failed_agents: int
    statuses: list

    @property
    def all_healthy(self) -> bool:
        return self.healthy_agents == self.total_agents

    def format_telegram(self) -> str:
        """
        Returns message formatted for Telegram:
        🏥 HEALTH CHECK — 21 AGENTES
        ✅ Supervisor Agent — OK
        ...
        ─────────────────
        21/21 agentes operativos
        """
        lines = [f"🏥 HEALTH CHECK — {self.total_agents} AGENTES"]
        for status in self.statuses:
            if status.is_initialized:
                lines.append(f"✅ {status.name} — OK")
            else:
                err = status.error or "Error desconocido"
                lines.append(f"❌ {status.name} — ERROR: {err}")
        lines.append("─────────────────")
        lines.append(f"{self.healthy_agents}/{self.total_agents} agentes operativos")
        return "\n".join(lines)


# Registry of all 21 agents with their importable module and class names.
# session_manager has no class — we use the sentinel "session_score" (a function)
# so check_agent handles it gracefully by checking for the attribute directly.
AGENT_REGISTRY = [
    {"name": "Supervisor Agent",       "module": "core.supervisor",                "class": "TradingSupervisor"},
    {"name": "Training Agent",         "module": "training.historical_agent",      "class": "HistoricalDataAgent"},
    {"name": "Analysis Agent SMC",     "module": "agents.analysis_agent",          "class": "AnalysisAgent"},
    {"name": "Signal Agent",           "module": "agents.signal_agent",            "class": "SignalAgent"},
    {"name": "Risk Manager",           "module": "core.risk_manager",              "class": "RiskManager"},
    {"name": "Decision Filter",        "module": "core.decision_filter",           "class": "DecisionFilter"},
    {"name": "Telegram Agent",         "module": "dashboard.telegram_bot",         "class": "TradingTelegramBot"},
    {"name": "Glint Agent",            "module": "connectors.glint_connector",     "class": "GlintConnector"},
    {"name": "Prediction Agent",       "module": "smc.ml_predictor",               "class": "MLPredictor"},
    {"name": "Lunar Agent",            "module": "agents.lunar_agent",             "class": "LunarCycleAgent"},
    {"name": "Elliott Agent",          "module": "agents.elliott_agent",           "class": "ElliottAgent"},
    {"name": "Institutional Flow",     "module": "agents.institutional_flow_agent","class": "InstitutionalFlowAgent"},
    {"name": "Alternative Data",       "module": "agents.alternative_data_agent",  "class": "AlternativeDataAgent"},
    {"name": "Microstructure",         "module": "agents.microstructure_agent",    "class": "MicrostructureAgent"},
    {"name": "FED Sentiment",          "module": "agents.fed_sentiment_agent",     "class": "FedSentimentAgent"},
    {"name": "OnChain Agent",          "module": "agents.onchain_agent",           "class": "OnChainAgent"},
    {"name": "Geopolitical",           "module": "agents.geopolitical_agent",      "class": "GeopoliticalAgent"},
    {"name": "Chaos Theory",           "module": "agents.chaos_agent",             "class": "ChaosAgent"},
    {"name": "Retail Psychology",      "module": "agents.retail_psychology_agent", "class": "RetailPsychologyAgent"},
    {"name": "Mode Manager",           "module": "core.mode_manager",              "class": "ModeManager"},
    {"name": "Session Manager",        "module": "core.session_manager",           "class": "session_score"},
]


class AgentHealthCheck:
    """
    Checks all 21 agents by attempting to import their module and locate
    their class (or function) by name.  No instances are created — this
    is a pure static import check so it is fast and side-effect-free.
    """

    def __init__(self):
        self._registry = AGENT_REGISTRY

    def check_agent(self, agent_def: dict) -> AgentStatus:
        """
        Attempts to import the module and find the class/function by name.

        - If import succeeds AND attribute exists → is_initialized=True
        - If import fails OR attribute missing → is_initialized=False, error set
        has_valid_output = True when is_initialized is True (no runtime metrics).
        signals_generated = 0 (basic check, no live metrics).
        accuracy = None (no history available at check time).
        """
        name = agent_def["name"]
        module_path = agent_def["module"]
        class_name = agent_def["class"]
        now = datetime.now(timezone.utc)

        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            return AgentStatus(
                name=name,
                module=module_path,
                class_name=class_name,
                is_initialized=False,
                has_valid_output=False,
                signals_generated=0,
                accuracy=None,
                last_check=now,
                error=str(exc),
            )

        if not hasattr(mod, class_name):
            return AgentStatus(
                name=name,
                module=module_path,
                class_name=class_name,
                is_initialized=False,
                has_valid_output=False,
                signals_generated=0,
                accuracy=None,
                last_check=now,
                error=f"AttributeError: '{module_path}' has no attribute '{class_name}'",
            )

        return AgentStatus(
            name=name,
            module=module_path,
            class_name=class_name,
            is_initialized=True,
            has_valid_output=True,
            signals_generated=0,
            accuracy=None,
            last_check=now,
            error=None,
        )

    def run_full_check(self) -> HealthReport:
        """Runs check_agent() for every entry in the registry."""
        now = datetime.now(timezone.utc)
        statuses = [self.check_agent(agent) for agent in self._registry]
        healthy = sum(1 for s in statuses if s.is_initialized)
        failed = len(statuses) - healthy
        return HealthReport(
            timestamp=now,
            total_agents=len(statuses),
            healthy_agents=healthy,
            failed_agents=failed,
            statuses=statuses,
        )

    def format_short_status(self, status: AgentStatus) -> str:
        """
        Returns a one-line summary:
          ✅ Signal Agent — OK
          ❌ Signal Agent — ERROR: ImportError: no module named ...
        """
        if status.is_initialized:
            return f"✅ {status.name} — OK"
        err = status.error or "Error desconocido"
        return f"❌ {status.name} — ERROR: {err}"
