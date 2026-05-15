"""Historical stress testing against real crash scenarios."""
from dataclasses import dataclass, field

@dataclass
class StressScenario:
    name: str; date: str; market_move_pct: float; duration_days: int
    recovery_days: int; asset_class: str; volatility_spike: float

@dataclass
class StressResult:
    scenario: StressScenario; portfolio_loss_usd: float; portfolio_loss_pct: float
    circuit_breaker_triggered: bool; survived: bool; recovery_estimate_days: int
    max_drawdown_pct: float; risk_level: str

@dataclass
class StressReport:
    scenarios_run: int; passed: int; failed: int; total_loss_worst_case: float
    survival_rate: float; circuit_breaker_triggers: int; recommendations: list
    ready_for_real_money: bool

class StressTester:
    SCENARIOS = [
        StressScenario("Black Monday 1987","1987-10-19",-22.6,1,252,"equity",8.0),
        StressScenario("Asia Crisis 1997","1997-10-27",-7.2,30,180,"forex",4.0),
        StressScenario("Dotcom Crash 2000","2000-03-10",-78.0,730,1825,"equity",3.0),
        StressScenario("Crisis 2008","2008-09-15",-55.0,365,1460,"all",6.0),
        StressScenario("Flash Crash 2010","2010-05-06",-9.2,1,1,"equity",10.0),
        StressScenario("COVID Crash 2020","2020-03-12",-34.0,30,90,"all",7.0),
        StressScenario("Luna Crash 2022","2022-05-09",-99.0,7,9999,"crypto",15.0),
        StressScenario("FTX Collapse 2022","2022-11-08",-25.0,14,730,"crypto",5.0),
        StressScenario("Hypothetical Flash","2026-01-01",-15.0,1,5,"all",12.0),
        StressScenario("Hypothetical Bear","2026-06-01",-40.0,90,365,"all",4.0),
    ]
    def run_scenario(self, scenario, equity, open_positions=None):
        if not open_positions:
            loss = equity * abs(scenario.market_move_pct/100) * 0.1
        else:
            loss = sum(p.get("size",0)*abs(scenario.market_move_pct/100) for p in open_positions)
        loss = min(loss, equity)
        loss_pct = loss/equity if equity>0 else 0.0
        cb = loss_pct > 0.05
        survived = loss_pct < 0.80
        if loss_pct < 0.02: rl = "SAFE"
        elif loss_pct < 0.05: rl = "CAUTION"
        elif loss_pct < 0.20: rl = "DANGER"
        else: rl = "RUIN"
        rec = int(scenario.recovery_days * (1 + loss_pct/10))
        return StressResult(scenario,loss,loss_pct,cb,survived,rec,loss_pct,rl)
    def run_all_scenarios(self, equity, open_positions=None, filter_asset_class=None):
        scenarios = self.SCENARIOS
        if filter_asset_class:
            scenarios = [s for s in scenarios if s.asset_class in (filter_asset_class,"all")]
        results = [self.run_scenario(s,equity,open_positions) for s in scenarios]
        passed = sum(1 for r in results if r.survived)
        failed = len(results)-passed
        cbs = sum(1 for r in results if r.circuit_breaker_triggered)
        sr = passed/len(results) if results else 1.0
        worst = max((r.portfolio_loss_usd for r in results),default=0.0)
        recs = []
        if sr < 0.8: recs.append("Reducir tamano de posicion")
        if cbs > 3: recs.append("Revisar gestion de riesgo")
        if passed == len(results): recs.append("Sistema robusto — listo para escalar")
        if worst > equity*0.5: recs.append("No usar apalancamiento")
        if not recs: recs.append("Sistema dentro de parametros aceptables")
        return StressReport(len(results),passed,failed,worst,sr,cbs,recs,sr>=0.8)
    def get_scenario_by_name(self, name):
        nl = name.lower()
        for s in self.SCENARIOS:
            if nl in s.name.lower(): return s
        return None
    @staticmethod
    def estimate_max_loss_pct(market_move_pct, leverage=1.0, position_pct=0.1):
        return float(min(abs(market_move_pct)*leverage*position_pct/100, 1.0))
