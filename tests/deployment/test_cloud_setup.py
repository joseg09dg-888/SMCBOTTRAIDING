import pytest, os, time
from pathlib import Path
from deployment.cloud_setup import CloudSetup
from deployment.health_monitor import HealthMonitor, BotHealth

PROJECT = r"C:\Users\jose-\projects\trading_agent"

# ── Dockerfile ──────────────────────────────────────────────────────────────
def test_dockerfile_base_image():
    assert "FROM python:3.12-slim" in CloudSetup().generate_dockerfile()

def test_dockerfile_cmd():
    assert "startup.py" in CloudSetup().generate_dockerfile()

def test_dockerfile_copy():
    assert "COPY" in CloudSetup().generate_dockerfile()

def test_dockerfile_pip_install():
    assert "pip install" in CloudSetup().generate_dockerfile()

def test_dockerfile_workdir():
    assert "WORKDIR /app" in CloudSetup().generate_dockerfile()

# ── docker-compose ──────────────────────────────────────────────────────────
def test_compose_service_name():
    assert "smc-trading-bot" in CloudSetup().generate_docker_compose()

def test_compose_restart():
    assert "restart: always" in CloudSetup().generate_docker_compose()

def test_compose_volumes():
    c = CloudSetup().generate_docker_compose()
    assert "memory" in c and "logs" in c

def test_compose_env_file():
    assert ".env" in CloudSetup().generate_docker_compose()

# ── Save files ───────────────────────────────────────────────────────────────
def test_save_dockerfile(tmp_path):
    cs = CloudSetup(project_root=str(tmp_path))
    path = cs.save_dockerfile()
    assert os.path.exists(path)
    assert open(path).read().startswith("FROM")

def test_save_docker_compose(tmp_path):
    cs = CloudSetup(project_root=str(tmp_path))
    path = cs.save_docker_compose()
    assert os.path.exists(path)
    assert "smc-trading-bot" in open(path).read()

# ── validate_env_file ────────────────────────────────────────────────────────
def test_validate_env_real_project():
    cs = CloudSetup(project_root=PROJECT)
    result = cs.validate_env_file()
    assert isinstance(result, dict) and "valid" in result and "missing" in result

def test_validate_env_missing(tmp_path):
    cs = CloudSetup(project_root=str(tmp_path))
    result = cs.validate_env_file()
    assert result["valid"] is False

# ── validate_requirements ────────────────────────────────────────────────────
def test_validate_requirements_real():
    cs = CloudSetup(project_root=PROJECT)
    result = cs.validate_requirements()
    assert isinstance(result, dict) and "valid" in result

# ── PM2 config ───────────────────────────────────────────────────────────────
def test_pm2_config_startup():
    assert "startup.py" in CloudSetup().generate_pm2_config()

def test_pm2_config_restart():
    assert "restart_delay" in CloudSetup().generate_pm2_config()

# ── checklist ────────────────────────────────────────────────────────────────
def test_checklist_returns_list():
    cl = CloudSetup().get_deployment_checklist()
    assert isinstance(cl, list) and len(cl) > 0

def test_checklist_has_fields():
    for item in CloudSetup().get_deployment_checklist():
        assert "name" in item and "status" in item

# ── HealthMonitor ─────────────────────────────────────────────────────────────
def test_health_returns_bot_health():
    hm = HealthMonitor(project_root=PROJECT)
    health = hm.check_health()
    assert isinstance(health, BotHealth)

def test_health_is_running_bool():
    assert isinstance(HealthMonitor().check_health().is_running, bool)

def test_health_log_size_non_negative():
    assert HealthMonitor(project_root=PROJECT).get_log_size_kb() >= 0.0

def test_health_never_raises():
    hm = HealthMonitor(project_root=r"C:\nonexistent\path")
    try:
        health = hm.check_health()
        assert isinstance(health, BotHealth)
    except Exception:
        pytest.fail("check_health raised")

def test_format_telegram_string():
    hm = HealthMonitor()
    msg = hm.format_telegram(hm.check_health())
    assert isinstance(msg, str)

def test_is_bot_stuck_old_log(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "bot_live.log"
    log.write_text("last line")
    old_time = time.time() - 400
    os.utime(log, (old_time, old_time))
    hm = HealthMonitor(project_root=str(tmp_path))
    assert hm.is_bot_stuck(max_silence_sec=300) is True

def test_is_bot_stuck_fresh_log(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / "bot_live.log"
    log.write_text("recent")
    hm = HealthMonitor(project_root=str(tmp_path))
    assert hm.is_bot_stuck(max_silence_sec=300) is False
