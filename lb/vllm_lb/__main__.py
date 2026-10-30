"""`python -m vllm_lb` — LISTEN FIRST, then find the broker engine, then loop forever.

Boot order is the "always on" guarantee: nothing above `proxy.start()` may fail. The
engine (`queue_workflows`, mounted from NFS) is imported and configured by the control loop
— retried every tick until it works — so a missing mount at boot is a `/lb/status` line
(`db_ok: false, engine unavailable: …`) while requests keep flowing to what is live.

ONLY WITH THE BROKER ON (`LB_BROKER_ENABLED`, config.py `broker_enabled`). Off — the default — nothing here
imports the engine or builds the broker adapter: there is no `ensure_engine`, no `BrokerLeases`,
and `/lb/status` says `"broker": {"enabled": false}`.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading

from vllm_lb import topology
from vllm_lb.catalog import ModelCatalog
from vllm_lb.config import Config
from vllm_lb.control import Controller
from vllm_lb.hub import Hub
from vllm_lb.hw import HwCollector
from vllm_lb.perflog import PerfRecorder
from vllm_lb.probe import probe_backend
from vllm_lb.proxy import Proxy

log = logging.getLogger("vllm_lb")


def engine_setup(cfg: Config) -> "callable":
    """A retryable `() -> ''|error` that imports + configures the broker engine."""
    backend = os.environ.get("LB_DB_BACKEND", "pg")

    def ensure() -> str:
        try:
            import queue_workflows as qw
        except Exception as exc:                 # ModuleNotFoundError when the mount is absent
            return f"{type(exc).__name__}: {exc}"[:200]
        try:
            qw.configure(db_backend=backend, db_url_env=cfg.db_url_env)
            if backend == "sqlite":              # dev/smoke: a private engine DB needs its schema
                from queue_workflows import db
                db.bootstrap()
        except Exception as exc:
            return f"configure: {type(exc).__name__}: {exc}"[:200]
        return ""

    return ensure


#: The tp-rank settings that are the BOX's, not a model's, in the LB's own environment (lb/.env): where
#: this box keeps vLLM's caches, whether its GPU is sampled, the pair's RoCE ports. Set once here,
#: never per GB10 box, so both ranks launch alike. Forwarded as given when set — empty too, since empty is how
#: the cache is switched off — and left out when unset, so tp-rank's defaults stay the only defaults. The prefill
#: budget is NOT one of them (2026-10-29): it is the fleet file's `clusters.<name>.max_num_batched_tokens`, carried
#: on every call by `cluster_run.tp_rank_env`; CLUSTER_MAX_NUM_BATCHED_TOKENS in lb/.env is refused at boot.
#:
#: Everything a MODEL decides — image, checkpoint, served name, parsers, speculative config, KV pin, PLE
#: mode, drafter prefix — comes from the ModelChoice's LaunchProfile through `cluster_exec.launch_env`.
#: ONE source: the LB's environment must never be a second one, or a stray variable could launch a model
#: the operator never chose.
#:
#: THE OPERATOR IS NOT A STRAY VARIABLE. Two engine knobs are set from the gateway's cluster panel and
#: stored with the switch (`cluster.Desired`, through POST /lb/cluster): the prefill budget, which then
#: replaces the fleet's CLUSTER_MAX_NUM_BATCHED_TOKENS for that launch, and the drafter's depth
#: (`num_speculative_tokens`, 0 = off), which then replaces the count in the ModelChoice's own speculative
#: config. That is the operator CHOOSING, on the record, what the catalog only defaults — so it is a
#: legitimate override where an environment variable is not. It stays the ONLY one: no variable may be
#: added here, or anywhere else, that changes a model's speculation (tests/test_cluster_wiring.py). The
#: precedence, resolved in ONE place (`cluster_exec.resolved_launch`) and applied at launch by
#: `SshExecutor.up`: the operator's stored value > the catalog's LaunchProfile (its tuned budget, its speculation) >
#: the fleet's budget / tp-rank's own depth. Left at null, a knob adds nothing to the launch.
# The cluster loops and the run — which declared clusters run now — live in vllm_lb/cluster_run.py; re-exported here
# for the tests that have always built them from this module.
from vllm_lb.cluster_run import (ClusterLoops, ClusterRun, TP_RANK_SETTINGS, build_cluster_loop,  # noqa: E402,F401
                                 build_cluster_loops)


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LB_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    try:
        cfg = Config.from_env()
    except ValueError as exc:
        # A configuration the LB cannot mean — a retired setting, a fleet file with a missing or unknown value — is
        # ONE line naming it and EX_CONFIG (78): `docker ps` shows "Restarting (78)", `docker logs` says why. Never a
        # traceback to read through, never a start on half a configuration.
        log.error("[config] refused: %s", exc)
        return 78
    # ONE ModelCatalog for this process: the switch validates a ModelChoice against it, the cluster loop
    # launches from it, and the page's pickers are drawn from it. It never raises — a catalog that cannot
    # be read is a line in /lb/status, not a boot that fails.
    catalog = ModelCatalog.load()
    # WHAT THE CATALOG RESOLVED TO on this fleet's cards (resolve.py), one line a profile, at boot: the target, the
    # image and where the matrix says it comes from, the measurement it is sized by — or why it cannot start here.
    for line in catalog.resolution_lines():
        log.info("[catalog] %s", line)
    if catalog.error:
        log.error("[catalog] %s", catalog.error)
    if cfg.broker_enabled and not catalog.error:
        # THE LEASE MODEL'S WINDOW is its catalog row's (`Config.lease_context_length`): a lease model no row serves
        # is a configuration the LB cannot mean — refused like any other (78), naming the model and the field. A
        # catalog that did not load is the status line above, and the lease loop keeps the reason in `last_error`.
        try:
            cfg.lease_context_length
        except ValueError as exc:
            log.error("[config] refused: %s", exc)
            return 78
    # The cluster switch is a file the proxy writes and the cluster loop reads. None = no cluster.
    # Built by the one factory that also wires the measured default and the clamp, so a fresh
    # container offers what the pool actually holds (worklog/vllm-cluster-defaults.md C2) — and given
    # this process's ONE ModelCatalog, so the switch validates a ModelChoice against the same
    # instance the page's pickers are drawn from.
    # A node this LB only observes has no switch: its server file is (vllm_lb/topology.py). One switch per ssh
    # cluster, by name; `switch` is THE one cluster's — None when the fleet names several (the proxy's singular
    # POST /lb/cluster then refuses, naming each cluster's route: worklog/parallel-clusters.md).
    # THE RUN (vllm_lb/cluster_run.py): the boot clusters' views and backends are the Hub's from its first moment; the
    # run adds their switches and, once the proxy listens, their loops — and takes the panel's activations later.
    hub = Hub(cfg)
    run = ClusterRun(cfg, hub, catalog)
    switches = run.boot_switches()
    switch = switches.get(cfg.cluster_name)
    if switch is not None:
        # The Hub's view of the cluster before the loop's first tick recommends for the model the switch
        # HOLDS (its own pool and cost per token), not for one configured constant. Reading the switch never
        # raises: a missing or unreadable file is the default.
        hub.set_cluster(Hub._initial_view(cfg, catalog.get(switch.load().model_choice)), cfg.cluster_name)
    # Hardware is best-effort and starts BEFORE the proxy only so the first /lb/status can
    # already carry a sample; `start()` never raises — an unavailable port costs tiles.
    hw = HwCollector(cfg)
    hw.start()
    # The real-traffic performance log: never raises, None when switched off (LB_PERF=0).
    recorder = PerfRecorder.from_config(cfg)
    proxy = Proxy(cfg, hub, hw=hw, cluster_switch=switch, perf_recorder=recorder, run=run)
    proxy.start()                    # ← the only thing that must succeed at boot
    if cfg.broker_enabled:
        from vllm_lb.leases import BrokerLeases   # the broker's adapter: only on the path that uses it
        leases, ensure_engine = BrokerLeases(cfg), engine_setup(cfg)
    else:
        leases, ensure_engine = None, None
    ctl = Controller(cfg, hub, leases, prober=probe_backend, ensure_engine=ensure_engine,
                     cluster_loop=run.start(leases), recorder=recorder)

    stop = threading.Event()

    def on_signal(*_):
        # THE FIRST SIGNAL DRAINS (`docker compose up -d` recreating this container, `docker stop`): the
        # control loop ends its tick, then `proxy.drain` lets the turns in flight finish. A SECOND ends the
        # drain at once — the operator's way out of a long one.
        if stop.is_set():
            raise SystemExit(0)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    log.info("[main] vllm-lb up on %s:%s — boxes=%s clusters=%s broker_parrot=%s",
             cfg.listen_host, proxy.port, [b.host_label for b in cfg.vllm_boxes], [c.name for c in cfg.clusters],
             "on" if cfg.broker_enabled else "off (LB_BROKER_ENABLED unset)")
    try:
        ctl.run_forever(stop)
        proxy.drain(cfg.drain_s)
    except SystemExit:
        log.warning("[main] a second signal: the drain is cut short")
    finally:
        if recorder is not None:
            recorder.close()         # the minute in progress, if it had any activity
        proxy.stop()
        hw.stop()
    log.info("[main] stopped%s", "; leases are kept for the next start to adopt" if cfg.broker_enabled else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
