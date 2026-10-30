"""fleet_telemetry — each box's hardware, sampled and broadcast on the LAN for the gateway.

Every few seconds a box's sender reads its CPU, memory, GPUs, temperatures, power, clocks, fans, network counters and
containers, and broadcasts them as one JSON datagram (UDP). The gateway (vllm_lb/hw.py) draws the hardware tiles from
it and keeps one hour of it as the box's FLIGHT RECORDER (vllm_lb/recorder.py). No credential, no database, no login
on the box: a box that stops sending simply goes quiet.

The package, by layer (DDD, stdlib only):

  figures.py   the figure table and the wire's field names — ONE definition, shared with the gateway's recorder
  model.py     the domain: GpuReading, ThermalZone, MemoryReading, NetCounters, ContainerUsage, ContainerSet, Sample
  readers/     one reader per SOURCE behind one interface (`source`, `read()`): nvidia-smi, the amdgpu driver's sysfs,
               /proc/stat, /proc/meminfo, CPU sensors (hwmon, cpufreq), /sys/class/thermal, hwmon fans,
               /proc/net/dev, the cgroup v2 tree. A GPU kind is a reader class (readers.GPU_READERS), never a branch.
  sampler.py   composes the readers into a Sample; a failing reader is its figures' reason, never the sender's death
  broadcast.py the LAN broadcaster (the default route's broadcast address and 255.255.255.255)
  __main__.py  `python -m fleet_telemetry`: settings from the environment, the preflight, the loop

ORIGIN. The sampler, the LAN broadcast and the deep hardware readers began as broker_parrot's hardware telemetry
(its hw_metrics, hw_lan, hw_watch and cgroup attribution, Apache-2.0, by the same author). They were copied into this
repository and rewritten here; this package is this repository's own, and nothing of that project is imported or
tracked (NOTICE).

NO FALLBACKS. A figure a box does not expose is None with its reason, never 0 and never another source standing in for
it; a configuration the sender cannot honour stops it before the first sample, naming what is missing.
"""
