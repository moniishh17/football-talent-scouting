"""
processing/injury_channel.py
==============================
Implements the RIGHT side of the diagram: Injury Prevention Channel

    IoT Wearables -> Real-Time Sync -> Analytics & Output

Real-world mapping:
    IoT Wearables    -> Physical sensors publishing over MQTT
    Real-Time Sync   -> Kafka/Kinesis stream + InfluxDB/TimescaleDB
    Analytics/Output -> Streaming anomaly detection -> alert dispatch
"""

import random
import time
import threading
from dataclasses import dataclass, field

from shared.cloud_infra import DistributedStorage, MessageQueue, Alert


# Safety thresholds used by the "Analytics & Output" stage
HEART_RATE_MAX = 190          # bpm - sustained above this triggers a fatigue alert
SPRINT_LOAD_MAX = 9.5         # arbitrary GPS-derived load unit
FATIGUE_STREAK_LIMIT = 3      # consecutive high readings before alert fires


@dataclass
class WearableReading:
    player_id: str
    heart_rate_bpm: int
    gps_load: float
    timestamp: float = field(default_factory=time.time)


def _simulate_wearable_sensor(player_id: str, mqtt: MessageQueue, n_readings: int, stop_event: threading.Event):
    """
    Simulates the 'IoT Wearables' stage: a single player's sensor
    publishing biometric readings onto the MQTT-style message queue.
    """
    for _ in range(n_readings):
        if stop_event.is_set():
            return
        reading = WearableReading(
            player_id=player_id,
            heart_rate_bpm=random.randint(120, 200),
            gps_load=round(random.uniform(3.0, 11.0), 2),
        )
        mqtt.publish({"type": "wearable_reading", "payload": reading})
        time.sleep(0.02)  # simulate sensor sampling interval


def start_wearable_simulation(player_ids: list[str], mqtt: MessageQueue, n_readings: int = 15) -> tuple[list[threading.Thread], threading.Event]:
    """Spins up one sensor-simulation thread per player (parallel IoT devices)."""
    stop_event = threading.Event()
    threads = []
    for pid in player_ids:
        t = threading.Thread(target=_simulate_wearable_sensor, args=(pid, mqtt, n_readings, stop_event))
        t.start()
        threads.append(t)
    return threads, stop_event


class RealTimeSyncProcessor:
    """
    Simulates the 'Real-Time Sync' stage: consumes the MQTT stream and
    writes each reading into a time-series store (DistributedStorage
    here stands in for InfluxDB/TimescaleDB).
    """

    def __init__(self, storage: DistributedStorage):
        self.storage = storage

    def handle_message(self, message: dict):
        if message.get("type") != "wearable_reading":
            return
        reading: WearableReading = message["payload"]
        self.storage.put(f"telemetry:{reading.player_id}", reading)


class AnalyticsAndOutput:
    """
    Simulates the 'Analytics & Output' stage: threshold analysis and
    fatigue-alert generation from streaming telemetry.
    """

    def __init__(self, storage: DistributedStorage):
        self.storage = storage
        self._streak: dict[str, int] = {}
        self.alerts: list[Alert] = []
        self._lock = threading.Lock()

    def handle_message(self, message: dict):
        if message.get("type") != "wearable_reading":
            return
        reading: WearableReading = message["payload"]

        is_high_risk = (
            reading.heart_rate_bpm >= HEART_RATE_MAX
            or reading.gps_load >= SPRINT_LOAD_MAX
        )

        with self._lock:
            streak = self._streak.get(reading.player_id, 0)
            streak = streak + 1 if is_high_risk else 0
            self._streak[reading.player_id] = streak

            if streak >= FATIGUE_STREAK_LIMIT:
                alert = Alert(
                    channel="injury_prevention",
                    severity="warning",
                    message=(
                        f"Player {reading.player_id}: sustained high load "
                        f"(HR={reading.heart_rate_bpm}bpm, GPS load={reading.gps_load}) "
                        f"-- possible fatigue/injury risk."
                    ),
                )
                self.alerts.append(alert)
                self.storage.put(f"alert:{reading.player_id}", alert)
                self._streak[reading.player_id] = 0  # reset after firing


def run_injury_pipeline(player_ids: list[str], storage: DistributedStorage, n_readings: int = 15) -> dict:
    """End-to-end run of the Injury Prevention Channel."""
    mqtt = MessageQueue()
    sync = RealTimeSyncProcessor(storage)
    analytics = AnalyticsAndOutput(storage)

    mqtt.subscribe(sync.handle_message)
    mqtt.subscribe(analytics.handle_message)

    threads, stop_event = start_wearable_simulation(player_ids, mqtt, n_readings)
    for t in threads:
        t.join()

    summary = {
        "players_monitored": len(player_ids),
        "total_readings_ingested": sum(
            len(storage.get_all(f"telemetry:{pid}")) for pid in player_ids
        ),
        "alerts_fired": len(analytics.alerts),
        "alert_details": [
            {"player": a.message.split(":")[0].split()[-1], "message": a.message, "severity": a.severity}
            for a in analytics.alerts
        ],
    }
    storage.put("injury:summary", summary)
    return summary
