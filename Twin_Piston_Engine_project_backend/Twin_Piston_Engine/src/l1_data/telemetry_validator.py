"""
TelemetryValidator — Complete L1 Telemetry Security & Validation Service.

Pipeline order (Requirement 5):
    incoming RawSignalRecord
    → 1. Structural validation (types, non-null, valid acquisition units)
    → 2. Integrity verification (HMAC signature check via PacketVerifier)
    → 3. Replay & sequence validation (Monotonicity & anti-replay via ReplayProtector)
    → 4. Raw range validation (Configured limits check)
    → 5. Stale / missing checks (Timestamp & update rate check via StaleSignalDetector)
    → 6. Quality & Provenance assembly (Updated SignalQuality on RawSignalRecord)

FAILURE ISOLATION (Requirement 7):
    - Security / Replay / Structural failure → Packet REJECTED (dropped)
    - Range / Stale channel failure → Packet ACCEPTED, affected channels flagged invalid
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config import AppSettings, get_settings
from src.core.logging import get_logger
from src.core.provenance import ChannelValidity
from src.core.schemas import SignalQuality
from src.l1_data.raw_signal_record import RawSignalRecord
from src.l1_data.telemetry_security import (
    PacketSigner,
    PacketVerifier,
    ReplayProtector,
    StaleSignalDetector,
)

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Outcome of TelemetryValidator pipeline evaluation."""

    accepted: bool
    record: RawSignalRecord | None = None
    rejection_reason: str = ""
    security_verified: bool = False
    invalid_channels: list[str] = field(default_factory=list)
    invalid_reasons: dict[str, str] = field(default_factory=dict)


class TelemetryValidator:
    """L1 Telemetry Security and Validation Pipeline Service."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        signer: PacketSigner | None = None,
        verifier: PacketVerifier | None = None,
        replay_protector: ReplayProtector | None = None,
        stale_detector: StaleSignalDetector | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._signer = signer or PacketSigner(self._settings)
        self._verifier = verifier or PacketVerifier(self._settings)
        self._replay_protector = replay_protector or ReplayProtector(self._settings)
        self._stale_detector = stale_detector or StaleSignalDetector(self._settings)

    def validate_packet(
        self,
        record: RawSignalRecord,
        signature: str | None = None,
        secret_key: str | bytes | None = None,
        source_id: str = "default",
    ) -> ValidationResult:
        """Process incoming RawSignalRecord through the 6-stage validation pipeline."""

        # ---------------------------------------------------------------------
        # Stage 1: Structural Validation
        # ---------------------------------------------------------------------
        if ((record.crank_period_us is not None and record.crank_period_us <= 0)
                or (record.fuel_pulse_hz is not None and record.fuel_pulse_hz < 0)
                or (record.oil_rtd_ohms is not None and record.oil_rtd_ohms <= 0)):
            return ValidationResult(
                accepted=False,
                record=None,
                rejection_reason="Structural validation failed: invalid physical acquisition unit bounds",
            )

        # ---------------------------------------------------------------------
        # Stage 2: Integrity Verification (HMAC Signature Check)
        # ---------------------------------------------------------------------
        security_verified = False
        sig_to_verify = signature or record.integrity_hash

        if self._settings.security.require_signature:
            if not sig_to_verify:
                return ValidationResult(
                    accepted=False,
                    record=None,
                    rejection_reason="Security failure: signature missing but signature requirement enabled",
                )

            valid_sig = self._verifier.verify_signature(record, sig_to_verify, secret_key)
            if not valid_sig:
                return ValidationResult(
                    accepted=False,
                    record=None,
                    rejection_reason="Security failure: HMAC signature verification failed (altered payload or key mismatch)",
                )
            security_verified = True

        # ---------------------------------------------------------------------
        # Stage 3: Replay & Sequence Validation
        # ---------------------------------------------------------------------
        seq_valid, seq_reason = self._replay_protector.validate_sequence(
            sequence_number=record.sequence_number,
            source_id=source_id,
        )
        if not seq_valid:
            return ValidationResult(
                accepted=False,
                record=None,
                rejection_reason=f"Replay protection failure: {seq_reason}",
            )

        # ---------------------------------------------------------------------
        # Stage 4: Raw Range Validation
        # ---------------------------------------------------------------------
        invalid_channels: list[str] = []
        invalid_reasons: dict[str, str] = {}

        raw_ranges = self._settings.raw_channel_ranges
        raw_fields: dict[str, float] = {
            "egt_cyl1_hot_uv": record.egt_cyl1_hot_uv,
            "egt_cyl2_hot_uv": record.egt_cyl2_hot_uv,
            "egt_cyl3_hot_uv": record.egt_cyl3_hot_uv,
            "egt_cyl4_hot_uv": record.egt_cyl4_hot_uv,
            "egt_cold_c": record.egt_cold_c,
            "cht_hot_uv": record.cht_hot_uv,
            "cht_cold_c": record.cht_cold_c,
            "oil_rtd_ohms": record.oil_rtd_ohms,
            "oil_p_counts": record.oil_p_counts,
            "map_counts": record.map_counts,
            "adc_vref_counts": record.adc_vref_counts,
            "crank_period_us": record.crank_period_us,
            "fuel_pulse_hz": record.fuel_pulse_hz,
            "ambient_temp_c": record.ambient_temp_c,
            "ambient_press_pa": record.ambient_press_pa,
        }

        raw_fields = {k: float(v) for k, v in raw_fields.items() if v is not None}  # None = no data
        # Optional scalar raw fields (Prompt 11): range-checked only when present.
        for opt_name in ("bus_v_counts", "alt_i_counts", "batt_i_counts", "fuel_press_counts", "coolant_ntc_ohms"):
            opt_val = getattr(record, opt_name)
            if opt_val is not None:
                raw_fields[opt_name] = float(opt_val)

        for ch_name, val in raw_fields.items():
            if ch_name in raw_ranges:
                r_cfg = raw_ranges[ch_name]
                if not (r_cfg.min <= val <= r_cfg.max):
                    invalid_channels.append(ch_name)
                    invalid_reasons[ch_name] = f"Range violation: {val} not in [{r_cfg.min}, {r_cfg.max}]"

        # ---------------------------------------------------------------------
        # Stage 5: Stale & Missing Signal Checks
        # ---------------------------------------------------------------------
        stale_map = self._stale_detector.check_stale(record)
        for ch_name, reason in stale_map.items():
            if ch_name not in invalid_channels:
                invalid_channels.append(ch_name)
                invalid_reasons[ch_name] = reason

        # ---------------------------------------------------------------------
        # Stage 6: Quality & Provenance Metadata Result
        # ---------------------------------------------------------------------
        # Merge with what the SOURCE reported (adapter / simulator): a channel it
        # flagged invalid stays invalid even if it passes the range checks here
        # (e.g. a MAP dropout reading 0 counts). Source reasons take precedence.
        source_sq = record.signal_quality
        for ch_name in source_sq.invalid_channels:
            if ch_name not in invalid_channels:
                invalid_channels.append(ch_name)
        merged_reasons = dict(invalid_reasons)
        merged_reasons.update(source_sq.invalid_reasons)
        for ch_name in invalid_channels:
            merged_reasons.setdefault(ch_name, invalid_reasons.get(ch_name, "Flagged invalid by source"))
        invalid_reasons = {k: merged_reasons[k] for k in invalid_channels}

        num_channels = len(raw_fields)
        quality_score = 1.0 if not invalid_channels else max(0.0, 1.0 - (len(invalid_channels) / num_channels))
        quality_score = min(quality_score, source_sq.score)

        new_quality = SignalQuality(
            score=quality_score,
            valid=source_sq.valid and len(invalid_channels) == 0,
            invalid_channels=invalid_channels,
            invalid_reasons=invalid_reasons,
            noise_level=source_sq.noise_level,
        )

        validated_record = record.model_copy(update={"signal_quality": new_quality})

        return ValidationResult(
            accepted=True,
            record=validated_record,
            security_verified=security_verified,
            invalid_channels=invalid_channels,
            invalid_reasons=invalid_reasons,
        )

    def reset(self) -> None:
        """Reset internal replay protector and stale detector states."""
        self._replay_protector.reset()
        self._stale_detector.reset()
