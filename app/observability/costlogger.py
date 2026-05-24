from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

_PRICE_TABLE: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {
        "input":  0.59 / 1_000_000,
        "output": 0.79 / 1_000_000,
    },
    "llama-3.1-8b-instant": {
        "input":  0.05 / 1_000_000,
        "output": 0.08 / 1_000_000,
    }
}

_FALLBACK_RATE: dict[str, float] = {
    "input":  1.00 / 1_000_000,
    "output": 1.00 / 1_000_000,
}


def log_generation(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    rates = _PRICE_TABLE.get(model)
    if rates is None:
        logger.warning(
            "cost_logger: unknown model '%s' – using fallback rate $1.00/1M tokens.",
            model,
        )
        rates = _FALLBACK_RATE

    cost = (input_tokens * rates["input"]) + (output_tokens * rates["output"])

    logger.debug(
        "cost_logger: model=%s  in=%d  out=%d  cost=$%.6f",
        model,
        input_tokens,
        output_tokens,
        cost,
    )
    return cost