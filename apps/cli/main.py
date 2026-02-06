import argparse
import sys
from typing import Optional

from engine.core.config import Config
from engine.main import run_engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="echoflux",
        description="EchoFlux — Real-time speech-to-text and translation",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="ASR model size (e.g. tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Source language code (e.g. en, ja, de)",
    )
    parser.add_argument(
        "--translate",
        type=str,
        default=None,
        help="Target translation language code (e.g. vi, fr, es)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="WebSocket server port",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice activity detection",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace):
    if args.model:
        config.set("asr.model_size", args.model)
    if args.lang:
        config.set("asr.language", args.lang)
    if args.translate:
        config.set("translation.enabled", True)
        config.set("translation.target_lang", args.translate)
        if args.lang:
            config.set("translation.source_lang", args.lang)
    if args.device:
        config.set("asr.device", args.device)
    if args.port:
        config.set("engine.port", args.port)
    if args.no_vad:
        config.set("vad.enabled", False)
    if args.log_level:
        config.set("logging.level", args.log_level)


def main():
    args = parse_args()
    config = Config(args.config)
    apply_overrides(config, args)

    try:
        from engine.core.logging import setup_logging
        logger = setup_logging(config)
        logger.info("EchoFlux CLI starting")
        logger.info("Config: model=%s, lang=%s, device=%s",
                     config.get("asr.model_size"),
                     config.get("asr.language"),
                     config.get("asr.device"))

        run_engine(args.config)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
