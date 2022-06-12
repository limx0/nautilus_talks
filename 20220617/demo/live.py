import msgspec.json
from dotenv import load_dotenv
from nautilus_trader.live.__main__ import main as live_main, validate


def main():
    raw = msgspec.json.encode(
        {
            "trader_id": "TESTER-001",
            "log_level": "INFO",
            "data_clients": {
                "IB": {
                    "factory_path": "nautilus_trader.adapters.interactive_brokers.factories:InteractiveBrokersLiveDataClientFactory",
                    "config_path": "nautilus_trader.adapters.interactive_brokers.config:InteractiveBrokersDataClientConfig",
                    "config": {
                        "gateway_host": "127.0.0.1",
                        "instrument_provider": dict(
                            load_all=True,
                            filters=tuple({"secType": "CASH", "pair": "EURUSD"}.items()),
                        ),
                        "routing": dict(venues={"IDEALPRO"}),
                    },
                }
            },
            # exec_clients={
            #     "IB": InteractiveBrokersExecClientConfig(),
            # },
            "timeout_connection": 90.0,
            "timeout_reconciliation": 5.0,
            "timeout_portfolio": 5.0,
            "timeout_disconnection": 5.0,
            "timeout_post_stop": 2.0,
        }
    )
    assert validate(raw)
    live_main(raw=raw)


if __name__ == "__main__":
    load_dotenv()
    main()
