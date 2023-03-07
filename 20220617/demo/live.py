from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig
)
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory
)
from nautilus_trader.config import InstrumentProviderConfig, TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from strategy import PairTrader, PairTraderConfig


# Configure the trading node
config_node = TradingNodeConfig(
    trader_id="TRADER-001",
    log_level="DEBUG",
    data_clients={
        "IB": InteractiveBrokersDataClientConfig(
            trading_mode="paper",
            start_gateway=False,
            read_only_api=False,
            gateway_host="127.0.0.1",
            gateway_port=4002,
            instrument_provider=InstrumentProviderConfig(
                load_all=True,
                filters=tuple(
                    {
                        "filters": (
                            tuple({"secType": "STK", "symbol": "SEC0", "exchange": "BVME.ETF"}.items()),
                            tuple({"secType": "STK", "symbol": "SMH", "exchange": "BVME.ETF"}.items()),
                        )
                    }.items()
                ),
            ),
        ),
    },
    exec_clients={
        "IB": InteractiveBrokersExecClientConfig(
            trading_mode="paper",
            start_gateway=False,
            read_only_api=False,
            gateway_host="127.0.0.1",
            gateway_port=4002,
        ),
    },
    timeout_connection=90.0,
    timeout_reconciliation=5.0,
    timeout_portfolio=5.0,
    timeout_disconnection=5.0,
    timeout_post_stop=2.0,
)

# Instantiate the node with a configuration
node = TradingNode(config=config_node)

# Configure your strategy
strategy_config = PairTraderConfig(
    source_symbol="SEME.IBIS2",
    target_symbol="SMH.BVME.ETF",
)
# Instantiate your strategy
strategy = PairTrader(config=strategy_config)

# Add your strategies and modules
node.trader.add_strategy(strategy)

# Register your client factories with the node (can take user defined factories)
node.add_data_client_factory("IB", InteractiveBrokersLiveDataClientFactory)
node.add_exec_client_factory("IB", InteractiveBrokersLiveExecClientFactory)
node.build()

# Stop and dispose of the node with SIGINT/CTRL+C
if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
