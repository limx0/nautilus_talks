import pathlib
from typing import Tuple

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.config import BacktestDataConfig
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import BacktestRunConfig
from nautilus_trader.config import BacktestVenueConfig
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import ImportableActorConfig
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.config import RiskEngineConfig
from nautilus_trader.config import StreamingConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog


CATALOG = ParquetDataCatalog(str(pathlib.Path(__file__).parent.joinpath("catalog")))


def main(
    instrument_ids: Tuple[str, str],
    catalog: ParquetDataCatalog,
    notional_trade_size_usd: int = 10_000,
    start_time: str = None,
    end_time: str = None,
    log_level: str = "ERROR",
    bypass_logging: bool = False,
    persistence: bool = False,
    **strategy_kwargs,
):
    # Create model prediction actor
    prediction = ImportableActorConfig(
        actor_path="model:PredictedPriceActor",
        config_path="model:PredictedPriceConfig",
        config=dict(
            source_symbol=instrument_ids[0],
            target_symbol=instrument_ids[1],
        ),
    )

    # Create strategy
    strategy = ImportableStrategyConfig(
        strategy_path="strategy:PairTrader",
        config_path="strategy:PairTraderConfig",
        config=dict(
            source_symbol=instrument_ids[0],
            target_symbol=instrument_ids[1],
            notional_trade_size_usd=notional_trade_size_usd,
            **strategy_kwargs,
        ),
    )

    # Create backtest engine
    engine = BacktestEngineConfig(
        trader_id="BACKTESTER-001",
        cache=CacheConfig(tick_capacity=100_000),
        logging=LoggingConfig(
            log_level=log_level,
            bypass_logging=bypass_logging,
        ),
        streaming=StreamingConfig(catalog_path=str(catalog.path)) if persistence else None,
        risk_engine=RiskEngineConfig(max_order_submit_rate="1000/00:00:01"),  # type: ignore
        strategies=[strategy],
        actors=[prediction],
    )
    venues = [
        BacktestVenueConfig(
            name="NASDAQ",
            oms_type="NETTING",
            account_type="CASH",
            base_currency="USD",
            starting_balances=["1_000_000 USD"],
        )
    ]

    data = [
        BacktestDataConfig(
            data_cls="nautilus_trader.model.data:Bar",
            catalog_path=str(catalog.path),
            catalog_fs_protocol=catalog.fs_protocol,
            catalog_fs_storage_options=catalog.fs_storage_options,
            instrument_id=InstrumentId.from_str(instrument_id),
            start_time=start_time,
            end_time=end_time,
        )
        for instrument_id in instrument_ids
    ]

    run_config = BacktestRunConfig(engine=engine, venues=venues, data=data)
    node = BacktestNode(configs=[run_config])
    node.run()


if __name__ == "__main__":
    # typer.run(main)
    catalog = CATALOG
    assert catalog.instruments(), "Couldn't load instruments, have you run `poetry run inv extract-catalog`?"
    main(
        catalog=catalog,
        instrument_ids=("SMH.NASDAQ", "SOXX.NASDAQ"),
        log_level="INFO",
        persistence=True,
        end_time="2020-06-01",
    )
    # print(result.instance_id)
