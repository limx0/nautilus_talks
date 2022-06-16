---
marp: true
---
<!-- class: invert -->

# NautilusTrader
![bg right width:500px](nautilus_logo.jpeg)

---
<!-- @chris -->

## Talk overview

* Speaker intro
* Introduction to NautilusTrader
* Features of NautilusTrader
* Using NautilusTrader
* Current state of the project
* Live Crypto examples
* Live walkthrough of pairs trading strategy
* Future developments

---
<!-- @chris -->

## Speaker - Chris
* Non-traditional background coming from another industry
* A long fascination and interest in financial markets, programming and computing
* Started writing a trading platform as MetaTrader 4, C Trader, NinjaTrader didn’t meet requirements
* NautilusTrader originally written in C#
* Started working for an FX trading firm based in Singapore focusing on ML
* Currently a software engineer for a market data vendor

---
<!-- @brad -->

## Speaker - Brad

* Equity Options trader/researcher by profession (Optiver / IMC)
* Spend my days writing Python for research & automation of trading strategies
* Also interested in sports betting 
    - dabbling in tennis and basketball betting in the past with very mild success

---

## Speaker - Brad
* Discovered Nautilus over a year ago and did a bit of a deep dive
* At the time I had been working on a similar (much more basic) system for quite a few years
* Decided to drop my (years of) work straight away and contribute to Nautilus
* Worked on:
    * Adapters (Betfair, Interactive Brokers)
    * Data loading / persistence
    * Helping Chris with ideas / implementation of other features I thought were necessary for a production trading system

---
<!-- @chris -->

## History and progression of NautilusTrader

* Wanted an extendable robust foundation for a platform using domain driven design, messaging, and to be more performant than existing platforms
* The traders were using Python and so by necessity developed what started as a thin Python client for the now distributed platform with C# data, risk and execution services
* Platform has a deep history in FX trading with FIX connections, which has influenced naming and features throughout
* Eventually the Python codebase (now heavily leveraging Cython), could stand on its own as a complete platform
* Has been open sourced for over two years, and has reached a level of maturity where we are excited to present it and hope it provides some value
* Moving towards Rust as the core language, there will always be a Python API. Eventually all Cython will go

---
<!-- @chris -->

## What is NautilusTrader, and why?
---

## Introduction

* NautilusTrader is an open-source, high-performance, production-grade algorithmic trading platform, 
* providing quantitative traders with the ability to backtest portfolios of automated trading strategies on historical data with an event-driven engine, 
* and also deploy those same strategies live, with no code changes

---

## Features

The major benefits of the platform are:

* **Highly performant event-driven Python** - native binary core components
* **Parity between backtesting and live trading** - identical strategy code
* **Reduced operational risk** - risk management functionality, logical correctness and type safety
* **Highly extendable** - message bus, custom components and actors, custom data, custom adapters
* **Open source** - NautilusTrader is fully open source software

---

**Performance**
* Event-driven backtest systems tend to have lower performance than vectorized methods
* Nautilus is written in Cython, with a Rust core currently being introduced incrementally
* The engine handles >100,000 events per second, allowing for backtesting tick data for even the most liquid products
* Backtests are trivially parallelizable via configuration objects

---

**Backtest/live parity**

* Nautilus is structured such that a huge portion of the system is unaware if it is running in backtest or live
* This is formalized in the actual code, with a common `NautilusKernel` object containing all the engines and core components
* The same strategy code which is developed for research and backtesting can be used to run live trading, with zero changes required

---

**Flexible**

* Multiple strategies/venues/instruments in a single instance
* Define custom external data types easily with full integration within the system
* Pass messages between components via the message bus

---

**Advanced**

* The platform favours quality over quantity when it comes to integrations, offering as much of the exchanges functionality as possible
* Advanced order types and conditional triggers. Execution instructions `post-only`, `reduce-only`, and icebergs. Contingency order lists including `OCO, OTO`
* Time in force for orders `IOC, FOK, GTD, AT_THE_OPEN, AT_THE_CLOSE`
* System clock allows scheduling events in backtest and live scenarios

---

**Extendable**

* Integrates with any REST, WebSocket or FIX API via modular adapters
* Capable of handling various asset classes including (but not limited to) FX, Equities, Futures, Options, CFDs, Crypto and Sports Betting - across multiple venues simultaneously
* Extend the core system via Actors and the message bus
* Custom data, portfolio statistics, etc

---

**Open Source**

* NautilusTrader is and will remain open source software
* Safely run your proprietary code & data on premises
* Cloud offering in the works, focusing on
    - Scaling backtesting
    - Visualisation and monitoring
    - Easy deployment of live instances
    - Data and code will remain your IP - Cloud will simply orchestrate

---

# Architecture

---

* Both a framework for trading systems, with several system implementations for backtesting, live trading and even a sandbox environment on the way (live data with simulated execution)
* Ports and adapters architecture, which adopted the '_engines architecture_' of the distributed C# system
* Highly modular and very well tested with a suite of over 3500 unit, integration and acceptance tests
* Loose coupling of the system components via a highly efficient message bus and single cache has allowed us to move quite quickly with changes and improvements
* Components interact using various messaging patterns through the message bus (Pub/Sub, Req/Rep or point-to-point) → meaning they don’t need knowledge of, or direct references/dependencies on each other
* The message bus (written in Cython) is highly performant, with benchmarks between direct method calls or message bus being nearly identical (slightly in favor of direct calls). The loose coupling makes this worth it

---

![bg center width:850px](architecture-overview.png)

---

## Language choices

* Using the right tools for the job: 
  - Python for data and non performance critical code, you can’t beat the ecosystem for iterating and prototyping fast, data and ML
  - Rust and Cython for performance critical components, enables the high performance even when run in a complex event-driven context
* Introduction of the Rust core, and expect gradual yet continuous improvements, we’re effectively creating a framework of trading components and a trading domain model, with Python bindings over a native binary core similar to numpy and pandas which also enjoy extremely high performance

---

# Nautilus as a Complete Trading System
<!-- @brad -->

<!-- Talk about concrete components / features -->

* _or why wouldn't I just use `XYZ project I found on github`_

---

* Backtest simulations → fill models, order latency, among other models.
* Orders → order tags, Time-In-Force, stop-loss/take-profit, bracket orders
* Accounts → cash and margin
* Positions → netting and hedging Order-Management-System, realised, unrealised pnls
* Portfolio component → query positions, values
* Persistence → every piece of data, and every event (live and backtest) can be persisted automatically
* CacheDatabase → store execution state (orders) and strategy state in-memory or Redis

---

* Actors → Custom components that can interact with the system in any way
* Message bus → Point-to-point, request-response, or pub-sub messaging across the entire system
* Custom data → Market stats, scoreboard, twitter feed, Historic and live
* Risk engine → Fat finger checks, order frequency limits etc
* Cache → easily pull data from a central place from strategy/actor with a single line
* Clock / Time events → Trigger events at times or on a timer for custom callbacks or triggers

---
<!-- @brad -->

# Current State of NautilusTrader

* Multiple people using in production
* Many more testing the waters with backtesting
* Short term areas that need work / testing: (it's not perfect!)
    - Accounting components (more testing required for margin accounts, sub-accounts)
    - Configuration and deployment

---

# User Guide

---

# Getting started with NautilusTrader
<!-- @brad -->

* Jupyterlab docker image (including sample data and backtest notebook)
    - `docker run -p 8888:8888 ghcr.io/nautechsystems/jupyterlab:develop`

* Examples directory
    - https://github.com/nautechsystems/nautilus_trader/tree/master/examples

---

## Writing a Strategy
<!-- @brad -->

---

### Configuration

```python
class OrderBookImbalanceConfig(StrategyConfig):
    instrument_id: str
    max_trade_size: Decimal
    trigger_min_size: float = 100.0
    trigger_imbalance_ratio: float = 0.20


class OrderBookImbalance(Strategy):
    def __init__(self, config: OrderBookImbalanceConfig):
```

---

### Start up

```python
    def on_start(self):
        """Actions to be performed on strategy start."""

        self.instrument = self.cache.instrument(self.config.instrument_id)
        
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.subscribe_quote_ticks(instrument_id=self.instrument.id)
```

---

### Data methods: 

`on_quote_tick/trade_tick/bar(self, data)` etc)

```python
def on_quote_tick(self, tick: QuoteTick):
    bid_volume = tick.bid_size
    ask_volume = tick.ask_size
    if not (bid_volume and ask_volume):
        return
    self.log.info(f"Top level: {tick.bid} @ {tick.ask}")
    smaller = min(bid_volume, ask_volume)
    larger = max(bid_volume, ask_volume)
    ratio = smaller / larger
    self.check_trigger(ratio)
```

---

### Event method 
(`OrderInitialized/Accepted/Filled PositionOpened/Changed` etc)

```python
def on_event(self, event: Event):
    if isinstance(event, OrderFilled):
        # Do something
        pass

```

---

# Writing an Adapter
<!-- @brad -->

---

## Adapters

* Convert external data or events into nautilus format
* Split into:
    * `DataClient` - quotes, trades, tweets, etc
    * `ExecutionClient` - order fills, positions, account balance updates, etc

---

```python
class Tweet(Data):
    def __init__(self, text: str, ts_init: int):
        super().__init__(ts_init, ts_init)
        self.text = text


class TwitterDataClient(LiveDataClient):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.stream = requests.get(
            url="https://api.twitter.com/2/tweets/search/stream", 
            auth=bearer_oauth, 
            stream=True,
        ) 

    def connect(self):
        self._loop.create_task(on_tweet())
        self._connected(True)

    async def on_tweet(self):
        for response_line in self.stream.iter_lines():
            if response_line:
                json_resp = json.loads(response_line)
                tweet = Tweet(text=json_resp['text'], ts_init=json_resp['timestamp'])
                # Feed data to nautilus engine
                self._handle_data(data=tweet)
            await asyncio.sleep(0.1)
```

---

## Getting data into Nautilus
<!-- @brad -->

- Nautilus is strongly typed → performance comes from strict typing
- Can't just load CSV files (but only a tiny bit of work required)
- A couple of options:

---
### Wranglers
* The quick choice; load nautilus objects from your own persistent data source 
    - CSV, JSON, Parquet, Pandas
* Objects are created on the fly
* Suitable while experimenting or your data is small (end of day data for example)

---

### DataCatalog
* The performant choice; nautilus will write your data into Parquet files optimised for reading
* Objects basically loaded from "file" as is (minimal conversions required)
* Uses the excellent `fsspec` library as a base (data can be loaded from sftp/s3/gcs/asdl/etc)
* Move improvements to come with rust ecosystem (zero-copy loading objects straight into memory)

---

# Running Backtests

---
## Running Backtests

<!-- @chris -->

* Nautilus requires a little bit of configuration to run.
    - it's a fully featured system and has correctness as a core principle
* Backtest configuration can be done in a couple of ways:
    - In a python file, manually, as in the examples
    - Using a `pydantic` model, the `"BacktestRunConfig"`  from python or JSON (`DataCatalog` only)

Walking through one of the manual examples:

_(exact same configuration applies to BacktestRunConfig)_

---

First, create the engine backtest engine

```python
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="BACKTESTER-001",
        )
    )
```

---

Next, add instruments

```python
    ETHUSDT_BINANCE = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(ETHUSDT_BINANCE)
```

This can be done in quite a few ways :
- Manually construct the actual `Instrument` object
- Use one of the `TestInstrumentProviders`
- Load from a `DataCatalog`
- Load directly from an adapter (broker or exchange)

---

Add the actual data to the engine (in this example; some trade ticks via a `Wrangler`)

```python
    # Load DataFrame from CSV of trade ticks
    provider = TestDataProvider()
    raw = provider.read_csv_ticks("binance-ethusdt-trades.csv")

    # Convert into Nautilus objects for a given instrument
    wrangler = TradeTickDataWrangler(instrument=ETHUSDT_BINANCE)
    ticks = wrangler.process(raw)
    
    # Add data to engine
    engine.add_data(ticks)
```

An `Instrument` is tied to a venue; so nautilus knows this data for `ETHUSDT_BINANCE` belongs to the `BINANCE` venue (covered in next slide) 

---

Define a venue(s) (this will create a simulated exchange, accounts etc)

```python
    BINANCE = Venue("BINANCE")

    engine.add_venue(
        venue=BINANCE,
        oms_type=OMSType.NETTING,
        account_type=AccountType.CASH,  # Spot cash account
        base_currency=None,  # Multi-currency account
        starting_balances=[
            Money(1_000_000, USDT), Money(10, ETH)
        ],
    )
```

_Nautilus checks data and instruments against venues, and will complain if you try and add a venue that does not have data, or define instruments without a venue_

---

Configure, create and add strategy(s)

```python
    config = EMACrossConfig(
        instrument_id=str(ETHUSDT_BINANCE.id),
        bar_type="ETHUSDT.BINANCE-250-TICK-LAST-INTERNAL",
        trade_size=Decimal("0.05"),
        fast_ema=10,
        slow_ema=20,
        order_id_tag="001",
    )

    strategy = EMACross(config=config)

    engine.add_strategy(strategy=strategy)

```

---

Run the backtest (and optionally pull out some reports)

```python
    # Run the engine (from start to end of data)
    engine.run()

    # Optionally view reports
    print(engine.trader.generate_account_report(BINANCE))

    # For repeated backtest runs make sure to reset the engine
    engine.reset()

    # Good practice to dispose of the object
    engine.dispose()
```

--- 

## BacktestRunConfig
<!-- @brad? -->

* A higher level config object also exists; the `BacktestRunConfig`.
* Basically the same options as above, but defined as `pydantic` models
* _Requires using the DataCatalog however_
* Can be more easily loaded from Dicts/JSON or other files
___

Example

```python

@pydantic.dataclasses.dataclass()
class BacktestRunConfig(Partialable):
    engine: Optional[BacktestEngineConfig] = None
    venues: Optional[List[BacktestVenueConfig]] = None
    data: Optional[List[BacktestDataConfig]] = None
    batch_size_bytes: Optional[int] = None


config = BacktestRunConfig(
    engine=engine,
    venues=[venue],
    data=[data],
)

node = BacktestNode(configs=[config])
results: List[BacktestResult] = node.run()

```

---
<!-- @chris -->
# Live Trading

--- 
<!-- @chris  -->
## Running a Live instance

* As per backtesting, there is a handful of config required to get running live
* Very similar to backtesting setup, just pointing at live connections now
    - Instead of `BacktestRunConfig` we use `TradingNodeConfig` object.
* Walking through a live version of the backtest example above:
---

* Start to define the `TradingNodeConfig`, including live-only settings for timeouts etc
* including a cache_database for persisting state (in memory or redis optional)

```python

config_node = TradingNodeConfig(
    trader_id="TESTER-001",
    log_level="INFO",
    exec_engine={
        "reconciliation_lookback_mins": 1440,
    },
    cache_database=CacheDatabaseConfig(type="in-memory"),
    timeout_connection=5.0,
    timeout_reconciliation=5.0,
    timeout_portfolio=5.0,
    timeout_disconnection=5.0,
    timeout_post_stop=2.0,
```

---

Add data client(s);
- This is the live source of market or other data

```python
    data_clients={
        "FTX": FTXDataClientConfig(
            api_key=None,  # "YOUR_FTX_API_KEY"
            api_secret=None,  # "YOUR_FTX_API_SECRET"
            subaccount=None,  # "YOUR_FTX_SUBACCOUNT"
            us=False,  # If client is for FTX US
            instrument_provider=InstrumentProviderConfig(load_all=True),
        ),
    },
)
```

--- 

Add (optionally) execution client(s); 
 - this is where orders will be sent and
 - Accounts, executiion events (fills, positions) received from

```python
    exec_clients={
        "FTX": FTXExecClientConfig(
            api_key=None,  # "YOUR_FTX_API_KEY"
            api_secret=None,  # "YOUR_FTX_API_SECRET"
            subaccount=None,  # "YOUR_FTX_SUBACCOUNT"
            us=False,  # If client is for FTX US
            instrument_provider=InstrumentProviderConfig(load_all=True),
        ),
    },
```

---

With the completed config, create a `TradingNode` and add any strategies

```python
# Instantiate the node with a configuration
node = TradingNode(config=config_node)

# Configure your strategy
strat_config = EMACrossConfig(
    instrument_id="ETH-PERP.FTX",
    bar_type="ETH-PERP.FTX-1-MINUTE-LAST-INTERNAL",
    fast_ema_period=10,
    slow_ema_period=20,
    trade_size=Decimal("0.01"),
    order_id_tag="001",
)
# Instantiate your strategy
strategy = EMACross(config=strat_config)

# Add your strategies, actors and modules
node.trader.add_strategy(strategy)

```

---
Finally, register factories for the clients and build/start the node

```python


# Register your client factories with the node (can take user defined factories)
node.add_data_client_factory("FTX", FTXLiveDataClientFactory)
node.add_exec_client_factory("FTX", FTXLiveExecClientFactory)
node.build()

# Stop and dispose of the node with SIGINT/CTRL+C
if __name__ == "__main__":
    try:
        node.start()
    finally:
        node.dispose()
```

---
<!-- @chris -->

# Quick Demo EMA Cross

---

# Walkthrough - Pairs Trading
<!-- @brad -->
<!-- pairs trading intro -->
<!-- walk through of pairs trading strategy -->

---
<!-- @chris -->

## Future developments
- More Rust
- More Crypto derivatives exchange adapters (chosen to niche into this space), IB provides a great option for accessing traditional markets
- Making adapters easier to write
- Accounting improvements
- Premium cloud product offering fully managed, hybrid cloud, on-prem options - coming soon! (goal is this year)
- https://nautilustrader.io
---
