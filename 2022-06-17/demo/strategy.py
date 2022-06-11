import datetime
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import secs_to_nanos, unix_nanos_to_dt
from nautilus_trader.core.message import Event
from nautilus_trader.model.c_enums.order_side import OrderSideParser
from nautilus_trader.model.data.bar import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AggregationSource, OrderSide, PositionSide, TimeInForce
from nautilus_trader.model.events.position import (
    PositionChanged,
    PositionClosed,
    PositionEvent,
    PositionOpened,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class PairTraderConfig(StrategyConfig):
    left_symbol: str
    right_symbol: str
    notional_trade_size_usd: int = 10_000
    min_model_timedelta: datetime.timedelta = datetime.timedelta(days=1)
    trade_width_std_dev: float = 1.5
    bar_spec: str = "10-SECOND-LAST"


class PairTrader(Strategy):
    def __init__(self, config: PairTraderConfig):
        super().__init__(config=config)
        self.left_id = InstrumentId.from_str(config.left_symbol)
        self.right_id = InstrumentId.from_str(config.right_symbol)
        self.model: Optional[LinearRegression] = None
        self.theo_right: Optional[float] = None
        self.hedge_ratio: Optional[float] = None
        self._min_model_timedelta = secs_to_nanos(self.config.min_model_timedelta.total_seconds())
        self._last_model = pd.Timestamp(0)
        self._current_edge: float = 0.0
        self._current_required_edge: float = 0.0
        self.bar_spec = BarSpecification.from_str(self.config.bar_spec)

    def bar_type(self, instrument_id: InstrumentId):
        return BarType(
            instrument_id=instrument_id, bar_spec=self.bar_spec, aggregation_source=AggregationSource.EXTERNAL
        )

    def on_start(self):
        # Set instruments
        self.left = self.cache.instrument(self.left_id)
        self.right = self.cache.instrument(self.right_id)

        # Subscribe to bars
        self.subscribe_bars(self.bar_type(instrument_id=self.left_id))
        self.subscribe_bars(self.bar_type(instrument_id=self.right_id))

    def on_bar(self, bar: Bar):
        self.check_model_fit(bar)
        self.update_theoretical(bar)
        self.check_for_entry(bar)
        self.check_for_exit(timer=None, bar=bar)

    def on_event(self, event: Event):
        self.check_for_hedge(timer=None, event=event)
        if isinstance(event, (PositionOpened, PositionChanged)):
            position = self.cache.position(event.position_id)
            self._log.info(f"{position}", color=LogColor.YELLOW)
            assert position.quantity < 80

    def check_model_fit(self, bar: Bar):
        # Check we have the minimum required data
        def _check_first_tick(instrument_id):
            bars = self.cache.bars(bar_type=self.bar_type(instrument_id))
            if not bars:
                return False
            delta = self.clock.timestamp_ns() - bars[-1].ts_init
            return delta > self._min_model_timedelta

        if not (_check_first_tick(self.left_id) and _check_first_tick(self.right_id)):
            return

        # Check we haven't fit a model yet today
        if unix_nanos_to_dt(bar.ts_init).date() == self._last_model.date():
            return

        # Fit a model
        df = bars_to_dataframe(
            left_instrument=self.left_id.value,
            left_bars=self.cache.bars(bar_type=self.bar_type(self.left_id)),
            right_instrument=self.right_id.value,
            right_bars=self.cache.bars(bar_type=self.bar_type(self.right_id)),
        ).dropna()
        X = df.loc[:, self.left_id.value].astype(float).values.reshape(-1, 1)
        Y = df.loc[:, self.right_id.value].astype(float).values.reshape(-1, 1)
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X, Y)
        self.log.info(f"Fit model @ {unix_nanos_to_dt(bar.ts_init)}, r2: {r2_score(Y, self.model.predict(X))}")
        self._last_model = unix_nanos_to_dt(bar.ts_init)
        pred = self.model.predict(X)
        errors = pred - Y
        self.std_pred = errors.std()
        self.hedge_ratio = float(self.model.coef_[0][0])

    def update_theoretical(self, bar: Bar):
        if bar.type.instrument_id == self.left_id and self.model is not None:
            # Update our theoretical value for `right`
            X = np.asarray([[bar.close.as_double()]])
            [[self.theo_right]] = self.model.predict(X)
        elif bar.type.instrument_id == self.right_id and self.theo_right is not None:
            # Update the edge between our theoretical for `right` and the actual market
            quote_right: Bar = self.cache.bar(self.bar_type(self.right_id))
            if not quote_right:
                return

            self._current_edge = 0
            market_right = quote_right.close
            if (self.theo_right - market_right) > 0:
                self._current_edge = self.theo_right - market_right
            elif (market_right - self.theo_right) > 0:
                self._current_edge = market_right - self.theo_right

    def check_for_entry(self, bar: Bar):
        if bar.type.instrument_id == self.left_id and self.model is not None:
            # Update our theoretical value for `right`
            X = np.asarray([[bar.close.as_double()]])
            [[self.theo_right]] = self.model.predict(X)
        elif bar.type.instrument_id == self.right_id and self.theo_right is not None:
            # Send in orders
            quote_right: Bar = self.cache.bar(self.bar_type(self.right_id))
            if not quote_right:
                return

            market_right = quote_right.close
            self._current_required_edge = self.std_pred * self.config.trade_width_std_dev

            if self._current_edge > self._current_required_edge:
                # Our theoretical price is above the market; we want to buy
                side = OrderSide.BUY
                max_volume = int(self.config.notional_trade_size_usd / market_right)
                capped_volume = self.cap_volume(instrument_id=self.right_id, max_quantity=max_volume)
                price = self.theo_right - self._current_required_edge
                self._log.debug(f"{OrderSideParser.to_str_py(side)} {max_volume=} {capped_volume=} {price=}")
            elif self._current_edge < -self._current_required_edge:
                # Our theoretical price is below the market; we want to sell
                side = OrderSide.SELL
                max_volume = int(self.config.notional_trade_size_usd / market_right)
                capped_volume = self.cap_volume(instrument_id=self.right_id, max_quantity=max_volume)
                price = self.theo_right + self._current_required_edge
                self._log.debug(f"{OrderSideParser.to_str_py(side)} {max_volume=} {capped_volume=} {price=}")
            else:
                return
            if capped_volume == 0:
                # We're at our max limit, cancel any remaining orders and return
                for order in self.cache.orders_open(instrument_id=self.right_id, strategy_id=self.id):
                    self.cancel_order(order=order)
                return
            self._log.info(
                f"Entry opportunity: {OrderSideParser.to_str_py(side)} market={market_right}, "
                f"theo={self.theo_right} {capped_volume=} ({self._current_edge=}, {self._current_required_edge=})"
            )
            # Cancel any existing orders
            for order in self.cache.orders_open(instrument_id=self.right_id, strategy_id=self.id):
                self.cancel_order(order=order)
            order = self.order_factory.limit(
                instrument_id=self.right_id,
                order_side=side,
                price=Price(price, self.right.price_precision),
                quantity=Quantity.from_int(capped_volume),
                time_in_force=TimeInForce.IOC,
            )
            self._log.info(f"ENTRY {order.info()}", color=LogColor.BLUE)
            self.submit_order(order)

    def cap_volume(self, instrument_id: InstrumentId, max_quantity: int) -> int:
        position_quantity = 0
        position = one(self.cache.positions(instrument_id=instrument_id, strategy_id=self.id))
        if position is not None:
            position_quantity = position.quantity
        return max(0, max_quantity - position_quantity)

    def check_for_hedge(self, timer=None, event: Optional[Event] = None):
        if not (isinstance(event, (PositionEvent,)) and event.instrument_id == self.right_id):
            return

        timer_name = f"hedge-{self.id}"
        try:
            self._hedge_position(event)
            # Keep scheduling this method to run until we're hedged
            if timer_name in self.clock.timer_names():
                self.clock.cancel_timer(timer_name)
            self.clock.set_time_alert(
                name=timer_name,
                alert_time=self.clock.utc_now() + pd.Timedelta(seconds=2),
                callback=partial(self.check_for_hedge, event=event),
            )
        except RepeatedEventComplete:
            # Hedge is complete, return
            if timer_name in self.clock.timer_names():
                self.clock.cancel_timer(timer_name)
            return

    def _hedge_position(self, event: PositionEvent):
        # We've opened or changed position in our base instrument, we will likely need to hedge.
        base_position = self.cache.position(event.position_id)
        hedge_quantity = int(round(base_position.quantity / self.hedge_ratio, 0))
        if isinstance(event, PositionClosed):
            # (possibly) Reducing our position in the hedge instrument
            target_position: Position = one(self.cache.positions(instrument_id=self.left_id, strategy_id=self.id))
            if target_position.is_closed:
                return
            quantity = target_position.quantity
            side = self.opposite_side(target_position.side)
        else:
            # (possibly) Increasing our position in hedge instrument
            side = self.opposite_side(base_position.side)
            quantity = self.cap_volume(instrument_id=self.left_id, max_quantity=hedge_quantity)

        if quantity == 0:
            # Fully hedged, cancel any existing orders
            for order in self.cache.orders_open(instrument_id=self.left_id, strategy_id=self.id):
                self.cancel_order(order=order)
            raise RepeatedEventComplete
        elif self.cache.orders_inflight(instrument_id=self.left_id, strategy_id=self.id):
            # Don't send more orders if we have some currently in-flight
            return

        # Cancel any existing orders
        for order in self.cache.orders_open(instrument_id=self.left_id, strategy_id=self.id):
            self.cancel_order(order=order)
        order = self.order_factory.market(
            instrument_id=self.left_id,
            order_side=side,
            # price=Price(self.left_last, self.left.price_precision),
            quantity=Quantity.from_int(quantity),
        )
        self._log.info(f"ENTRY HEDGE {order.info()}", color=LogColor.BLUE)
        self.submit_order(order)
        return order

    def check_for_exit(self, timer=None, bar: Optional[Bar] = None):
        # Keep checking that we have successfully got a hedge
        timer_name = f"exit-{self.id}"
        try:
            self._exit_position(bar=bar)
            # Keep scheduling this method to run until we're exited
            if timer_name in self.clock.timer_names():
                self.clock.cancel_timer(timer_name)
            self.clock.set_time_alert(
                name=timer_name,
                alert_time=self.clock.utc_now() + pd.Timedelta(seconds=2),
                callback=partial(self.check_for_exit, bar=bar),
            )
        except RepeatedEventComplete:
            # Hedge is complete, return
            if timer_name in self.clock.timer_names():
                self.clock.cancel_timer(timer_name)
            return

    def _exit_position(self, bar: Bar):
        position: Position = one(self.cache.positions(instrument_id=self.right_id, strategy_id=self.id))
        if position is not None:
            if position.is_closed:
                raise RepeatedEventComplete()
            if self._current_edge < (self._current_required_edge * 0.25):
                if self.cache.orders_inflight(instrument_id=self.right_id, strategy_id=self.id):
                    # Order currently in-flight, don't send again
                    return
                self._log.info(
                    f"Trigger to close position {self._current_edge=} {self._current_required_edge} (* 0.25)",
                    color=LogColor.CYAN,
                )
                # We're close back to fair value, we should try and close our position
                order = self.order_factory.market(
                    instrument_id=self.right_id,
                    order_side=self.opposite_side(position.side),
                    # price=Price(self.left_last, self.right.price_precision),
                    quantity=position.quantity,
                )
                self._log.info(f"CLOSE {order.info()}", color=LogColor.BLUE)
                self.submit_order(order)

    def opposite_side(self, side: PositionSide):
        return {PositionSide.LONG: OrderSide.SELL, PositionSide.SHORT: OrderSide.BUY}[side]

    @property
    def left_last(self):
        bar: Bar = self.cache.bar(bar_type=self.bar_type(self.left_id))
        return bar.close

    @property
    def right_last(self):
        bar: Bar = self.cache.bar(bar_type=self.bar_type(self.right_id))
        return bar.close

    def on_stop(self):
        self.close_all_positions(self.left_id)
        self.close_all_positions(self.right_id)


class RepeatedEventComplete(Exception):
    pass


def one(iterable):
    if len(iterable) == 0:
        return None
    elif len(iterable) > 1:
        raise AssertionError("Too many values")
    else:
        return iterable[0]


def ticks_to_dataframe(left_instrument: str, left_ticks, right_instrument: str, right_ticks) -> pd.DataFrame:
    ldf = pd.DataFrame([t.to_dict(t) for t in left_ticks]).set_index("ts_init")
    rdf = pd.DataFrame([t.to_dict(t) for t in right_ticks]).set_index("ts_init")
    data = pd.concat(
        [
            ldf.rename({"price": left_instrument}, axis=1)[[left_instrument]],
            rdf.rename({"price": right_instrument}, axis=1)[[right_instrument]],
        ]
    ).sort_index()
    return data.fillna(method="ffill", limit=1).sort_index()


def bars_to_dataframe(
    left_instrument: str, left_bars: List[Bar], right_instrument: str, right_bars: List[Bar]
) -> pd.DataFrame:
    def bars_to_frame(bars, instrument_id):
        df = pd.DataFrame([t.to_dict(t) for t in bars]).astype({"close": float})
        return df.assign(instrument_id=instrument_id).set_index(["instrument_id", "ts_init"])

    ldf = bars_to_frame(bars=left_bars, instrument_id=left_instrument)
    rdf = bars_to_frame(bars=right_bars, instrument_id=right_instrument)
    data = pd.concat([ldf, rdf])["close"].unstack(0).sort_index().fillna(method="ffill")
    return data
