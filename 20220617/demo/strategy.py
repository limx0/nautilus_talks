import datetime
from functools import partial
from typing import Optional

import pandas as pd
from model import ModelUpdate, Prediction
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.c_enums.order_side import OrderSideParser
from nautilus_trader.model.data.bar import Bar, BarSpecification
from nautilus_trader.model.data.base import DataType
from nautilus_trader.model.enums import OrderSide, PositionSide, TimeInForce
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
from util import make_bar_type, one


class PairTraderConfig(StrategyConfig):
    source_symbol: str
    target_symbol: str
    notional_trade_size_usd: int = 10_000
    min_model_timedelta: datetime.timedelta = datetime.timedelta(days=1)
    trade_width_std_dev: float = 2.5
    bar_spec: str = "10-SECOND-LAST"


class PairTrader(Strategy):
    def __init__(self, config: PairTraderConfig):
        super().__init__(config=config)
        self.source_id = InstrumentId.from_str(config.source_symbol)
        self.target_id = InstrumentId.from_str(config.target_symbol)
        self.model: Optional[ModelUpdate] = None
        self.hedge_ratio: Optional[float] = None
        self.std_pred: Optional[float] = None
        self.prediction: Optional[float] = None
        self._current_edge: float = 0.0
        self._current_required_edge: float = 0.0
        self.bar_spec = BarSpecification.from_str(self.config.bar_spec)

    def on_start(self):
        # Set instruments
        self.source = self.cache.instrument(self.source_id)
        self.target = self.cache.instrument(self.target_id)

        # Subscribe to bars
        self.subscribe_bars(make_bar_type(instrument_id=self.source_id, bar_spec=self.bar_spec))
        self.subscribe_bars(make_bar_type(instrument_id=self.target_id, bar_spec=self.bar_spec))

        # Subscribe to model and predictions
        self.subscribe_data(data_type=DataType(ModelUpdate, metadata={"instrument_id": self.target_id.value}))
        self.subscribe_data(data_type=DataType(Prediction, metadata={"instrument_id": self.target_id.value}))

    def on_bar(self, bar: Bar):
        self._check_for_entry(bar)
        self._check_for_exit(timer=None, bar=bar)

    def on_data(self, data: Data):
        if isinstance(data, ModelUpdate):
            self._on_model_update(data)
        elif isinstance(data, Prediction):
            self._on_prediction(data)
        else:
            raise TypeError()

    def on_event(self, event: Event):
        self._check_for_hedge(timer=None, event=event)
        if isinstance(event, (PositionOpened, PositionChanged)):
            position = self.cache.position(event.position_id)
            self._log.info(f"{position}", color=LogColor.YELLOW)
            assert position.quantity < 80

    def _on_model_update(self, model_update: ModelUpdate):
        self.model = model_update.model
        self.hedge_ratio = model_update.hedge_ratio
        self.std_pred = model_update.std_prediction

    def _on_prediction(self, prediction: Prediction):
        self.prediction = prediction.prediction
        self._update_theoretical(prediction=prediction)

    def _update_theoretical(self, prediction: Prediction):
        # Update the "edge" between our theoretical for `target` and the actual market
        quote_right: Bar = self.cache.bar(make_bar_type(self.target_id, bar_spec=self.bar_spec))
        if not quote_right:
            return

        self._current_edge = 0
        market_right = quote_right.close
        if (prediction.prediction - market_right) > 0:
            self._current_edge = prediction.prediction - market_right
        elif (market_right - prediction.prediction) > 0:
            self._current_edge = market_right - prediction.prediction

    def _check_for_entry(self, bar: Bar):
        if bar.type.instrument_id == self.target_id and self.prediction is not None:
            # Send in orders
            quote_target: Bar = self.cache.bar(make_bar_type(self.target_id, bar_spec=self.bar_spec))
            if not quote_target:
                return

            market_right = quote_target.close
            self._current_required_edge = self.std_pred * self.config.trade_width_std_dev

            if self._current_edge > self._current_required_edge:
                # Our theoretical price is above the market; we want to buy
                side = OrderSide.BUY
                max_volume = int(self.config.notional_trade_size_usd / market_right)
                capped_volume = self._cap_volume(instrument_id=self.target_id, max_quantity=max_volume)
                price = self.prediction - self._current_required_edge
                self._log.debug(f"{OrderSideParser.to_str_py(side)} {max_volume=} {capped_volume=} {price=}")
            elif self._current_edge < -self._current_required_edge:
                # Our theoretical price is below the market; we want to sell
                side = OrderSide.SELL
                max_volume = int(self.config.notional_trade_size_usd / market_right)
                capped_volume = self._cap_volume(instrument_id=self.target_id, max_quantity=max_volume)
                price = self.prediction + self._current_required_edge
                self._log.debug(f"{OrderSideParser.to_str_py(side)} {max_volume=} {capped_volume=} {price=}")
            else:
                return
            if capped_volume == 0:
                # We're at our max limit, cancel any remaining orders and return
                for order in self.cache.orders_open(instrument_id=self.target_id, strategy_id=self.id):
                    self.cancel_order(order=order)
                return
            self._log.info(
                f"Entry opportunity: {OrderSideParser.to_str_py(side)} market={market_right}, "
                f"theo={self.prediction} {capped_volume=} ({self._current_edge=}, {self._current_required_edge=})"
            )
            # Cancel any existing orders
            for order in self.cache.orders_open(instrument_id=self.target_id, strategy_id=self.id):
                self.cancel_order(order=order)
            order = self.order_factory.limit(
                instrument_id=self.target_id,
                order_side=side,
                price=Price(price, self.target.price_precision),
                quantity=Quantity.from_int(capped_volume),
                time_in_force=TimeInForce.IOC,
            )
            self._log.info(f"ENTRY {order.info()}", color=LogColor.BLUE)
            self.submit_order(order)

    def _cap_volume(self, instrument_id: InstrumentId, max_quantity: int) -> int:
        position_quantity = 0
        position = one(self.cache.positions(instrument_id=instrument_id, strategy_id=self.id))
        if position is not None:
            position_quantity = position.quantity
        return max(0, max_quantity - position_quantity)

    def _check_for_hedge(self, timer=None, event: Optional[Event] = None):
        if not (isinstance(event, (PositionEvent,)) and event.instrument_id == self.target_id):
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
                callback=partial(self._check_for_hedge, event=event),
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
            target_position: Position = one(self.cache.positions(instrument_id=self.source_id, strategy_id=self.id))
            if target_position.is_closed:
                return
            quantity = target_position.quantity
            side = self._opposite_side(target_position.side)
        else:
            # (possibly) Increasing our position in hedge instrument
            side = self._opposite_side(base_position.side)
            quantity = self._cap_volume(instrument_id=self.source_id, max_quantity=hedge_quantity)

        if quantity == 0:
            # Fully hedged, cancel any existing orders
            for order in self.cache.orders_open(instrument_id=self.source_id, strategy_id=self.id):
                self.cancel_order(order=order)
            raise RepeatedEventComplete
        elif self.cache.orders_inflight(instrument_id=self.source_id, strategy_id=self.id):
            # Don't send more orders if we have some currently in-flight
            return

        # Cancel any existing orders
        for order in self.cache.orders_open(instrument_id=self.source_id, strategy_id=self.id):
            self.cancel_order(order=order)
        order = self.order_factory.market(
            instrument_id=self.source_id,
            order_side=side,
            quantity=Quantity.from_int(quantity),
        )
        self._log.info(f"ENTRY HEDGE {order.info()}", color=LogColor.BLUE)
        self.submit_order(order)
        return order

    def _check_for_exit(self, timer=None, bar: Optional[Bar] = None):
        if not self.cache.positions(strategy_id=self.id):
            return

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
                callback=partial(self._check_for_exit, bar=bar),
            )
        except RepeatedEventComplete:
            # Hedge is complete, return
            if timer_name in self.clock.timer_names():
                self.clock.cancel_timer(timer_name)
            return

    def _exit_position(self, bar: Bar):
        position: Position = one(self.cache.positions(instrument_id=self.target_id, strategy_id=self.id))
        if position is not None:
            if position.is_closed:
                raise RepeatedEventComplete()
            if self._current_edge < (self._current_required_edge * 0.25):
                if self.cache.orders_inflight(instrument_id=self.target_id, strategy_id=self.id):
                    # Order currently in-flight, don't send again
                    return
                self._log.info(
                    f"Trigger to close position {self._current_edge=} {self._current_required_edge} (* 0.25)",
                    color=LogColor.CYAN,
                )
                # We're close back to fair value, we should try and close our position
                order = self.order_factory.market(
                    instrument_id=self.target_id,
                    order_side=self._opposite_side(position.side),
                    quantity=position.quantity,
                )
                self._log.info(f"CLOSE {order.info()}", color=LogColor.BLUE)
                self.submit_order(order)

    def _opposite_side(self, side: PositionSide):
        return {PositionSide.LONG: OrderSide.SELL, PositionSide.SHORT: OrderSide.BUY}[side]

    def on_stop(self):
        self.close_all_positions(self.source_id)
        self.close_all_positions(self.target_id)


class RepeatedEventComplete(Exception):
    pass
