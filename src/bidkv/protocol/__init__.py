"""bidkv.protocol — CompressionBid 协议层公共 API。"""

from bidkv.protocol.bid import (
    FEATURE_GATE_ID,
    BidAcceptance,
    BidPool,
    CompressionBid,
    compute_utility,
    make_bid_id,
)
from bidkv.protocol.errors import (
    BidCapacityError,
    BidExecutionError,
    BidExpiredError,
    CompressionBidError,
)
from bidkv.protocol.provider import CompressionBidProvider

__all__ = [
    "FEATURE_GATE_ID",
    "BidAcceptance",
    "BidCapacityError",
    "BidExecutionError",
    "BidExpiredError",
    "BidPool",
    "CompressionBid",
    "CompressionBidError",
    "CompressionBidProvider",
    "compute_utility",
    "make_bid_id",
]
