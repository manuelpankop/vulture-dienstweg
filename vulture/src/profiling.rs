use std::cell::Cell;

thread_local! {
    static TRANSFER_TRIPS: Cell<u64> = const { Cell::new(0) };
    static TRANSFER_RULES: Cell<u64> = const { Cell::new(0) };
}

pub(crate) fn transfer_trip() {
    TRANSFER_TRIPS.with(|n| n.set(n.get() + 1));
}

pub(crate) fn transfer_rule() {
    TRANSFER_RULES.with(|n| n.set(n.get() + 1));
}

pub(crate) fn take_transfer_counts() -> (u64, u64) {
    (TRANSFER_TRIPS.with(|n| n.replace(0)), TRANSFER_RULES.with(|n| n.replace(0)))
}
