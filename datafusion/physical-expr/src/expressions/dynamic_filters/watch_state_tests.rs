// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The latest watch value must survive periods without subscribers.
use super::*;
use crate::expressions::lit;

#[test]
fn update_without_receivers_retains_latest_watch_generation() {
    let filter = DynamicFilterPhysicalExpr::new(vec![], lit(true));
    assert_eq!(filter.state_watch.receiver_count(), 0);
    filter.update(lit(false)).unwrap();
    assert_eq!(
        filter.state_watch.borrow().generation(),
        filter.inner.read().generation
    );
    let receiver = filter.state_watch.subscribe();
    assert_eq!(
        receiver.borrow().generation(),
        filter.inner.read().generation
    );
}

#[test]
fn completion_without_receivers_retains_complete_watch_state() {
    let filter = DynamicFilterPhysicalExpr::new(vec![], lit(true));
    assert_eq!(filter.state_watch.receiver_count(), 0);
    filter.update(lit(false)).unwrap();
    filter.mark_complete();
    assert!(filter.inner.read().is_complete);
    assert!(matches!(
        *filter.state_watch.borrow(),
        FilterState::Complete { generation } if generation == filter.inner.read().generation
    ));
}

#[test]
fn update_after_completion_publishes_new_generation() {
    let filter = DynamicFilterPhysicalExpr::new(vec![], lit(true));
    filter.mark_complete();

    filter.update(lit(false)).unwrap();
    assert_eq!(filter.inner.read().generation, 2);
}
