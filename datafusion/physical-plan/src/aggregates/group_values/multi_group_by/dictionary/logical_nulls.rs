// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::GroupColumn;
use arrow::array::{
    Array, ArrayRef, AsArray, BooleanBufferBuilder, DictionaryArray, PrimitiveArray,
};
use arrow::datatypes::{ArrowDictionaryKeyType, DataType};
use arrow::downcast_dictionary_array;
use datafusion_common::Result;
use datafusion_expr::GroupSelection;
use std::mem::size_of;
use std::sync::Arc;

pub(in crate::aggregates::group_values::multi_group_by) fn wrap(
    inner: Box<dyn GroupColumn>,
    value_type: &DataType,
) -> Box<dyn GroupColumn> {
    if matches!(
        value_type,
        DataType::Null
            | DataType::Dictionary(..)
            | DataType::RunEndEncoded(..)
            | DataType::Union(..)
    ) {
        Box::new(EncodedNullDictionaryColumn { inner })
    } else {
        inner
    }
}

/// Adapt value types that can represent a NULL without a physical validity bit.
struct EncodedNullDictionaryColumn {
    inner: Box<dyn GroupColumn>,
}

impl EncodedNullDictionaryColumn {
    fn normalized(array: &ArrayRef) -> Option<ArrayRef> {
        let dictionary = array.as_ref();
        downcast_dictionary_array!(
            dictionary => Self::normalize_keys(dictionary),
            _ => unreachable!("dictionary group column requires a dictionary array"),
        )
    }

    fn normalize_keys<K: ArrowDictionaryKeyType>(
        dict: &DictionaryArray<K>,
    ) -> Option<ArrayRef> {
        if !dict.values().is_nullable() {
            return None;
        }
        let nulls = dict.logical_nulls()?;
        if nulls.null_count() == dict.keys().null_count() {
            return None;
        }
        let keys = PrimitiveArray::<K>::new(dict.keys().values().clone(), Some(nulls));
        Some(Arc::new(DictionaryArray::<K>::new(
            keys,
            Arc::clone(dict.values()),
        )))
    }

    fn normalized_row(array: &ArrayRef, row: usize) -> Option<ArrayRef> {
        if array.is_null(row) || !array.as_any_dictionary().values().is_nullable() {
            return None;
        }
        Self::normalized(&array.slice(row, 1))
    }
}

impl GroupColumn for EncodedNullDictionaryColumn {
    fn equal_to(&self, lhs_row: usize, array: &ArrayRef, rhs_row: usize) -> bool {
        if let Some(normalized) = Self::normalized_row(array, rhs_row) {
            self.inner.equal_to(lhs_row, &normalized, 0)
        } else {
            self.inner.equal_to(lhs_row, array, rhs_row)
        }
    }

    fn append_val(&mut self, array: &ArrayRef, row: usize) -> Result<()> {
        if let Some(normalized) = Self::normalized_row(array, row) {
            self.inner.append_val(&normalized, 0)
        } else {
            self.inner.append_val(array, row)
        }
    }

    fn vectorized_equal_to(
        &mut self,
        lhs_rows: &[usize],
        array: &ArrayRef,
        rhs_rows: &[usize],
        equal_to_results: &mut BooleanBufferBuilder,
    ) {
        let normalized = Self::normalized(array);
        self.inner.vectorized_equal_to(
            lhs_rows,
            normalized.as_ref().unwrap_or(array),
            rhs_rows,
            equal_to_results,
        );
    }

    fn vectorized_append(&mut self, array: &ArrayRef, rows: &[usize]) -> Result<()> {
        let normalized = Self::normalized(array);
        self.inner
            .vectorized_append(normalized.as_ref().unwrap_or(array), rows)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn size(&self) -> usize {
        size_of::<Self>() + self.inner.size()
    }

    fn build(self: Box<Self>) -> ArrayRef {
        self.inner.build()
    }

    fn values_preserving(&self, selection: GroupSelection<'_>) -> Result<ArrayRef> {
        self.inner.values_preserving(selection)
    }

    fn take_n(&mut self, n: usize) -> ArrayRef {
        self.inner.take_n(n)
    }
}
