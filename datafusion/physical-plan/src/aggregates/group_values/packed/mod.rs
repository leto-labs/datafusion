// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Multi-column group keys packed into one `u64`.
//!
//! Every column maps its values to dense codes (code 0 is NULL): integers by
//! their offset from an observed minimum, booleans directly, and byte views by
//! a per-column dictionary. The codes of all columns, concatenated, are the
//! complete group identity: [`PackedKeyTable`] interns them with one hash and
//! one compare, or with a direct index when the key has at most
//! [`table::DIRECT_BITS`] bits.
//!
//! A value outside the current layout widens its column. Existing keys are
//! repacked under the new layout; group ids do not change. When the columns
//! need more than 64 bits, the groups move, in order, into the general
//! interner, which keeps their ids and takes over every later batch.

mod bytes;
mod columns;
#[cfg(test)]
mod single_tests;
pub mod table;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{
    BinaryViewType, DataType, Date32Type, Date64Type, Decimal32Type, Decimal64Type,
    Decimal128Type, DurationMicrosecondType, DurationMillisecondType,
    DurationNanosecondType, DurationSecondType, Int8Type, Int16Type, Int32Type,
    Int64Type, SchemaRef, StringViewType, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
    UInt16Type, UInt32Type, UInt64Type,
};
use datafusion_common::{DataFusionError, Result, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::{EmitTo, GroupSelection};

use super::GroupValues;
use bytes::BytesColumn;
use columns::{BoolColumn, IntColumn, KeyColumn, Observed, Proposal};
pub use table::PackedKeyTable;

const KEY_BITS: u32 = 64;
/// A column with more distinct values than this is not low-cardinality: its
/// codes would take most of the key, and the general interner does the same
/// per-value work without the dictionary. A dictionary whose views repeat (a
/// Parquet dictionary page) never looks costly, so this bound also limits
/// the values learned before the groups move (ClickBench q35's URLs).
const MAX_DICTIONARY: usize = 1 << 16;

/// The constructor of the general interner for the same schema.
pub(super) type Fallback = fn(SchemaRef) -> Result<Box<dyn GroupValues>>;

/// Whether every column of `schema` has packed codes.
pub(super) fn supported(schema: &SchemaRef) -> bool {
    schema
        .fields()
        .iter()
        .try_fold(0_u32, |bits, field| {
            bits.checked_add(make_column(field.data_type())?.width())
        })
        .is_some_and(|bits| bits <= KEY_BITS)
}

/// Whether `schema` is one integer-coded column (integers, dates, times,
/// timestamps, durations and decimals). Its codes are offsets from the
/// observed minimum: a small domain indexes a direct table, a domain below
/// 2^32 uses 8-byte hash slots, and wider ones use the general interner.
pub(super) fn single_column(schema: &SchemaRef) -> bool {
    match schema.fields().iter().collect::<Vec<_>>().as_slice() {
        [field] => {
            !matches!(
                field.data_type(),
                DataType::Boolean | DataType::Utf8View | DataType::BinaryView
            ) && make_column(field.data_type()).is_some()
        }
        _ => false,
    }
}

fn make_column(data_type: &DataType) -> Option<Box<dyn KeyColumn>> {
    macro_rules! int {
        ($t:ty) => {
            Some(Box::new(IntColumn::<$t>::new(data_type.clone())))
        };
    }
    match data_type {
        DataType::Int8 => int!(Int8Type),
        DataType::Int16 => int!(Int16Type),
        DataType::Int32 => int!(Int32Type),
        DataType::Int64 => int!(Int64Type),
        DataType::UInt8 => int!(UInt8Type),
        DataType::UInt16 => int!(UInt16Type),
        DataType::UInt32 => int!(UInt32Type),
        DataType::UInt64 => int!(UInt64Type),
        DataType::Date32 => int!(Date32Type),
        DataType::Date64 => int!(Date64Type),
        DataType::Time32(TimeUnit::Second) => int!(Time32SecondType),
        DataType::Time32(TimeUnit::Millisecond) => int!(Time32MillisecondType),
        DataType::Time64(TimeUnit::Microsecond) => int!(Time64MicrosecondType),
        DataType::Time64(TimeUnit::Nanosecond) => int!(Time64NanosecondType),
        DataType::Timestamp(TimeUnit::Second, _) => int!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => int!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => int!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => int!(TimestampNanosecondType),
        DataType::Duration(TimeUnit::Second) => int!(DurationSecondType),
        DataType::Duration(TimeUnit::Millisecond) => int!(DurationMillisecondType),
        DataType::Duration(TimeUnit::Microsecond) => int!(DurationMicrosecondType),
        DataType::Duration(TimeUnit::Nanosecond) => int!(DurationNanosecondType),
        DataType::Decimal32(_, _) => int!(Decimal32Type),
        DataType::Decimal64(_, _) => int!(Decimal64Type),
        DataType::Decimal128(_, _) => int!(Decimal128Type),
        DataType::Boolean => Some(Box::new(BoolColumn)),
        DataType::Utf8View => Some(Box::new(BytesColumn::<StringViewType>::new())),
        DataType::BinaryView => Some(Box::new(BytesColumn::<BinaryViewType>::new())),
        _ => None,
    }
}

const SINGLE_KEY_MIGRATION: &str = "packed single-key migration";

/// A refused migration grant leaves the packed groups and incoming batch untouched.
pub fn migration_needs_memory(error: &DataFusionError) -> bool {
    matches!(
        error,
        DataFusionError::Context(context, cause)
            if context == SINGLE_KEY_MIGRATION
                && matches!(cause.find_root(), DataFusionError::ResourcesExhausted(_))
    )
}

enum PackBuffer {
    Single(Vec<u32>),
    Multi(Vec<u64>),
}

impl PackBuffer {
    fn clear_shrink(&mut self, rows: usize) {
        match self {
            Self::Single(keys) => {
                keys.clear();
                keys.shrink_to(rows);
            }
            Self::Multi(keys) => {
                keys.clear();
                keys.shrink_to(rows);
            }
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Single(keys) => keys.capacity() * size_of::<u32>(),
            Self::Multi(keys) => keys.capacity() * size_of::<u64>(),
        }
    }
}

/// [`GroupValues`] for keys whose columns' codes fit in 64 bits.
pub struct GroupValuesPacked {
    schema: SchemaRef,
    columns: Vec<Box<dyn KeyColumn>>,
    /// The layout of the stored keys: each column's width and shift. A
    /// column's own `width` can run ahead of it until the next relayout.
    widths: Vec<u32>,
    shifts: Vec<u32>,
    table: PackedKeyTable,
    keys: PackBuffer,
    fallback: Option<Box<dyn GroupValues>>,
    make_fallback: Fallback,
    /// The widest layout before the groups move to the general interner.
    /// One column stops at 8-byte slots: in 16-byte slots, the packed keys
    /// and codes cost more memory than the single-column interner's table
    /// and saved no time (`group_values_interning` bench, ClickBench UserID).
    max_bits: u32,
}

impl GroupValuesPacked {
    pub(super) fn try_new(schema: SchemaRef, make_fallback: Fallback) -> Result<Self> {
        let columns = schema
            .fields()
            .iter()
            .map(|f| make_column(f.data_type()))
            .collect::<Option<Vec<_>>>();
        let Some(columns) = columns else {
            return internal_err!("packed group keys do not support {schema}");
        };
        if columns.iter().map(|column| column.width()).sum::<u32>() > KEY_BITS {
            return internal_err!("packed group keys exceed {KEY_BITS} fixed bits");
        }
        let columns_len = columns.len();
        let single_integer = columns_len == 1
            && !matches!(
                schema.field(0).data_type(),
                DataType::Boolean | DataType::Utf8View | DataType::BinaryView
            );
        let mut packed = Self {
            widths: vec![0; columns.len()],
            shifts: vec![0; columns.len()],
            schema,
            columns,
            table: PackedKeyTable::new(),
            keys: if single_integer {
                PackBuffer::Single(Vec::new())
            } else {
                PackBuffer::Multi(Vec::new())
            },
            fallback: None,
            make_fallback,
            max_bits: if columns_len == 1 {
                table::NARROW_BITS
            } else {
                KEY_BITS
            },
        };
        packed.initial_layout();
        Ok(packed)
    }

    /// The layout of columns without values: fixed widths (booleans) only.
    fn initial_layout(&mut self) {
        let shift = self.set_layout();
        self.table.relayout(shift, 0, |key| key);
    }

    /// Record the columns' widths as the stored layout; returns its bits.
    fn set_layout(&mut self) -> u32 {
        let mut shift = 0;
        for ((column, w), s) in self
            .columns
            .iter()
            .zip(self.widths.iter_mut())
            .zip(self.shifts.iter_mut())
        {
            *w = column.width();
            *s = shift;
            shift += *w;
        }
        shift
    }

    /// The codes of column `index` in the stored keys of the first `n` groups.
    fn stored_codes(&self, index: usize, n: usize) -> impl Iterator<Item = u64> + '_ {
        let (width, shift) = (self.widths[index], self.shifts[index]);
        self.table.keys()[..n].iter().map(move |&key| {
            if width == 0 {
                0
            } else {
                (key >> shift) & (u64::MAX >> (64 - width))
            }
        })
    }

    /// Whether the general interner has taken over.
    #[cfg(test)]
    pub(super) fn is_fallback(&self) -> bool {
        self.fallback.is_some()
    }

    #[cfg(test)]
    pub(super) fn key_bits(&self) -> u32 {
        self.table.bits()
    }

    #[cfg(test)]
    pub(super) fn is_direct(&self) -> bool {
        self.table.is_direct()
    }

    /// Widen the layout to cover every observed value. Returns false when no
    /// layout fits in `max_bits`.
    fn relayout(&mut self, rows: usize) -> bool {
        let choose = |spare: bool, all: bool| -> (Vec<Option<Proposal>>, u32) {
            let proposals: Vec<_> = self
                .columns
                .iter()
                .map(|c| (all || !c.fits()).then(|| c.propose(spare)))
                .collect();
            let total = proposals
                .iter()
                .zip(&self.columns)
                .map(|(p, c)| p.map_or(c.width(), |p| p.width))
                .sum();
            (proposals, total)
        };
        let spare = choose(true, false);
        let exact = choose(false, false);
        // Prefer spare room, unless exact widths reach a cheaper table: a
        // direct index (16 bits), then 8-byte hash slots (32 bits).
        let mut chosen = None;
        for limit in [table::DIRECT_BITS, table::NARROW_BITS, KEY_BITS]
            .into_iter()
            .filter(|&limit| limit <= self.max_bits)
        {
            if spare.1 <= limit {
                chosen = Some(spare.clone());
            } else if exact.1 <= limit {
                chosen = Some(exact.clone());
            }
            if chosen.is_some() {
                break;
            }
        }
        let (proposals, total) = match chosen {
            Some(chosen) => chosen,
            None => {
                let all = choose(false, true);
                if all.1 > self.max_bits {
                    return false;
                }
                all
            }
        };
        for (column, proposal) in self.columns.iter_mut().zip(proposals) {
            if let Some(proposal) = proposal {
                column.adopt(proposal);
            }
        }
        let old: Vec<(u32, u32)> = self
            .widths
            .iter()
            .copied()
            .zip(self.shifts.clone())
            .collect();
        let bits = self.set_layout();
        debug_assert_eq!(bits, total);
        let expected = if matches!(&self.keys, PackBuffer::Single(_)) {
            // A big, duplicate-heavy batch must not buy a large group index.
            self.table.len()
        } else {
            self.table.len() + rows
        };
        let (columns, new) = (&self.columns, &self.shifts);
        self.table.relayout(bits, expected, |key| {
            let mut out = 0;
            for ((column, &(width, old_shift)), &new_shift) in
                columns.iter().zip(&old).zip(new)
            {
                if width == 0 {
                    continue;
                }
                let code = (key >> old_shift) & (u64::MAX >> (64 - width));
                if code != 0 {
                    out |= column.translate(code) << new_shift;
                }
            }
            out
        });
        for column in self.columns.iter_mut() {
            column.settle();
        }
        true
    }

    /// Move every group, in order, into the general interner.
    fn fall_back(
        &mut self,
        rows: usize,
        reservation: Option<&MemoryReservation>,
    ) -> Result<&mut Box<dyn GroupValues>> {
        if matches!(&self.keys, PackBuffer::Single(_)) {
            let prior_size = self.size();
            let keys = self.table.keys();
            let column = &mut self.columns[0];
            // Each incoming row can contribute at most one group. Give the
            // primitive table room for the whole triggering batch before releasing
            // the grant's excess, so its first insertions cannot grow the table.
            let max_groups = keys.len().saturating_add(rows);
            let bytes = column.single_migration_bytes(keys.len(), max_groups);
            if let Some(owner) = reservation {
                // The fair pool admits a spillable reservation by its own
                // total, so charge the overlap directly on the table owner.
                owner
                    .try_grow(bytes)
                    .map_err(|error| error.context(SINGLE_KEY_MIGRATION))?;
            }

            let inner = match column.fall_back_single(keys, max_groups) {
                Ok(inner) => inner,
                Err(error) => {
                    if let Some(owner) = reservation {
                        owner.shrink(bytes);
                    }
                    return Err(error);
                }
            };
            self.finish_fallback(inner);
            if let Some(owner) = reservation {
                // Keep the live replacement charged. Release the estimate's
                // excess and freed packed buffers together, in one shrink.
                let released = (prior_size + bytes)
                    .checked_sub(self.size())
                    .expect("single-key migration estimate covers the replacement");
                owner.shrink(released);
            }
            return Ok(self.fallback.as_mut().unwrap());
        }
        let groups = self.table.len();
        let arrays = self.build(groups)?;
        let mut inner = (self.make_fallback)(Arc::clone(&self.schema))?;
        if groups > 0 {
            let mut ids = Vec::with_capacity(groups);
            inner.intern(&arrays, &mut ids)?;
            if ids.iter().enumerate().any(|(i, &id)| i != id) {
                // The general interner can number a batch's new groups out of
                // row order (hash-colliding rows take a later pass). One row
                // per call forces first-seen order, which the ids require.
                inner = (self.make_fallback)(Arc::clone(&self.schema))?;
                for row in 0..groups {
                    let slice: Vec<ArrayRef> =
                        arrays.iter().map(|a| a.slice(row, 1)).collect();
                    inner.intern(&slice, &mut ids)?;
                    if ids[..] != [row] {
                        return internal_err!(
                            "general interner reassigned packed group ids"
                        );
                    }
                }
            }
        }
        Ok(self.finish_fallback(inner))
    }

    fn finish_fallback(
        &mut self,
        inner: Box<dyn GroupValues>,
    ) -> &mut Box<dyn GroupValues> {
        self.table.clear_shrink(0);
        self.keys.clear_shrink(0);
        for column in &mut self.columns {
            column.reset();
        }
        self.fallback.insert(inner)
    }

    /// The key columns of the first `n` groups. A column with a pending
    /// conversion translates its stored codes first.
    fn build(&mut self, n: usize) -> Result<Vec<ArrayRef>> {
        let mut codes = Vec::with_capacity(n);
        let mut arrays = Vec::with_capacity(self.columns.len());
        for index in 0..self.columns.len() {
            codes.clear();
            codes.extend(self.stored_codes(index, n));
            let column = &mut self.columns[index];
            column.translate_all(&mut codes);
            arrays.push(column.build(&codes)?);
        }
        Ok(arrays)
    }

    fn intern_inner(
        &mut self,
        cols: &[ArrayRef],
        groups: &mut Vec<usize>,
        reservation: Option<&MemoryReservation>,
    ) -> Result<()> {
        if let Some(fallback) = &mut self.fallback {
            return fallback.intern(cols, groups);
        }
        let rows = cols.first().map_or(0, |c| c.len());
        // Group ids of the table are u32.
        if self.table.len() + rows >= u32::MAX as usize {
            return self.fall_back(rows, reservation)?.intern(cols, groups);
        }
        for (index, array) in cols.iter().enumerate().take(self.columns.len()) {
            let mut observed = self.columns[index].observe(array)?;
            if observed == Observed::Convert {
                let codes: Vec<u64> =
                    self.stored_codes(index, self.table.len()).collect();
                let column = &mut self.columns[index];
                column.convert(&codes);
                observed = column.observe(array)?;
            }
            if observed == Observed::Costly {
                return self.fall_back(rows, reservation)?.intern(cols, groups);
            }
        }
        let too_many = self
            .columns
            .iter()
            .any(|c| c.cardinality() > MAX_DICTIONARY || c.costly());
        if too_many || (!self.columns.iter().all(|c| c.fits()) && !self.relayout(rows)) {
            return self.fall_back(rows, reservation)?.intern(cols, groups);
        }
        match &mut self.keys {
            PackBuffer::Single(keys) => {
                let [array] = cols else {
                    return internal_err!("single-key interning requires one column");
                };
                keys.reserve_exact(rows.saturating_sub(keys.len()));
                keys.resize(rows, 0);
                self.columns[0].pack_single(array, keys);
                self.table.intern_narrow(keys, groups);
            }
            PackBuffer::Multi(keys) => {
                keys.clear();
                keys.resize(rows, 0);
                for ((column, array), &shift) in
                    self.columns.iter().zip(cols).zip(&self.shifts)
                {
                    column.pack(array, keys, shift);
                }
                self.table.intern(keys, groups);
            }
        }
        Ok(())
    }
}

impl GroupValues for GroupValuesPacked {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        self.intern_inner(cols, groups, None)
    }

    fn intern_reserved(
        &mut self,
        cols: &[ArrayRef],
        groups: &mut Vec<usize>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        self.intern_inner(cols, groups, Some(reservation))
    }

    fn size(&self) -> usize {
        size_of::<Self>()
            + self.table.size()
            + self.keys.size()
            + self.widths.capacity() * size_of::<u32>()
            + self.shifts.capacity() * size_of::<u32>()
            + self.columns.capacity() * size_of::<Box<dyn KeyColumn>>()
            + self
                .columns
                .iter()
                .map(|c| size_of_val(c.as_ref()) + c.size())
                .sum::<usize>()
            + self.fallback.as_ref().map_or(0, |f| f.size())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        match &self.fallback {
            Some(fallback) => fallback.len(),
            None => self.table.len(),
        }
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        if let Some(fallback) = &mut self.fallback {
            return fallback.emit(emit_to);
        }
        let n = match emit_to {
            EmitTo::All => self.table.len(),
            EmitTo::First(n) => n.min(self.table.len()),
        };
        let arrays = self.build(n)?;
        match emit_to {
            EmitTo::All => {
                // No group remains: forget dictionaries and counters, so they
                // describe only the groups that follow (no stale values, no
                // late fallback from a cumulative count).
                self.table.clear();
                for column in self.columns.iter_mut() {
                    column.reset();
                }
                self.initial_layout();
            }
            EmitTo::First(_) => {
                self.table.drop_first(n);
                for column in self.columns.iter_mut() {
                    column.release();
                }
            }
        }
        Ok(arrays)
    }

    fn values_preserving(
        &mut self,
        selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        if let Some(fallback) = &mut self.fallback {
            return fallback.values_preserving(selection);
        }
        selection.validate_num_groups(self.table.len())?;
        let mut codes = Vec::with_capacity(selection.len());
        let mut arrays = Vec::with_capacity(self.columns.len());
        for index in 0..self.columns.len() {
            let (width, shift) = (self.widths[index], self.shifts[index]);
            codes.clear();
            codes.extend(selection.iter().map(|group| {
                if width == 0 {
                    0
                } else {
                    (self.table.keys()[group] >> shift) & (u64::MAX >> (64 - width))
                }
            }));
            let column = &mut self.columns[index];
            column.translate_all(&mut codes);
            arrays.push(column.build(&codes)?);
        }
        Ok(arrays)
    }

    fn supports_values_preserving(&self) -> bool {
        self.fallback
            .as_ref()
            .is_none_or(|f| f.supports_values_preserving())
    }

    fn shrink_to_fit(&mut self) {
        if let Some(fallback) = &mut self.fallback {
            fallback.shrink_to_fit();
        }
        self.table.shrink_to_fit();
        self.keys.clear_shrink(0);
        for column in &mut self.columns {
            column.shrink_to_fit();
        }
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        if let Some(fallback) = &mut self.fallback {
            return fallback.clear_shrink(num_rows);
        }
        self.table.clear_shrink(num_rows);
        self.keys.clear_shrink(num_rows);
        for column in self.columns.iter_mut() {
            column.reset();
        }
        self.initial_layout();
    }
}
