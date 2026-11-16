// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::*;
use crate::aggregates::order::GroupClusteringMode;
use arrow::array::{
    Array, BooleanArray, Decimal32Array, Decimal64Array, Int32Array, Int64Array,
    StringViewArray,
};
use arrow::datatypes::{Field, Schema};
use datafusion_common::ScalarValue;
use std::sync::Arc;

fn schema(cols: &[ArrayRef]) -> SchemaRef {
    Arc::new(Schema::new(
        cols.iter()
            .enumerate()
            .map(|(i, a)| Field::new(i.to_string(), a.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ))
}

fn check(mut interner: Box<dyn GroupValues>, cols: &[ArrayRef]) -> Result<()> {
    assert!(interner.supports_values_preserving());
    assert!(
        interner
            .values_preserving(GroupSelection::all(0))?
            .iter()
            .all(|a| a.is_empty())
    );
    let mut groups = vec![];
    interner.intern(cols, &mut groups)?;
    // Populate front caches before reading without removing groups.
    interner.intern(cols, &mut groups)?;
    let selection = [interner.len() - 1, 0, interner.len() - 1];
    for _ in 0..2 {
        let output = interner.values_preserving(GroupSelection::try_from_indices(
            &selection,
            interner.len(),
        )?)?;
        for (col, out) in cols.iter().zip(&output) {
            for (row, &group) in selection.iter().enumerate() {
                let input_row = groups.iter().position(|&id| id == group).unwrap();
                assert_eq!(
                    ScalarValue::try_from_array(out, row)?,
                    ScalarValue::try_from_array(col, input_row)?
                );
            }
        }
    }
    assert!(
        interner
            .values_preserving(GroupSelection::try_from_indices(&[], interner.len())?)?
            .iter()
            .all(|a| a.is_empty())
    );
    assert!(
        interner
            .values_preserving(GroupSelection::all(interner.len() + 1))
            .is_err()
    );
    let before = interner.size();
    interner.shrink_to_fit();
    assert!(interner.size() <= before);
    let mut again = vec![];
    interner.intern(cols, &mut again)?;
    assert_eq!(groups, again);
    let all = interner.values_preserving(GroupSelection::all(interner.len()))?;
    let first = interner.emit(EmitTo::First(1))?;
    for (expected, actual) in all.iter().zip(&first) {
        assert_eq!(expected.slice(0, 1).as_ref(), actual.as_ref());
    }
    interner.shrink_to_fit();
    let remaining = interner.values_preserving(GroupSelection::all(interner.len()))?;
    assert_eq!(remaining, interner.emit(EmitTo::All)?);
    interner.clear_shrink(0);
    assert!(
        interner
            .values_preserving(GroupSelection::all(0))?
            .iter()
            .all(|a| a.is_empty())
    );
    Ok(())
}

#[test]
fn fork_interners_keep_values_and_ids_across_preserving_reads_and_shrink() -> Result<()> {
    let ints = vec![
        Arc::new(Int32Array::from(vec![Some(2), None, Some(7), Some(2)])) as ArrayRef,
    ];
    check(
        new_group_values(schema(&ints), &GroupClustering::None)?,
        &ints,
    )?;
    let views = vec![Arc::new(StringViewArray::from(vec![
        Some("ab"),
        None,
        Some("cd"),
        Some("ab"),
    ])) as ArrayRef];
    check(
        new_group_values(schema(&views), &GroupClustering::None)?,
        &views,
    )?;
    check(
        Box::new(batch_codes::BatchCodes::new(Box::new(
            GroupValuesBytesView::new(OutputType::Utf8View),
        ))),
        &views,
    )?;
    let pair = vec![Arc::clone(&views[0]), Arc::clone(&views[0])];
    check(
        Box::new(inline_cache::InlineCache::<2>::new(Box::new(
            GroupValuesColumn::<false>::try_new(schema(&pair))?,
        ))),
        &pair,
    )?;
    check(
        new_group_values(schema(&pair), &GroupClustering::None)?,
        &pair,
    )?;
    let fallback = vec![
        Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX, i64::MIN])) as ArrayRef,
        Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX, i64::MIN])) as ArrayRef,
    ];
    let mut packed =
        packed::GroupValuesPacked::try_new(schema(&fallback), unordered_multi_column)?;
    packed.intern(&fallback, &mut vec![])?;
    assert!(packed.is_fallback());
    // check starts with an empty interner and drives the transition itself.
    check(
        new_group_values(schema(&fallback), &GroupClustering::None)?,
        &fallback,
    )?;
    Ok(())
}

#[test]
fn packed_narrow_decimals_preserve_types_nulls_and_relayout() -> Result<()> {
    let cols = vec![
        Arc::new(
            Decimal32Array::from(vec![Some(-17), None, Some(26), Some(-17)])
                .with_precision_and_scale(5, 2)?,
        ) as ArrayRef,
        Arc::new(
            Decimal64Array::from(vec![Some(121), Some(-22), None, Some(121)])
                .with_precision_and_scale(14, 3)?,
        ) as ArrayRef,
    ];
    assert!(packed::supported(&schema(&cols)));
    check(
        new_group_values(schema(&cols), &GroupClustering::None)?,
        &cols,
    )
}

#[test]
fn ordered_factory_keeps_upstream_clustered_storage() -> Result<()> {
    let cols = vec![Arc::new(BooleanArray::from(vec![false, true])) as ArrayRef; 2];
    let schema = schema(&cols);
    let clustering = GroupClustering::try_new(&GroupClusteringMode::Full)?;
    let mut actual = new_group_values(Arc::clone(&schema), &clustering)?;
    let mut expected = GroupValuesClustered::try_new(schema)?;
    assert_eq!(size_of_val(actual.as_ref()), size_of_val(&expected));
    assert_eq!(actual.size(), expected.size());
    let (mut actual_ids, mut expected_ids) = (vec![], vec![]);
    actual.intern(&cols, &mut actual_ids)?;
    expected.intern(&cols, &mut expected_ids)?;
    assert_eq!(actual_ids, expected_ids);
    assert_eq!(actual.size(), expected.size());
    assert_eq!(actual.emit(EmitTo::All)?, expected.emit(EmitTo::All)?);
    Ok(())
}

/// The result changes only when the wrapper forwards shrink_to_fit.
struct ForwardingProbe(bool, bool);
impl GroupValues for ForwardingProbe {
    fn intern(&mut self, _cols: &[ArrayRef], _groups: &mut Vec<usize>) -> Result<()> {
        Ok(())
    }
    fn size(&self) -> usize {
        0
    }
    fn is_empty(&self) -> bool {
        false
    }
    fn len(&self) -> usize {
        1
    }
    fn emit(&mut self, _emit: EmitTo) -> Result<Vec<ArrayRef>> {
        self.values_preserving(GroupSelection::all(1))
    }
    fn clear_shrink(&mut self, _rows: usize) {}
    fn values_preserving(
        &mut self,
        _selection: GroupSelection<'_>,
    ) -> Result<Vec<ArrayRef>> {
        Ok(vec![Arc::new(BooleanArray::from(vec![self.0]))])
    }
    fn supports_values_preserving(&self) -> bool {
        self.1
    }
    fn shrink_to_fit(&mut self) {
        self.0 = true;
    }
}

#[test]
fn every_cache_wrapper_forwards_the_three_new_methods() -> Result<()> {
    let wrappers: Vec<Box<dyn GroupValues>> = vec![
        Box::new(inline_cache::InlineCache::<1>::new(Box::new(
            ForwardingProbe(false, true),
        ))),
        Box::new(inline_cache::InlineCache::<2>::new(Box::new(
            ForwardingProbe(false, true),
        ))),
        Box::new(batch_codes::BatchCodes::new(Box::new(ForwardingProbe(
            false, true,
        )))),
    ];
    for mut wrapper in wrappers {
        assert!(wrapper.supports_values_preserving());
        wrapper.shrink_to_fit();
        assert_eq!(
            wrapper.values_preserving(GroupSelection::all(1))?[0].as_ref(),
            &BooleanArray::from(vec![true])
        );
    }
    let unsupported: Vec<Box<dyn GroupValues>> = vec![
        Box::new(inline_cache::InlineCache::<1>::new(Box::new(
            ForwardingProbe(false, false),
        ))),
        Box::new(inline_cache::InlineCache::<2>::new(Box::new(
            ForwardingProbe(false, false),
        ))),
        Box::new(batch_codes::BatchCodes::new(Box::new(ForwardingProbe(
            false, false,
        )))),
    ];
    for wrapper in unsupported {
        assert!(!wrapper.supports_values_preserving());
    }
    Ok(())
}

#[test]
fn fixed_layouts_above_64_bits_use_general_grouping() -> Result<()> {
    let cols =
        vec![Arc::new(BooleanArray::from(vec![false, true, false])) as ArrayRef; 33];
    let schema = schema(&cols);
    assert!(!packed::supported(&schema));
    assert!(
        packed::GroupValuesPacked::try_new(Arc::clone(&schema), unordered_multi_column)
            .is_err()
    );
    check(new_group_values(schema, &GroupClustering::None)?, &cols)
}
