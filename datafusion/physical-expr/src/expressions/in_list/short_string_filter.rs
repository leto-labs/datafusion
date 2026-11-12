// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, AsArray, BooleanArray};
use datafusion_common::{Result, exec_err};

use super::result::build_in_list_result;
use super::static_filter::{StaticFilter, handle_dictionary};
use crate::expressions::short_string::inline_view;

/// Two inline constants avoid hashing. Longer lists or strings keep
/// the general hash strategy. Dictionary inputs are tested once per value.
#[derive(Debug)]
pub(super) struct ShortStringFilter {
    constants: [u128; 2],
    evaluate: fn(&Self, &dyn Array, bool) -> Result<BooleanArray>,
    null_count: usize,
}

impl ShortStringFilter {
    pub(super) fn try_new(array: &dyn Array) -> Option<Self> {
        let array = array.as_string_view_opt()?;
        if array.len() - array.null_count() > 2 {
            return None;
        }
        let mut constants = [0; 2];
        let mut count = 0;
        for i in 0..array.len() {
            if !array.is_null(i) {
                constants[count] = inline_view(array.value(i))?;
                count += 1;
            }
        }
        let evaluate = match count {
            0 => Self::evaluate::<0>,
            1 => Self::evaluate::<1>,
            2 => Self::evaluate::<2>,
            _ => unreachable!(),
        };
        Some(Self {
            constants,
            evaluate,
            null_count: array.null_count(),
        })
    }

    fn evaluate<const N: usize>(
        &self,
        v: &dyn Array,
        negated: bool,
    ) -> Result<BooleanArray> {
        let contains = |word| {
            self.constants[..N]
                .iter()
                .fold(false, |found, &v| found | (word == v))
        };
        // Use separate monomorphized loops; avoid a string-type branch per row.
        if let Some(array) = v.as_string_view_opt() {
            let views = array.views();
            return Ok(build_in_list_result(
                v.len(),
                v.nulls(),
                self.null_count > 0,
                negated,
                |i| contains(views[i]),
            ));
        }
        exec_err!(
            "ShortStringFilter: expected a string-view array, got {}",
            v.data_type()
        )
    }
}

impl StaticFilter for ShortStringFilter {
    fn null_count(&self) -> usize {
        self.null_count
    }

    fn contains(&self, v: &dyn Array, negated: bool) -> Result<BooleanArray> {
        if v.data_type().is_null() {
            return Ok(BooleanArray::new_null(v.len()));
        }
        handle_dictionary!(self, v, negated);
        (self.evaluate)(self, v, negated)
    }
}

#[cfg(test)]
mod tests {
    use super::super::array_static_filter::ArrayStaticFilter;
    use super::super::strategy::instantiate_static_filter;
    use super::*;
    use arrow::array::*;
    use arrow::compute::cast;
    use arrow::datatypes::DataType;
    use std::sync::Arc;

    #[test]
    fn short_strings_match_generic_filter() -> Result<()> {
        let strings = [
            Some(""),
            Some("MAIL"),
            Some("SHIP"),
            Some("é"),
            Some("a\0b"),
            Some("abcdefghijkl"),
            Some("abcdefghijklm"),
            Some("MAIL-longer-than-inline"),
            Some("😀😀😀"),
            None,
        ];
        let needles: ArrayRef = Arc::new(StringArray::from(strings.to_vec()));
        for list in [
            vec![],
            vec![None],
            vec![Some("MAIL"), Some("SHIP")],
            vec![Some(""), None, Some("é")],
            vec![Some("MAIL"), Some("MAIL"), None],
            vec![Some("abcdefghijkl"), Some("a\0b")],
            vec![Some("😀😀😀")],
        ] {
            for data_type in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
                let haystack = cast(&StringArray::from(list.clone()), &data_type)?;
                assert_eq!(
                    ShortStringFilter::try_new(haystack.as_ref()).is_some(),
                    data_type == DataType::Utf8View
                );
                let filter = instantiate_static_filter(Arc::clone(&haystack))?;
                let reference = ArrayStaticFilter::try_new(Arc::clone(&haystack))?;
                let values = cast(needles.as_ref(), &data_type)?;
                let keys = Int32Array::from(
                    (0..values.len())
                        .map(|i| {
                            // Keep both a NULL key and a key pointing at a NULL value.
                            (i != 1).then_some(i as i32)
                        })
                        .collect::<Vec<_>>(),
                );
                let dictionary: ArrayRef = Arc::new(DictionaryArray::<
                    arrow::datatypes::Int32Type,
                >::try_new(
                    keys, Arc::clone(&values)
                )?);
                for array in [
                    Arc::clone(&values),
                    values.slice(1, values.len() - 1),
                    values.slice(0, 0),
                    dictionary,
                ] {
                    for negated in [false, true] {
                        assert_eq!(
                            filter.contains(array.as_ref(), negated)?,
                            reference.contains(array.as_ref(), negated)?
                        );
                    }
                }
            }
        }
        assert!(
            ShortStringFilter::try_new(&StringViewArray::from(vec![
                "too-long-for-inline"
            ]))
            .is_none()
        );
        assert!(
            ShortStringFilter::try_new(&StringViewArray::from(vec!["a", "b", "c"]))
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn short_string_filter_seeded_properties() -> Result<()> {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(0xf251_2026_1111);
        for count in 0..=2 {
            for _ in 0..100 {
                let list = (0..count)
                    .map(|_| {
                        let len = rng.random_range(0..=12);
                        Some(
                            (0..len)
                                .map(|_| char::from(rng.random_range(0_u8..=127)))
                                .collect::<String>(),
                        )
                    })
                    .chain(std::iter::once(None))
                    .collect::<Vec<_>>();
                let needles = (0..100)
                    .map(|_| {
                        if rng.random_bool(0.5) {
                            list[rng.random_range(0..list.len())].clone()
                        } else {
                            let len = rng.random_range(0..=20);
                            Some(
                                (0..len)
                                    .map(|_| char::from(rng.random_range(0_u8..=127)))
                                    .collect::<String>(),
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                let haystack: ArrayRef = Arc::new(StringViewArray::from_iter(list));
                let values = StringViewArray::from_iter(needles);
                let filter = ShortStringFilter::try_new(haystack.as_ref()).unwrap();
                let reference = ArrayStaticFilter::try_new(haystack)?;
                for negated in [false, true] {
                    assert_eq!(
                        filter.contains(&values, negated)?,
                        reference.contains(&values, negated)?
                    );
                }
            }
        }
        Ok(())
    }

    /// Run explicitly, pinned to one CPU. Both implementations use the same
    /// process, input view array, allocator, and untimed predicate preparation.
    #[test]
    #[ignore = "diagnostic ABBA kernel screen; emits timings"]
    fn short_string_in_kernel_screen() -> Result<()> {
        use datafusion_common::instant::Instant;
        use std::hint::black_box;

        let modes = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"];
        let values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..8192).map(|i| modes[i % modes.len()]),
        ));
        let list: ArrayRef = Arc::new(StringViewArray::from(vec!["MAIL", "SHIP"]));
        let containers = ["SM", "MED", "LG", "JUMBO", "WRAP"]
            .iter()
            .flat_map(|prefix| {
                ["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"]
                    .map(move |kind| format!("{prefix} {kind}"))
            })
            .collect::<Vec<_>>();
        let container_values: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..8192).map(|i| containers[i % containers.len()].as_str()),
        ));
        let container_list: ArrayRef =
            Arc::new(StringViewArray::from(vec!["SM CASE", "SM BOX"]));
        for (scenario, values, list) in [
            ("shipmode", values, list),
            ("container-pair", container_values, container_list),
        ] {
            for data_type in [DataType::Utf8View, DataType::Utf8, DataType::LargeUtf8] {
                let values = cast(values.as_ref(), &data_type)?;
                let list = cast(list.as_ref(), &data_type)?;
                let reference = ArrayStaticFilter::try_new(Arc::clone(&list))?;
                let candidate = instantiate_static_filter(Arc::clone(&list))?;
                assert_eq!(
                    reference.contains(values.as_ref(), false)?,
                    candidate.contains(values.as_ref(), false)?
                );
                let a = &reference as &dyn StaticFilter;
                let b: &dyn StaticFilter = candidate.as_ref();
                let batches = 2000;
                for (block, (arm, filter)) in [
                    ("generic", a),
                    ("short", b),
                    ("short", b),
                    ("generic", a),
                    ("generic", a),
                    ("short", b),
                    ("short", b),
                    ("generic", a),
                ]
                .into_iter()
                .enumerate()
                {
                    black_box(filter.contains(values.as_ref(), false)?);
                    let start = Instant::now();
                    for _ in 0..batches {
                        black_box(
                            black_box(filter)
                                .contains(black_box(values.as_ref()), false)?,
                        );
                    }
                    let elapsed = start.elapsed();
                    eprintln!(
                        "short_string_in scenario={scenario}-{data_type} block={block} arm={arm} rows={} batches={batches} elapsed_ns={} ns_per_row={:.4}",
                        values.len(),
                        elapsed.as_nanos(),
                        elapsed.as_nanos() as f64 / (values.len() * batches) as f64
                    );
                }
            }
        }
        Ok(())
    }
}
