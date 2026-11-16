// Modified by the Longbow authors; see third_party/README.md.

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the
// License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use arrow::array::{Array, Int64Array};
use arrow::datatypes::Field;

#[test]
fn output_coalescers_do_not_share_the_plan_schema_counter() -> Result<()> {
    let batch = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from_iter_values(0..32)) as Arc<dyn Array>,
    )])?;
    let schema = batch.schema();
    let baseline = Arc::strong_count(&schema);
    let mut outputs = Vec::new();
    for _ in 0..4 {
        let c = SharedCoalescer::new(&schema, 32, 1);
        let mut ready = Vec::new();
        c.push_range_and_drain(&batch, 3, 5, &mut ready)?;
        assert!(ready.is_empty());
        outputs.extend(c.finalize()?);
    }
    assert_eq!(Arc::strong_count(&schema), baseline);
    assert_eq!(outputs.len(), 4);
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(output, &batch.slice(3, 5));
        assert!(!Arc::ptr_eq(output.schema_ref(), &schema));
        for other in &outputs[..i] {
            assert!(!Arc::ptr_eq(output.schema_ref(), other.schema_ref()));
        }
    }
    Ok(())
}

#[test]
fn private_schema_reuses_equal_values_and_refreshes_metadata() -> Result<()> {
    let mut p = BatchPartitioner::new_hash_partitioner(
        vec![Arc::new(
            datafusion_physical_expr::expressions::Column::new("x", 0),
        )],
        2,
        metrics::Time::new(),
    )?;
    let make_batch = |metadata: &str| {
        RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(
                vec![Field::new("x", DataType::Int64, false)],
                [("generation".to_owned(), metadata.to_owned())],
            )),
            vec![Arc::new(Int64Array::from_iter_values(0..32))],
        )
    };
    let first = p.partition_iter(make_batch("first")?)?.next().unwrap()?.1;
    let equal = p.partition_iter(make_batch("first")?)?.next().unwrap()?.1;
    assert!(Arc::ptr_eq(first.schema_ref(), equal.schema_ref()));
    let changed = make_batch("second")?;
    let expected = changed.schema();
    let second = p.partition_iter(changed)?.next().unwrap()?.1;
    assert_eq!(second.schema_ref(), &expected);
    assert!(!Arc::ptr_eq(first.schema_ref(), second.schema_ref()));
    Ok(())
}
