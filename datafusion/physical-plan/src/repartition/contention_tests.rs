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
use arrow::array::Int64Array;
use arrow::datatypes::Field;

// Four concurrent producers, one shared input schema, 384 outputs each.
// Holding all outputs makes schema ownership observable without timing or PMU.
#[test]
fn grouped_take_schema_counter_is_private_to_producer() -> Result<()> {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![Field::new("value", DataType::Int64, false)],
        [("source".to_owned(), "rx-counter".to_owned())].into(),
    ));
    let rows = 8192;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
    )?;
    let barrier = std::sync::Barrier::new(4);
    let outputs = std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|_| {
                let batch = &batch;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let mut p = BatchPartitioner::new_hash_partitioner(
                        vec![Arc::new(
                            datafusion_physical_expr::expressions::Column::new(
                                "value", 0,
                            ),
                        )],
                        384,
                        metrics::Time::new(),
                    )?;
                    p.partition_iter(batch.clone())?.collect::<Result<Vec<_>>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Result<Vec<_>>>()
    })?;
    let shared = outputs
        .iter()
        .flatten()
        .filter(|(_, b)| Arc::ptr_eq(b.schema_ref(), &schema))
        .count();
    eprintln!(
        "RX producers=4 partitions=384 rows_per_input={rows} outputs={} shared_input_schema_outputs={shared} input_schema_strong_count={}",
        outputs.iter().map(Vec::len).sum::<usize>(),
        Arc::strong_count(&schema)
    );
    assert_eq!(
        shared, 0,
        "tiny output slices must not update the shared input schema counter"
    );
    for output in &outputs {
        for (_, b) in output {
            assert_eq!(b.schema_ref().as_ref(), schema.as_ref());
            assert!(!Arc::ptr_eq(
                &b.schema_ref().fields()[0],
                &schema.fields()[0]
            ));
        }
        assert_eq!(
            output.iter().map(|(_, b)| b.num_rows()).sum::<usize>(),
            rows
        );
        let mut values = output
            .iter()
            .flat_map(|(_, b)| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        assert_eq!(values, (0..rows as i64).collect::<Vec<_>>());
    }
    for (left, output) in outputs.iter().enumerate() {
        for other in outputs.iter().skip(left + 1) {
            assert!(!Arc::ptr_eq(
                output[0].1.schema_ref(),
                other[0].1.schema_ref()
            ));
        }
    }
    Ok(())
}
