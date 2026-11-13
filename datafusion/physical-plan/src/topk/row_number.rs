// Modified by the Longbow authors; see third_party/README.md.

//! Compact, bounded ROW_NUMBER for schemas consisting only of ranking keys.
//! Store encoded values rather than one source-batch reference per group, and
//! decode full output batches across group boundaries.
use super::*;
use rank_memory::{add, mul};
mod compact_row;
use compact_row::CompactRow;

struct Entry {
    order: CompactRow,
    payload: CompactRow,
}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order.cmp(&other.order)
    }
}

type Groups = HashMap<CompactRow, BinaryHeap<Entry>>;

pub(crate) struct RowNumberTopK {
    schema: SchemaRef,
    partition_exprs: Vec<Arc<dyn PhysicalExpr>>,
    orders: LexOrdering,
    partition_converter: RowConverter,
    order_converter: RowConverter,
    payload_converter: RowConverter,
    groups: Groups,
    // Encoded data and heap capacity, updated only for the changed group.
    row_bytes: usize,
    rows: usize,
    heap_bytes: usize,
    k: usize,
    batch_size: usize,
    metrics: TopKMetrics,
    // All owned buffers above are released before their pool credits.
    reservation: MemoryReservation,
}

impl RowNumberTopK {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        partition: usize,
        schema: SchemaRef,
        partition_exprs: Vec<Arc<dyn PhysicalExpr>>,
        partition_fields: Vec<SortField>,
        orders: LexOrdering,
        k: usize,
        batch_size: usize,
        runtime: &Arc<RuntimeEnv>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Self> {
        rank_memory::validate(&schema, &partition_exprs, &orders, k, batch_size)?;
        let reservation = MemoryConsumer::new(format!("RowNumberTopK[{partition}]"))
            .register(&runtime.memory_pool);
        reservation.try_grow(add(
            4096,
            mul(
                schema.fields().len() + orders.len() + partition_fields.len(),
                2048,
            )?,
        )?)?;
        let order_converter = RowConverter::new(build_sort_fields(&orders, &schema)?)?;
        let partition_converter = RowConverter::new(if partition_fields.is_empty() {
            vec![SortField::new(arrow::datatypes::DataType::Null)]
        } else {
            partition_fields
        })?;
        let payload_converter = RowConverter::new(
            schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;
        Ok(Self {
            schema,
            partition_exprs,
            orders,
            partition_converter,
            order_converter,
            payload_converter,
            groups: HashMap::new(),
            row_bytes: 0,
            rows: 0,
            heap_bytes: 0,
            k,
            batch_size,
            metrics: TopKMetrics::new(metrics, partition),
            reservation,
        })
    }

    fn retained(&self) -> Result<usize> {
        let table = mul(
            self.groups.capacity(),
            2 * (size_of::<(CompactRow, BinaryHeap<Entry>)>() + 16),
        )?;
        add(
            add(add(self.row_bytes, self.heap_bytes)?, table)?,
            size_of::<Self>()
                + self.partition_converter.size()
                + self.order_converter.size()
                + self.payload_converter.size()
                + self.orders.capacity() * size_of::<PhysicalSortExpr>()
                + self.partition_exprs.capacity() * size_of::<Arc<dyn PhysicalExpr>>(),
        )
    }

    pub(crate) fn insert_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let baseline = self.metrics.baseline.clone();
        let _timer = baseline.elapsed_compute().timer();
        if batch.num_rows() == 0 {
            return Ok(());
        }
        // Charge conversion, owned-key copies, table/heap reallocations and
        // simultaneously live old/new allocations before constructing them.
        let mut scratch = add(4096, mul(batch.num_rows(), 4096)?)?;
        for array in batch.columns() {
            scratch = add(scratch, mul(rank_memory::encoded_bound(array)?, 16)?)?;
        }
        scratch = add(scratch, mul(self.heap_bytes, 2)?)?;
        scratch = add(
            scratch,
            mul(
                self.groups.capacity(),
                8 * (size_of::<(CompactRow, BinaryHeap<Entry>)>() + 16),
            )?,
        )?;
        self.reservation
            .try_resize(add(self.retained()?, scratch)?.max(self.reservation.size()))?;
        let mut partitions = self
            .partition_exprs
            .iter()
            .map(|e| {
                e.evaluate(batch)
                    .and_then(|a| a.into_array(batch.num_rows()))
            })
            .collect::<Result<Vec<_>>>()?;
        if partitions.is_empty() {
            partitions.push(Arc::new(arrow::array::NullArray::new(batch.num_rows())));
        }
        let keys = self.partition_converter.convert_columns(&partitions)?;
        let order_arrays = self
            .orders
            .iter()
            .map(|e| {
                e.expr
                    .evaluate(batch)
                    .and_then(|a| a.into_array(batch.num_rows()))
            })
            .collect::<Result<Vec<_>>>()?;
        let orders = self.order_converter.convert_columns(&order_arrays)?;
        let payloads = self.payload_converter.convert_columns(batch.columns())?;
        let mut replacements = 0;
        for i in 0..batch.num_rows() {
            use std::hash::BuildHasher;
            let key = keys.row(i);
            let order = orders.row(i);
            let hash = self.groups.hasher().hash_one(key);
            let mut fresh_heap_capacity = 0;
            let (_, heap) = self
                .groups
                .raw_entry_mut()
                .from_hash(hash, |k| k.as_ref() == key.as_ref())
                .or_insert_with(|| {
                    let key = CompactRow::new(key.as_ref());
                    self.row_bytes += key.allocated();
                    // Most windows keep one or two rows. Starting with K would
                    // over-allocate for large K and sparse groups.
                    let heap = BinaryHeap::with_capacity(self.k.min(2));
                    fresh_heap_capacity = heap.capacity();
                    (key, heap)
                });
            self.heap_bytes = add(
                self.heap_bytes,
                mul(fresh_heap_capacity, size_of::<Entry>())?,
            )?;
            if heap.len() == self.k
                && order.as_ref() >= heap.peek().unwrap().order.as_ref()
            {
                continue;
            }
            let payload = payloads.row(i);
            let entry = Entry {
                order: CompactRow::new(order.as_ref()),
                payload: CompactRow::new(payload.as_ref()),
            };
            self.row_bytes = add(
                self.row_bytes,
                add(entry.order.allocated(), entry.payload.allocated())?,
            )?;
            if heap.len() == self.k {
                let mut worst = heap.peek_mut().unwrap();
                self.row_bytes -= worst.order.allocated() + worst.payload.allocated();
                *worst = entry;
            } else {
                let old_capacity = heap.capacity();
                heap.push(entry);
                self.rows += 1;
                self.heap_bytes = add(
                    self.heap_bytes,
                    mul(heap.capacity() - old_capacity, size_of::<Entry>())?,
                )?;
            }
            replacements += 1;
        }
        self.metrics.row_replacements.add(replacements);
        drop(keys);
        drop(orders);
        drop(payloads);
        drop(partitions);
        drop(order_arrays);
        let retained = self.retained()?;
        assert!(
            retained <= self.reservation.size(),
            "row-number working bound"
        );
        self.reservation.shrink(self.reservation.size() - retained);
        Ok(())
    }

    pub(crate) fn emit(mut self) -> Result<SendableRecordBatchStream> {
        self.batch_size = self.batch_size.min(self.rows.max(1));
        let extra = add(
            mul(
                self.groups.len(),
                size_of::<(CompactRow, BinaryHeap<Entry>)>(),
            )?,
            mul(self.batch_size, size_of::<Entry>())?,
        )?;
        let retained = self.retained()?;
        self.reservation.try_grow(extra)?;
        let mut groups: Vec<_> = std::mem::take(&mut self.groups).into_iter().collect();
        groups.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        struct Emitter {
            groups: std::vec::IntoIter<(CompactRow, BinaryHeap<Entry>)>,
            current: std::vec::IntoIter<Entry>,
            retained: usize,
            extra: usize,
            owner: RowNumberTopK,
        }
        let schema = Arc::clone(&self.schema);
        let emitter = Emitter {
            groups: groups.into_iter(),
            current: Vec::new().into_iter(),
            retained,
            extra,
            owner: self,
        };
        let stream = futures::stream::try_unfold(emitter, |mut e| async move {
            tokio::task::yield_now().await;
            let baseline = e.owner.metrics.baseline.clone();
            let _timer = baseline.elapsed_compute().timer();
            let mut rows = Vec::with_capacity(e.owner.batch_size);
            while rows.len() < e.owner.batch_size {
                if let Some(row) = e.current.next() {
                    rows.push(row);
                } else if let Some((_, heap)) = e.groups.next() {
                    e.current = heap.into_sorted_vec().into_iter();
                } else {
                    break;
                }
            }
            if rows.is_empty() {
                return Ok(None);
            }
            let encoded = rows
                .iter()
                .try_fold(0, |n, r| add(n, r.payload.as_ref().len()))?;
            let output = add(
                mul(encoded, 4)?,
                mul(mul(rows.len(), e.owner.schema.fields().len())?, 256)?,
            )?;
            e.owner.reservation.try_resize(
                add(add(e.retained, e.extra)?, output)?.max(e.owner.reservation.size()),
            )?;
            let parser = e.owner.payload_converter.parser();
            let arrays = e
                .owner
                .payload_converter
                .convert_rows(rows.iter().map(|r| parser.parse(r.payload.as_ref())))?;
            let batch = RecordBatch::try_new(Arc::clone(&e.owner.schema), arrays)?;
            (&batch).record_output(&e.owner.metrics.baseline);
            drop(rows);
            Ok(Some((batch, e)))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_execution::{
        memory_pool::FairSpillPool, runtime_env::RuntimeEnvBuilder,
    };
    use datafusion_physical_expr::expressions::col;
    use futures::TryStreamExt;

    fn make_state(k: usize, limit: usize) -> (RowNumberTopK, Arc<RuntimeEnv>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(FairSpillPool::new(limit)))
            .build_arc()
            .unwrap();
        let orders =
            LexOrdering::new([
                PhysicalSortExpr::new_default(col("v", &schema).unwrap()).desc()
            ])
            .unwrap();
        let state = RowNumberTopK::try_new(
            0,
            Arc::clone(&schema),
            vec![col("k", &schema).unwrap()],
            vec![SortField::new(DataType::Int64)],
            orders,
            k,
            17,
            &runtime,
            &ExecutionPlanMetricsSet::new(),
        )
        .unwrap();
        (state, runtime)
    }

    #[tokio::test]
    async fn compact_row_number_batches_across_groups_and_releases_memory() -> Result<()>
    {
        for k in [1, 2, 20] {
            let (mut state, runtime) = make_state(k, 64 * 1024 * 1024);
            for _ in 0..3 {
                let batch = RecordBatch::try_new(
                    Arc::clone(&state.schema),
                    vec![
                        Arc::new(Int64Array::from_iter(
                            (0..100).map(|i| if i == 0 { None } else { Some(i) }),
                        )),
                        Arc::new(Float64Array::from(vec![Some(7.0); 100])),
                    ],
                )?;
                state.insert_batch(&batch)?;
            }
            assert!(runtime.memory_pool.reserved() > 0);
            let batches: Vec<_> = state.emit()?.try_collect().await?;
            assert_eq!(
                batches.iter().map(|b| b.num_rows()).sum::<usize>(),
                100 * k.min(3)
            );
            assert!(batches.iter().all(|b| b.num_rows() <= 17));
            assert_eq!(batches.len(), (100 * k.min(3)).div_ceil(17));
            assert_eq!(runtime.memory_pool.reserved(), 0);
        }
        Ok(())
    }

    #[tokio::test]
    async fn compact_row_number_paused_stream_and_refusal_return_credits() -> Result<()> {
        use futures::StreamExt;
        let (mut state, runtime) = make_state(2, 8 * 1024 * 1024);
        let batch = RecordBatch::try_new(
            Arc::clone(&state.schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..100)),
                Arc::new(Float64Array::from_iter_values((0..100).map(|i| i as f64))),
            ],
        )?;
        state.insert_batch(&batch)?;
        let mut stream = state.emit()?;
        stream.next().await.unwrap()?;
        assert!(runtime.memory_pool.reserved() > 0);
        drop(stream);
        assert_eq!(runtime.memory_pool.reserved(), 0);
        let (mut state, runtime) = make_state(2, 64 * 1024);
        let error = state.insert_batch(&batch).unwrap_err();
        assert!(error.to_string().contains("RowNumberTopK"), "{error}");
        drop(state);
        assert_eq!(runtime.memory_pool.reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn compact_row_number_many_groups_match_sorted_values() -> Result<()> {
        use arrow::array::Array;
        let (mut state, runtime) = make_state(2, 128 * 1024 * 1024);
        let mut expected = vec![Vec::<f64>::new(); 10_000];
        for batch in 0..20 {
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for row in 0..4096 {
                let key = (batch * 4096 + row) % expected.len();
                let value = ((batch * 97 + row * 13) % 10_007) as f64;
                expected[key].push(value);
                keys.push(key as i64);
                values.push(value);
            }
            let input = RecordBatch::try_new(
                Arc::clone(&state.schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(Float64Array::from(values)),
                ],
            )?;
            state.insert_batch(&input)?;
        }
        println!("row_number_10k_groups_retained_bytes={}", state.retained()?);
        let output: Vec<_> = state.emit()?.try_collect().await?;
        let mut actual = vec![Vec::<f64>::new(); expected.len()];
        for batch in output {
            let keys = batch
                .column(0)
                .as_primitive::<arrow::datatypes::Int64Type>();
            let values = batch
                .column(1)
                .as_primitive::<arrow::datatypes::Float64Type>();
            assert_eq!(keys.null_count(), 0);
            for row in 0..batch.num_rows() {
                actual[keys.value(row) as usize].push(values.value(row));
            }
        }
        for values in &mut expected {
            values.sort_by(|a, b| b.total_cmp(a));
            values.truncate(2);
        }
        assert_eq!(actual, expected);
        assert_eq!(runtime.memory_pool.reserved(), 0);
        Ok(())
    }
}
