// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! FunctionalDependencies keeps track of functional dependencies
//! inside DFSchema.

use std::fmt::{Display, Formatter};
use std::ops::Deref;
use std::vec::IntoIter;

use crate::utils::{merge_and_order_indices, set_difference};
use crate::{DFSchema, HashSet, JoinType, NullEquality};

/// This object defines a constraint on a table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum Constraint {
    /// Columns with the given indices form a composite primary key (they are
    /// jointly unique and not nullable):
    PrimaryKey(Vec<usize>),
    /// Columns with the given indices form a composite unique key:
    Unique(Vec<usize>),
}

/// This object encapsulates a list of functional constraints:
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq, PartialOrd)]
pub struct Constraints {
    inner: Vec<Constraint>,
}

impl Constraints {
    /// Create a new [`Constraints`] object from the given `constraints`.
    /// Users should use the [`Constraints::default`] or [`SqlToRel::new_constraint_from_table_constraints`]
    /// functions for constructing [`Constraints`] instances. This constructor
    /// is for internal purposes only and does not check whether the argument
    /// is valid. The user is responsible for supplying a valid vector of
    /// [`Constraint`] objects.
    ///
    /// [`SqlToRel::new_constraint_from_table_constraints`]: https://docs.rs/datafusion/latest/datafusion/sql/planner/struct.SqlToRel.html#method.new_constraint_from_table_constraints
    pub fn new_unverified(constraints: Vec<Constraint>) -> Self {
        Self { inner: constraints }
    }

    /// Extends the current constraints with the given `other` constraints.
    pub fn extend(&mut self, other: Constraints) {
        self.inner.extend(other.inner);
    }

    /// Projects constraints using the given projection indices. Returns `None`
    /// if any of the constraint columns are not included in the projection.
    pub fn project(&self, proj_indices: &[usize]) -> Option<Self> {
        let projected = self
            .inner
            .iter()
            .filter_map(|constraint| {
                match constraint {
                    Constraint::PrimaryKey(indices) => {
                        let new_indices =
                            update_elements_with_matching_indices(indices, proj_indices);
                        // Only keep the constraint if all columns are preserved:
                        (new_indices.len() == indices.len())
                            .then_some(Constraint::PrimaryKey(new_indices))
                    }
                    Constraint::Unique(indices) => {
                        let new_indices =
                            update_elements_with_matching_indices(indices, proj_indices);
                        // Only keep the constraint if all columns are preserved:
                        (new_indices.len() == indices.len())
                            .then_some(Constraint::Unique(new_indices))
                    }
                }
            })
            .collect::<Vec<_>>();

        (!projected.is_empty()).then_some(Constraints::new_unverified(projected))
    }
}

impl IntoIterator for Constraints {
    type Item = Constraint;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl Display for Constraints {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let pk = self
            .inner
            .iter()
            .map(|c| format!("{c:?}"))
            .collect::<Vec<_>>();
        let pk = pk.join(", ");
        write!(f, "constraints=[{pk}]")
    }
}

impl Deref for Constraints {
    type Target = [Constraint];

    fn deref(&self) -> &Self::Target {
        self.inner.as_slice()
    }
}

/// This object defines a functional dependence in the schema. A functional
/// dependence defines a relationship between determinant keys and dependent
/// columns. A determinant key is a column, or a set of columns, whose value
/// uniquely determines values of some other (dependent) columns. If two rows
/// have the same determinant key, dependent columns in these rows are
/// necessarily the same. If the determinant key is unique, the set of
/// dependent columns is equal to the entire schema and the determinant key can
/// serve as a primary key. Note that a primary key may "downgrade" into a
/// determinant key due to an operation such as a join, and this object is
/// used to track dependence relationships in such cases. For more information
/// on functional dependencies, see:
/// <https://www.scaler.com/topics/dbms/functional-dependency-in-dbms/>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionalDependence {
    // Column indices of the (possibly composite) determinant key:
    pub source_indices: Vec<usize>,
    // Column indices of dependent column(s):
    pub target_indices: Vec<usize>,
    /// Flag indicating whether one of the `source_indices` can receive NULL values.
    /// For a data source, if the constraint in question is `Constraint::Unique`,
    /// this flag is `true`. If the constraint in question is `Constraint::PrimaryKey`,
    /// this flag is `false`.
    /// Note that as the schema changes between different stages in a plan,
    /// such as after LEFT JOIN or RIGHT JOIN operations, this property may
    /// change.
    pub nullable: bool,
    /// The NULL-comparison semantics under which this dependency holds. The
    /// conservative default, [`NullEquality::NullEqualsNothing`], means it
    /// holds only across rows whose determinant contains no NULLs; e.g. a
    /// nullable `UNIQUE` constraint permits multiple NULL rows that may differ.
    /// [`NullEquality::NullEqualsNull`] means it also holds when NULL
    /// determinant values are treated as equal; e.g. a `GROUP BY` key.
    pub null_equality: NullEquality,
    // The functional dependency mode:
    pub mode: Dependency,
}

/// Describes functional dependency mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dependency {
    /// A determinant key may occur only once.
    Single,
    /// A determinant key may occur multiple times (in multiple rows).
    Multi,
}

impl FunctionalDependence {
    // Creates a new functional dependence.
    pub fn new(
        source_indices: Vec<usize>,
        target_indices: Vec<usize>,
        nullable: bool,
    ) -> Self {
        Self {
            source_indices,
            target_indices,
            nullable,
            // Assume the dependency does not hold across NULL rows by default:
            null_equality: NullEquality::NullEqualsNothing,
            // Start with the least restrictive mode by default:
            mode: Dependency::Multi,
        }
    }

    pub fn with_mode(mut self, mode: Dependency) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_null_equality(mut self, null_equality: NullEquality) -> Self {
        self.null_equality = null_equality;
        self
    }

    /// Returns `true` if this dependency remains usable for operations that
    /// treat NULL determinant values as equal (`GROUP BY`, `DISTINCT` and
    /// sorting, which place all NULL keys together): it must hold under
    /// NULLs-are-equal semantics, have a non-nullable determinant (e.g. a
    /// `PRIMARY KEY`), or have no nullable source field in the given `schema`.
    pub fn is_valid_across_nulls(&self, schema: &DFSchema) -> bool {
        self.null_equality == NullEquality::NullEqualsNull
            || !self.nullable
            || self
                .source_indices
                .iter()
                .all(|&source_idx| !schema.field(source_idx).is_nullable())
    }
}

/// This object encapsulates all functional dependencies in a given relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionalDependencies {
    deps: Vec<FunctionalDependence>,
}

impl FunctionalDependencies {
    /// Creates an empty `FunctionalDependencies` object.
    pub fn empty() -> Self {
        Self { deps: vec![] }
    }

    /// Creates a new `FunctionalDependencies` object from a vector of
    /// `FunctionalDependence` objects.
    pub fn new(dependencies: Vec<FunctionalDependence>) -> Self {
        Self { deps: dependencies }
    }

    /// Creates a new `FunctionalDependencies` object from the given constraints.
    pub fn new_from_constraints(
        constraints: Option<&Constraints>,
        n_field: usize,
    ) -> Self {
        if let Some(Constraints { inner: constraints }) = constraints {
            // Construct dependency objects based on each individual constraint:
            let dependencies = constraints
                .iter()
                .map(|constraint| {
                    // All the field indices are associated with the whole table
                    // since we are dealing with table level constraints:
                    let dependency = match constraint {
                        Constraint::PrimaryKey(indices) => FunctionalDependence::new(
                            indices.to_vec(),
                            (0..n_field).collect::<Vec<_>>(),
                            false,
                        ),
                        Constraint::Unique(indices) => FunctionalDependence::new(
                            indices.to_vec(),
                            (0..n_field).collect::<Vec<_>>(),
                            true,
                        ),
                    };
                    // As primary keys are guaranteed to be unique, set the
                    // functional dependency mode to `Dependency::Single`:
                    dependency.with_mode(Dependency::Single)
                })
                .collect::<Vec<_>>();
            Self::new(dependencies)
        } else {
            // There is no constraint, return an empty object:
            Self::empty()
        }
    }

    pub fn with_dependency(mut self, mode: Dependency) -> Self {
        self.deps.iter_mut().for_each(|item| item.mode = mode);
        self
    }

    /// Merges the given functional dependencies with these.
    pub fn extend(&mut self, other: FunctionalDependencies) {
        self.deps.extend(other.deps);
    }

    /// Sanity checks if functional dependencies are valid. For example, if
    /// there are 10 fields, we cannot receive any index further than 9.
    pub fn is_valid(&self, n_field: usize) -> bool {
        self.deps.iter().all(
            |FunctionalDependence {
                 source_indices,
                 target_indices,
                 ..
             }| {
                source_indices
                    .iter()
                    .max()
                    .map(|&max_index| max_index < n_field)
                    .unwrap_or(true)
                    && target_indices
                        .iter()
                        .max()
                        .map(|&max_index| max_index < n_field)
                        .unwrap_or(true)
            },
        )
    }

    /// Adds the `offset` value to `source_indices` and `target_indices` for
    /// each functional dependency.
    pub fn add_offset(&mut self, offset: usize) {
        self.deps.iter_mut().for_each(
            |FunctionalDependence {
                 source_indices,
                 target_indices,
                 ..
             }| {
                *source_indices = add_offset_to_vec(source_indices, offset);
                *target_indices = add_offset_to_vec(target_indices, offset);
            },
        )
    }

    /// Updates `source_indices` and `target_indices` of each functional
    /// dependence using the index mapping given in `proj_indices`.
    ///
    /// Assume that `proj_indices` is \[2, 5, 8\] and we have a functional
    /// dependence \[5\] (`source_indices`) -> \[5, 8\] (`target_indices`).
    /// In the updated schema, fields at indices \[2, 5, 8\] will transform
    /// to \[0, 1, 2\]. Therefore, the resulting functional dependence will
    /// be \[1\] -> \[1, 2\].
    pub fn project_functional_dependencies(
        &self,
        proj_indices: &[usize],
        // The argument `n_out` denotes the schema field length, which is needed
        // to correctly associate a `Single`-mode dependence with the whole table.
        n_out: usize,
    ) -> FunctionalDependencies {
        let mut projected_func_dependencies = Vec::with_capacity(self.deps.len());
        for FunctionalDependence {
            source_indices,
            target_indices,
            nullable,
            null_equality,
            mode,
        } in &self.deps
        {
            let new_source_indices =
                update_elements_with_matching_indices(source_indices, proj_indices);
            let new_target_indices = if *mode == Dependency::Single {
                // Associate with all of the fields in the schema:
                (0..n_out).collect()
            } else {
                // Update associations according to projection:
                update_elements_with_matching_indices(target_indices, proj_indices)
            };
            // All of the composite indices should still be valid after projection;
            // otherwise, functional dependency cannot be propagated.
            if new_source_indices.len() == source_indices.len() {
                let new_func_dependence = FunctionalDependence::new(
                    new_source_indices,
                    new_target_indices,
                    *nullable,
                )
                .with_mode(*mode)
                .with_null_equality(*null_equality);
                projected_func_dependencies.push(new_func_dependence);
            }
        }
        FunctionalDependencies::new(projected_func_dependencies)
    }

    /// This function joins this set of functional dependencies with the `other`
    /// according to the given `join_type`.
    pub fn join(
        &self,
        other: &FunctionalDependencies,
        join_type: &JoinType,
        left_cols_len: usize,
    ) -> FunctionalDependencies {
        // Get mutable copies of left and right side dependencies:
        let mut right_func_dependencies = other.clone();
        let mut left_func_dependencies = self.clone();

        match join_type {
            JoinType::Inner | JoinType::Left | JoinType::Right => {
                // Add offset to right schema:
                right_func_dependencies.add_offset(left_cols_len);

                // Result may have multiple values, update the dependency mode:
                left_func_dependencies =
                    left_func_dependencies.with_dependency(Dependency::Multi);
                right_func_dependencies =
                    right_func_dependencies.with_dependency(Dependency::Multi);

                if *join_type == JoinType::Left {
                    // Downgrade the right side, since it may have additional NULL values:
                    right_func_dependencies.downgrade_dependencies();
                } else if *join_type == JoinType::Right {
                    // Downgrade the left side, since it may have additional NULL values:
                    left_func_dependencies.downgrade_dependencies();
                }
                // Combine left and right functional dependencies:
                left_func_dependencies.extend(right_func_dependencies);
                left_func_dependencies
            }
            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
                // These joins preserve functional dependencies of the left side:
                left_func_dependencies
            }
            JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => {
                // These joins preserve functional dependencies of the right side:
                right_func_dependencies
            }
            JoinType::Full => {
                // All of the functional dependencies are lost in a FULL join:
                FunctionalDependencies::empty()
            }
        }
    }

    /// This function downgrades a functional dependency when nullability becomes
    /// a possibility:
    /// - If the dependency in question is UNIQUE (i.e. nullable), a new null value
    ///   invalidates the dependency.
    /// - If the dependency in question is PRIMARY KEY (i.e. not nullable), a new
    ///   null value turns it into UNIQUE mode.
    fn downgrade_dependencies(&mut self) {
        // Delete nullable dependencies, since they are no longer valid:
        self.deps.retain(|item| !item.nullable);
        // Survivors had a non-null determinant, so every new NULL comes from a
        // padded row whose targets are all NULL too:
        self.deps.iter_mut().for_each(|item| {
            item.nullable = true;
            item.null_equality = NullEquality::NullEqualsNull;
        });
    }

    /// This function ensures that functional dependencies involving uniquely
    /// occurring determinant keys cover their entire table in terms of
    /// dependent columns.
    pub fn extend_target_indices(&mut self, n_out: usize) {
        self.deps.iter_mut().for_each(
            |FunctionalDependence {
                 mode,
                 target_indices,
                 ..
             }| {
                // If unique, cover the whole table:
                if *mode == Dependency::Single {
                    *target_indices = (0..n_out).collect::<Vec<_>>();
                }
            },
        )
    }
}

impl Deref for FunctionalDependencies {
    type Target = [FunctionalDependence];

    fn deref(&self) -> &Self::Target {
        self.deps.as_slice()
    }
}

/// Calculates functional dependencies for aggregate output, when there is a GROUP BY expression.
///
/// `group_indices[i]` is the index of the input field that GROUP BY
/// expression `i` passes through unchanged (a column reference), or `None` if
/// the expression computes a new value. Only a column reference can carry an
/// input dependency; a computed expression such as `CAST(pk AS INT)` cannot,
/// even though its output name may equal the column's name.
pub fn aggregate_functional_dependencies(
    input: &DFSchema,
    group_indices: &[Option<usize>],
    output: &DFSchema,
) -> FunctionalDependencies {
    if group_indices.is_empty() {
        return FunctionalDependencies::empty();
    }
    let valid = input
        .functional_dependencies()
        .iter()
        .filter(|dep| dep.is_valid_across_nulls(input))
        .collect::<Vec<_>>();
    let mut result = vec![];
    for dep in input.functional_dependencies().iter() {
        if dep.source_indices.is_empty() {
            continue;
        }
        let Some(sources) = dep
            .source_indices
            .iter()
            .map(|source| {
                group_indices
                    .iter()
                    .position(|index| *index == Some(*source))
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if !dep.is_valid_across_nulls(input) {
            // A nullable UNIQUE key still identifies at most one input row
            // for each non-NULL key. Preserve that conditional uniqueness,
            // but do not use this fact to close dependencies across NULLs.
            if dep.mode == Dependency::Single {
                result.push(
                    FunctionalDependence::new(
                        sources,
                        (0..output.fields().len()).collect(),
                        dep.nullable,
                    )
                    .with_mode(Dependency::Single)
                    .with_null_equality(NullEquality::NullEqualsNothing),
                );
            }
            continue;
        }
        let mut closure = dep.source_indices.iter().copied().collect::<HashSet<_>>();
        loop {
            let previous_len = closure.len();
            for candidate in &valid {
                if candidate
                    .source_indices
                    .iter()
                    .all(|index| closure.contains(index))
                {
                    closure.extend(candidate.target_indices.iter().copied());
                }
            }
            if closure.len() == previous_len {
                break;
            }
        }
        // An input-unique determinant remains unique even with computed keys.
        // A repeated determinant must determine every group before it can
        // determine aggregate outputs. Unrelated and computed groups cannot
        // disappear merely because they added no input dependency targets.
        let complete = dep.mode == Dependency::Single
            || group_indices
                .iter()
                .all(|index| index.is_some_and(|i| closure.contains(&i)));
        let (targets, mode) = if complete {
            ((0..output.fields().len()).collect(), Dependency::Single)
        } else {
            (
                group_indices
                    .iter()
                    .enumerate()
                    .filter_map(|(index, source)| {
                        source
                            .is_some_and(|i| closure.contains(&i))
                            .then_some(index)
                    })
                    .collect(),
                Dependency::Multi,
            )
        };
        result.push(
            FunctionalDependence::new(sources, targets, dep.nullable)
                .with_mode(mode)
                .with_null_equality(dep.null_equality),
        );
    }
    // GROUP BY is unique on the entire key, including NULL groups. An
    // already-unique determinant also guarantees this; a Multi dependency
    // on only part of the key cannot replace the complete-key guarantee.
    let group_key = (0..group_indices.len()).collect::<Vec<_>>();
    if let Some(dep) = result
        .iter_mut()
        .find(|dep| dep.mode == Dependency::Single && dep.is_valid_across_nulls(output))
    {
        if dep.source_indices == group_key {
            dep.null_equality = NullEquality::NullEqualsNull;
        }
    } else {
        let nullable = group_key
            .iter()
            .any(|&index| output.field(index).is_nullable());
        result.push(
            FunctionalDependence::new(
                group_key,
                (0..output.fields().len()).collect(),
                nullable,
            )
            .with_mode(Dependency::Single)
            .with_null_equality(NullEquality::NullEqualsNull),
        );
    }
    FunctionalDependencies::new(result)
}

/// Returns target indices, for the determinant keys that are inside
/// group by expressions.
///
/// Only dependencies that hold when NULL determinant values are grouped
/// together are considered (see
/// [`FunctionalDependence::is_valid_across_nulls`]): a nullable `UNIQUE`
/// column permits multiple NULL rows that may differ in the other columns,
/// so it does not determine them across the single NULL group.
///
/// `group_by_input_indices` holds, for each GROUP BY expression, the index of
/// the `schema` field it references, or `None` for a computed expression.
pub fn get_target_functional_dependencies(
    schema: &DFSchema,
    group_by_input_indices: &[Option<usize>],
) -> Option<Vec<usize>> {
    let dependencies = schema.functional_dependencies();
    if dependencies.is_empty() {
        return None;
    }
    let dependencies = dependencies
        .iter()
        .filter(|dependency| dependency.is_valid_across_nulls(schema));
    get_target_indices_from_dependencies(dependencies, group_by_input_indices)
}

fn get_target_indices_from_dependencies<'a>(
    dependencies: impl IntoIterator<Item = &'a FunctionalDependence>,
    group_by_input_indices: &[Option<usize>],
) -> Option<Vec<usize>> {
    let mut combined_target_indices = HashSet::new();
    for dependency in dependencies {
        // If the GROUP BY expression contains a determinant key, we can use
        // the associated fields after aggregation even if they are not part
        // of the GROUP BY expression.
        if dependency
            .source_indices
            .iter()
            .all(|source_idx| group_by_input_indices.contains(&Some(*source_idx)))
        {
            combined_target_indices.extend(dependency.target_indices.iter());
        }
    }
    (!combined_target_indices.is_empty()).then_some({
        let mut result = combined_target_indices.into_iter().collect::<Vec<_>>();
        result.sort_unstable();
        result
    })
}

/// Returns indices for the minimal subset of GROUP BY expressions that are
/// functionally equivalent to the original set of GROUP BY expressions.
///
/// `group_by_input_indices` holds, for each GROUP BY expression, the index of
/// the `schema` field it references, or `None` for a computed expression. If
/// any GROUP BY expression is computed, returns `None`.
pub fn get_required_group_by_exprs_indices(
    schema: &DFSchema,
    group_by_input_indices: &[Option<usize>],
) -> Option<Vec<usize>> {
    let dependencies = schema.functional_dependencies();
    let mut groupby_expr_indices = group_by_input_indices
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()?;

    groupby_expr_indices.sort_unstable();
    for dependency in &dependencies.deps {
        // If all source indices are among GROUP BY expression indices, we
        // can remove target indices from GROUP BY expression indices and
        // use source indices instead. The dependency must also hold across
        // NULL keys, which GROUP BY collapses into a single group.
        if dependency
            .source_indices
            .iter()
            .all(|source_idx| groupby_expr_indices.contains(source_idx))
            && dependency.is_valid_across_nulls(schema)
        {
            groupby_expr_indices =
                set_difference(&groupby_expr_indices, &dependency.target_indices);
            groupby_expr_indices =
                merge_and_order_indices(groupby_expr_indices, &dependency.source_indices);
        }
    }
    groupby_expr_indices
        .iter()
        .map(|idx| {
            group_by_input_indices
                .iter()
                .position(|input_idx| *input_idx == Some(*idx))
        })
        .collect()
}

/// Returns indices for the minimal subset of ORDER BY expressions that are
/// functionally equivalent to the original set of ORDER BY expressions.
///
/// `sort_input_indices` holds, for each ORDER BY expression, the index of the
/// `schema` field it references, or `None` for a computed expression.
pub fn get_required_sort_exprs_indices(
    schema: &DFSchema,
    sort_input_indices: &[Option<usize>],
) -> Vec<usize> {
    let dependencies = schema.functional_dependencies();

    let mut known_field_indices = HashSet::with_capacity(sort_input_indices.len());
    let mut required_sort_expr_indices = Vec::with_capacity(sort_input_indices.len());

    for (sort_expr_idx, field_idx) in sort_input_indices.iter().enumerate() {
        // If the sort expression doesn't reference a schema field (e.g. a
        // computed expression), we can't reason about it via functional
        // dependencies, so conservatively keep it.
        let Some(field_idx) = *field_idx else {
            required_sort_expr_indices.push(sort_expr_idx);
            continue;
        };

        // A sort expression is removable if its value is functionally determined
        // by fields that already appear earlier in the sort order (and the
        // dependency remains valid across NULL rows).
        let removable = dependencies.deps.iter().any(|dependency| {
            dependency.target_indices.contains(&field_idx)
                && dependency
                    .source_indices
                    .iter()
                    .all(|source_idx| known_field_indices.contains(source_idx))
                && dependency.is_valid_across_nulls(schema)
        });

        if removable {
            continue;
        }

        known_field_indices.insert(field_idx);
        required_sort_expr_indices.push(sort_expr_idx);
    }

    required_sort_expr_indices
}

/// Updates entries inside the `entries` vector with their corresponding
/// indices inside the `proj_indices` vector.
fn update_elements_with_matching_indices(
    entries: &[usize],
    proj_indices: &[usize],
) -> Vec<usize> {
    entries
        .iter()
        .filter_map(|val| proj_indices.iter().position(|proj_idx| proj_idx == val))
        .collect()
}

/// Adds `offset` value to each entry inside `in_data`.
fn add_offset_to_vec<T: Copy + std::ops::Add<Output = T>>(
    in_data: &[T],
    offset: T,
) -> Vec<T> {
    in_data.iter().map(|&item| item + offset).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraints_iter() {
        let constraints = Constraints::new_unverified(vec![
            Constraint::PrimaryKey(vec![10]),
            Constraint::Unique(vec![20]),
        ]);
        let mut iter = constraints.iter();
        assert_eq!(iter.next(), Some(&Constraint::PrimaryKey(vec![10])));
        assert_eq!(iter.next(), Some(&Constraint::Unique(vec![20])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_project_constraints() {
        let constraints = Constraints::new_unverified(vec![
            Constraint::PrimaryKey(vec![1, 2]),
            Constraint::Unique(vec![0, 3]),
        ]);

        // Project keeping columns 1,2,3
        let projected = constraints.project(&[1, 2, 3]).unwrap();
        assert_eq!(
            projected,
            Constraints::new_unverified(vec![Constraint::PrimaryKey(vec![0, 1])])
        );

        // Project keeping only column 0 - should return None as no constraints are preserved
        assert!(constraints.project(&[0]).is_none());
    }

    #[test]
    fn test_get_updated_id_keys() {
        let fund_dependencies =
            FunctionalDependencies::new(vec![FunctionalDependence::new(
                vec![1],
                vec![0, 1, 2],
                true,
            )]);
        let res = fund_dependencies.project_functional_dependencies(&[1, 2], 2);
        let expected = FunctionalDependencies::new(vec![FunctionalDependence::new(
            vec![0],
            vec![0, 1],
            true,
        )]);
        assert_eq!(res, expected);
    }
}

#[cfg(test)]
mod aggregate_dependency_tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};

    fn schemas(dep: FunctionalDependence, nullable: bool) -> (DFSchema, DFSchema) {
        let input = DFSchema::try_from(Schema::new(
            (0..4)
                .map(|i| Field::new(format!("c{i}"), DataType::Int64, nullable))
                .collect::<Vec<_>>(),
        ))
        .unwrap()
        .with_functional_dependencies(FunctionalDependencies::new(vec![dep]))
        .unwrap();
        let output = DFSchema::try_from(Schema::new(vec![
            Field::new("c0", DataType::Int64, nullable),
            Field::new("group", DataType::Int64, nullable),
            Field::new("count", DataType::Int64, false),
        ]))
        .unwrap();
        (input, output)
    }

    #[test]
    fn aggregate_public_helper_does_not_promote_partial_keys() {
        let (input, output) =
            schemas(FunctionalDependence::new(vec![0], vec![0, 1], false), false);
        for group in [vec![Some(0), Some(2)], vec![Some(0), None]] {
            let deps = aggregate_functional_dependencies(&input, &group, &output);
            assert_eq!(deps.len(), 2);
            assert_eq!(deps[0].source_indices, [0]);
            assert_eq!(deps[0].target_indices, [0]);
            assert_eq!(deps[0].mode, Dependency::Multi);
            assert_eq!(deps[1].source_indices, [0, 1]);
            assert_eq!(deps[1].target_indices, [0, 1, 2]);
            assert_eq!(deps[1].mode, Dependency::Single);
            assert_eq!(deps[1].null_equality, NullEquality::NullEqualsNull);
        }
    }

    #[test]
    fn aggregate_public_helper_retains_strict_nullable_uniqueness() {
        let (input, output) = schemas(
            FunctionalDependence::new(vec![0], vec![0, 1, 2, 3], true)
                .with_mode(Dependency::Single),
            true,
        );
        let deps =
            aggregate_functional_dependencies(&input, &[Some(0), Some(1)], &output);
        assert_eq!(deps.len(), 2);
        assert_eq!(deps[0].source_indices, [0]);
        assert_eq!(deps[0].target_indices, [0, 1, 2]);
        assert_eq!(deps[0].mode, Dependency::Single);
        assert!(deps[0].nullable);
        assert_eq!(deps[0].null_equality, NullEquality::NullEqualsNothing);
        assert!(!deps[0].is_valid_across_nulls(&output));
        assert_eq!(deps[1].source_indices, [0, 1]);
        assert_eq!(deps[1].null_equality, NullEquality::NullEqualsNull);
    }
}
