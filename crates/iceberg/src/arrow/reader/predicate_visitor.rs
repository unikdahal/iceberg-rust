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

//! Visitors that translate Iceberg bound predicates into the pieces needed for
//! Arrow-level evaluation: collecting referenced field IDs and producing
//! per-record-batch predicate closures.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_arith::boolean::{and, and_kleene, is_not_null, is_null, not, or, or_kleene};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int32Type, Int64Type, Time64MicrosecondType,
    TimestampMicrosecondType, TimestampNanosecondType,
};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum as ArrowDatum, RecordBatch, Scalar};
use arrow_buffer::BooleanBuffer;
use arrow_cast::cast::cast;
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::{ArrowError, DataType, TimeUnit};
use arrow_string::like::starts_with;
use fnv::FnvHashSet;
use parquet::schema::types::SchemaDescriptor;

use crate::arrow::get_arrow_datum;
use crate::error::{Result, invalid_data};
use crate::expr::visitors::bound_predicate_visitor::BoundPredicateVisitor;
use crate::expr::{BoundPredicate, BoundReference};
use crate::spec::Datum;

/// A visitor to collect field ids from bound predicates.
pub(super) struct CollectFieldIdVisitor {
    pub(super) field_ids: HashSet<i32>,
}

impl CollectFieldIdVisitor {
    pub(super) fn field_ids(self) -> HashSet<i32> {
        self.field_ids
    }
}

impl BoundPredicateVisitor for CollectFieldIdVisitor {
    type T = ();

    fn always_true(&mut self) -> Result<()> {
        Ok(())
    }

    fn always_false(&mut self) -> Result<()> {
        Ok(())
    }

    fn and(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn or(&mut self, _lhs: (), _rhs: ()) -> Result<()> {
        Ok(())
    }

    fn not(&mut self, _inner: ()) -> Result<()> {
        Ok(())
    }

    fn is_null(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_null(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn is_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_nan(&mut self, reference: &BoundReference, _predicate: &BoundPredicate) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        _literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        _literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<()> {
        self.field_ids.insert(reference.field().id);
        Ok(())
    }
}

/// A visitor to convert Iceberg bound predicates to Arrow predicates.
pub(super) struct PredicateConverter<'a> {
    /// The Parquet schema descriptor.
    pub(super) parquet_schema: &'a SchemaDescriptor,
    /// The map between field id and leaf column index in Parquet schema.
    pub(super) column_map: &'a HashMap<i32, usize>,
    /// The required column indices in Parquet schema for the predicates.
    pub(super) column_indices: &'a Vec<usize>,
}

impl PredicateConverter<'_> {
    /// When visiting a bound reference, we return index of the leaf column in the
    /// required column indices which is used to project the column in the record batch.
    /// Return None if the field id is not found in the column map, which is possible
    /// due to schema evolution.
    fn bound_reference(&mut self, reference: &BoundReference) -> Result<Option<usize>> {
        // The leaf column's index in Parquet schema.
        if let Some(column_idx) = self.column_map.get(&reference.field().id) {
            if self.parquet_schema.get_column_root(*column_idx).is_group() {
                return Err(invalid_data!(
                    "Leaf column `{}` in predicates isn't a root column in Parquet schema.",
                    reference.field().name
                ));
            }

            // The leaf column's index in the required column indices.
            let index = self
                .column_indices
                .iter()
                .position(|&idx| idx == *column_idx)
                .ok_or(invalid_data!(
                "Leaf column `{}` in predicates cannot be found in the required column indices.",
                reference.field().name
            ))?;

            Ok(Some(index))
        } else {
            Ok(None)
        }
    }

    /// Compiles membership once per physical column type. Large sets use a
    /// hash lookup per row, including the negated form used by equality deletes.
    fn build_set_predicate(
        &self,
        column_idx: usize,
        literals: &FnvHashSet<Datum>,
        negate: bool,
    ) -> Result<Box<PredicateResult>> {
        let literals: Vec<_> = literals
            .iter()
            .map(get_arrow_datum)
            .collect::<Result<_>>()?;
        let mut set: Option<Option<InSet>> = None;
        Ok(Box::new(move |batch| {
            let mut column = project_column(&batch, column_idx)?;
            if let Some(literal) = literals.first() {
                column = promote_column_for_literal(column, literal.get().0.data_type())?;
            }
            if literals.len() > IN_SET_THRESHOLD {
                if set.is_none() {
                    set = Some(InSet::build(column.data_type(), &literals)?);
                }
                if let Some(mask) = set
                    .as_ref()
                    .unwrap()
                    .as_ref()
                    .and_then(|set| set.mask(&column))
                {
                    return if negate { not(&mask) } else { Ok(mask) };
                }
            }
            let mut acc = constant_bool_array(negate, batch.num_rows());
            for literal in &literals {
                let literal = try_cast_literal(literal, column.data_type())?;
                acc = if negate {
                    and(&acc, &neq(&column, literal.as_ref())?)?
                } else {
                    or(&acc, &eq(&column, literal.as_ref())?)?
                };
            }
            Ok(acc)
        }))
    }

    /// Build an Arrow predicate that always returns true.
    fn build_always_true(&self) -> Result<Box<PredicateResult>> {
        Ok(Box::new(|batch| {
            Ok(constant_bool_array(true, batch.num_rows()))
        }))
    }

    /// Build an Arrow predicate that always returns false.
    fn build_always_false(&self) -> Result<Box<PredicateResult>> {
        Ok(Box::new(|batch| {
            Ok(constant_bool_array(false, batch.num_rows()))
        }))
    }
}

/// Builds a non-null `BooleanArray` of `len` elements all set to `value`.
fn constant_bool_array(value: bool, len: usize) -> BooleanArray {
    let buffer = if value {
        BooleanBuffer::new_set(len)
    } else {
        BooleanBuffer::new_unset(len)
    };

    BooleanArray::new(buffer, None)
}

/// Gets the leaf column from the record batch for the required column index. Only
/// supports top-level columns for now.
fn project_column(
    batch: &RecordBatch,
    column_idx: usize,
) -> std::result::Result<ArrayRef, ArrowError> {
    let column = batch.column(column_idx);

    match column.data_type() {
        DataType::Struct(_) => Err(ArrowError::SchemaError(
            "Does not support struct column yet.".to_string(),
        )),
        _ => Ok(column.clone()),
    }
}

fn compute_is_nan(array: &ArrayRef) -> std::result::Result<BooleanArray, ArrowError> {
    // Compute NaN over the contiguous values slice, then fold the null bitmap
    // in with a single bitwise AND so that null slots become false.
    let (is_nan, nulls) = match array.data_type() {
        DataType::Float32 => {
            let arr = array.as_primitive::<Float32Type>();
            (
                BooleanBuffer::from_iter(arr.values().iter().map(|v| v.is_nan())),
                arr.nulls(),
            )
        }
        DataType::Float64 => {
            let arr = array.as_primitive::<Float64Type>();
            (
                BooleanBuffer::from_iter(arr.values().iter().map(|v| v.is_nan())),
                arr.nulls(),
            )
        }
        _ => unreachable!("is_nan is only valid for float types"),
    };

    let values = match nulls {
        Some(nulls) => &is_nan & nulls.inner(),
        None => is_nan,
    };

    Ok(BooleanArray::new(values, None))
}

pub(super) type PredicateResult =
    dyn FnMut(RecordBatch) -> std::result::Result<BooleanArray, ArrowError> + Send + 'static;

impl BoundPredicateVisitor for PredicateConverter<'_> {
    type T = Box<PredicateResult>;

    fn always_true(&mut self) -> Result<Box<PredicateResult>> {
        self.build_always_true()
    }

    fn always_false(&mut self) -> Result<Box<PredicateResult>> {
        self.build_always_false()
    }

    fn and(
        &mut self,
        mut lhs: Box<PredicateResult>,
        mut rhs: Box<PredicateResult>,
    ) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let left = lhs(batch.clone())?;
            let right = rhs(batch)?;
            and_kleene(&left, &right)
        }))
    }

    fn or(
        &mut self,
        mut lhs: Box<PredicateResult>,
        mut rhs: Box<PredicateResult>,
    ) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let left = lhs(batch.clone())?;
            let right = rhs(batch)?;
            or_kleene(&left, &right)
        }))
    }

    fn not(&mut self, mut inner: Box<PredicateResult>) -> Result<Box<PredicateResult>> {
        Ok(Box::new(move |batch| {
            let pred_ret = inner(batch)?;
            not(&pred_ret)
        }))
    }

    fn is_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                is_null(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn not_null(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                is_not_null(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn is_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                compute_is_nan(&column)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_nan(
        &mut self,
        reference: &BoundReference,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            Ok(Box::new(move |batch| {
                let column = project_column(&batch, idx)?;
                let is_nan = compute_is_nan(&column)?;
                not(&is_nan)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn less_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let left = promote_column_for_literal(left, literal.get().0.data_type())?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                lt(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn less_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let left = promote_column_for_literal(left, literal.get().0.data_type())?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                lt_eq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn greater_than(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let left = promote_column_for_literal(left, literal.get().0.data_type())?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                gt(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn greater_than_or_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let left = promote_column_for_literal(left, literal.get().0.data_type())?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                gt_eq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let left = promote_column_for_literal(left, literal.get().0.data_type())?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                eq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_eq(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let left = promote_column_for_literal(left, literal.get().0.data_type())?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                neq(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                starts_with(&left, literal.as_ref())
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_starts_with(
        &mut self,
        reference: &BoundReference,
        literal: &Datum,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            let literal = get_arrow_datum(literal)?;

            Ok(Box::new(move |batch| {
                let left = project_column(&batch, idx)?;
                let literal = try_cast_literal(&literal, left.data_type())?;
                // update here if arrow ever adds a native not_starts_with
                not(&starts_with(&left, literal.as_ref())?)
            }))
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }

    fn r#in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            self.build_set_predicate(idx, literals, false)
        } else {
            // A missing column, treating it as null.
            self.build_always_false()
        }
    }

    fn not_in(
        &mut self,
        reference: &BoundReference,
        literals: &FnvHashSet<Datum>,
        _predicate: &BoundPredicate,
    ) -> Result<Box<PredicateResult>> {
        if let Some(idx) = self.bound_reference(reference)? {
            self.build_set_predicate(idx, literals, true)
        } else {
            // A missing column, treating it as null.
            self.build_always_true()
        }
    }
}

/// Literal count above which `IN` and `NOT IN` switch from one comparison
/// pass per literal to a hash lookup per row.
pub(crate) const IN_SET_THRESHOLD: usize = 8;

/// The literals of an `IN` predicate as a hash set over one column type.
enum InSet {
    Int32(FnvHashSet<i32>),
    Int64(FnvHashSet<i64>),
    Date32(FnvHashSet<i32>),
    TimestampMicrosecond(FnvHashSet<i64>, DataType),
    TimestampNanosecond(FnvHashSet<i64>, DataType),
    Time64Microsecond(FnvHashSet<i64>),
    Utf8(FnvHashSet<String>),
    LargeUtf8(FnvHashSet<String>),
    Utf8View(FnvHashSet<String>),
}

impl InSet {
    /// Builds the set for `column_type`, casting each literal to it first.
    /// `None` keeps the comparison kernels for unsupported types and null casts.
    fn build(
        column_type: &DataType,
        literals: &[Arc<dyn ArrowDatum + Send + Sync>],
    ) -> std::result::Result<Option<Self>, ArrowError> {
        macro_rules! collect {
            ($variant:ident, $array_type:ty $(, $data_type:expr)?) => {{
                let mut set = FnvHashSet::default();
                for literal in literals {
                    let literal = try_cast_literal(literal, column_type)?;
                    let (array, _) = literal.get();
                    // A narrowing cast may produce null. The comparison
                    // kernels propagate that null through the whole mask;
                    // ignoring the literal would change their semantics.
                    if array.is_null(0) {
                        return Ok(None);
                    }
                    set.insert(array.as_primitive::<$array_type>().value(0));
                }
                Ok(Some(Self::$variant(set $(, $data_type)?)))
            }};
        }
        match column_type {
            DataType::Int32 => collect!(Int32, Int32Type),
            DataType::Int64 => collect!(Int64, Int64Type),
            DataType::Date32 => collect!(Date32, Date32Type),
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                collect!(
                    TimestampMicrosecond,
                    TimestampMicrosecondType,
                    column_type.clone()
                )
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                collect!(
                    TimestampNanosecond,
                    TimestampNanosecondType,
                    column_type.clone()
                )
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                collect!(Time64Microsecond, Time64MicrosecondType)
            }
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                let mut set = FnvHashSet::default();
                for literal in literals {
                    let literal = try_cast_literal(literal, column_type)?;
                    let (array, _) = literal.get();
                    if !array.is_null(0) {
                        let value = match column_type {
                            DataType::Utf8 => array.as_string::<i32>().value(0),
                            DataType::LargeUtf8 => array.as_string::<i64>().value(0),
                            DataType::Utf8View => array.as_string_view().value(0),
                            _ => unreachable!(),
                        };
                        set.insert(value.to_string());
                    }
                }
                Ok(Some(match column_type {
                    DataType::Utf8 => Self::Utf8(set),
                    DataType::LargeUtf8 => Self::LargeUtf8(set),
                    DataType::Utf8View => Self::Utf8View(set),
                    _ => unreachable!(),
                }))
            }
            _ => Ok(None),
        }
    }

    /// Membership of each row, null where the row is null, like the comparison kernels.
    /// `None` when `column` is not the type the set was built for.
    fn mask(&self, column: &ArrayRef) -> Option<BooleanArray> {
        match (self, column.data_type()) {
            (Self::Int32(set), DataType::Int32) => Some(
                column
                    .as_primitive::<Int32Type>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(&value)))
                    .collect(),
            ),
            (Self::Int64(set), DataType::Int64) => Some(
                column
                    .as_primitive::<Int64Type>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(&value)))
                    .collect(),
            ),
            (Self::Date32(set), DataType::Date32) => Some(
                column
                    .as_primitive::<Date32Type>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(&value)))
                    .collect(),
            ),
            (
                Self::TimestampMicrosecond(set, data_type),
                DataType::Timestamp(TimeUnit::Microsecond, _),
            ) if data_type == column.data_type() => Some(
                column
                    .as_primitive::<TimestampMicrosecondType>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(&value)))
                    .collect(),
            ),
            (
                Self::TimestampNanosecond(set, data_type),
                DataType::Timestamp(TimeUnit::Nanosecond, _),
            ) if data_type == column.data_type() => Some(
                column
                    .as_primitive::<TimestampNanosecondType>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(&value)))
                    .collect(),
            ),
            (Self::Time64Microsecond(set), DataType::Time64(TimeUnit::Microsecond)) => Some(
                column
                    .as_primitive::<Time64MicrosecondType>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(&value)))
                    .collect(),
            ),
            (Self::Utf8(set), DataType::Utf8) => Some(
                column
                    .as_string::<i32>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(value)))
                    .collect(),
            ),
            (Self::LargeUtf8(set), DataType::LargeUtf8) => Some(
                column
                    .as_string::<i64>()
                    .iter()
                    .map(|value| value.map(|value| set.contains(value)))
                    .collect(),
            ),
            (Self::Utf8View(set), DataType::Utf8View) => Some(
                column
                    .as_string_view()
                    .iter()
                    .map(|value| value.map(|value| set.contains(value)))
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// Evaluate comparisons after the same lossless numeric widening as the
/// output transformer. Narrowing a table literal to an old physical column
/// can overflow, round, or become null and reject valid rows. Set predicates
/// call this once per batch rather than casting once per literal.
fn promote_column_for_literal(
    column: ArrayRef,
    literal_type: &DataType,
) -> std::result::Result<ArrayRef, ArrowError> {
    // Parquet may restore the Arrow dictionary type from the embedded
    // schema. Compare the dictionary's values in the table type, preserving
    // both dictionary and value nulls when flattening.
    let mut source_type = column.data_type();
    while let DataType::Dictionary(_, value_type) = source_type {
        source_type = value_type;
    }
    let decimal = |data_type: &DataType| match data_type {
        DataType::Decimal32(precision, scale)
        | DataType::Decimal64(precision, scale)
        | DataType::Decimal128(precision, scale)
        | DataType::Decimal256(precision, scale) => Some((*precision, *scale)),
        _ => None,
    };
    let promote = match (source_type, literal_type) {
        // These conversions are infallible over the complete source domain.
        (
            DataType::Int8 | DataType::Int16 | DataType::UInt8 | DataType::UInt16,
            DataType::Int32 | DataType::Int64,
        )
        | (DataType::Int32 | DataType::UInt32, DataType::Int64)
        | (DataType::Float32, DataType::Float64) => true,
        _ => match (decimal(source_type), decimal(literal_type)) {
            (Some((source_precision, source_scale)), Some((target_precision, target_scale))) => {
                source_scale == target_scale
                    && source_precision <= target_precision
                    && source_type != literal_type
            }
            _ => source_type == literal_type && source_type != column.data_type(),
        },
    };
    if promote {
        cast(&column, literal_type)
    } else {
        Ok(column)
    }
}

/// The Arrow compute kernels that we use must match the type exactly, so first cast the literal
/// into the type of the batch we read from Parquet before sending it to the compute kernel.
fn try_cast_literal(
    literal: &Arc<dyn ArrowDatum + Send + Sync>,
    column_type: &DataType,
) -> std::result::Result<Arc<dyn ArrowDatum + Send + Sync>, ArrowError> {
    let literal_array = literal.get().0;

    // No cast required
    if literal_array.data_type() == column_type {
        return Ok(Arc::clone(literal));
    }

    let literal_array = cast(literal_array, column_type)?;
    Ok(Arc::new(Scalar::new(literal_array)))
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use arrow_array::{Array, BooleanArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::schema::parser::parse_message_type;
    use parquet::schema::types::SchemaDescriptor;

    use super::{CollectFieldIdVisitor, PredicateConverter, constant_bool_array};
    use crate::expr::visitors::bound_predicate_visitor::visit;
    use crate::expr::{Bind, Predicate, Reference};
    use crate::spec::{NestedField, PrimitiveType, Schema, SchemaRef, Type};

    fn table_schema_simple() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_identifier_field_ids(vec![2])
                .with_fields(vec![
                    NestedField::optional(1, "foo", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "bar", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(3, "baz", Type::Primitive(PrimitiveType::Boolean)).into(),
                    NestedField::optional(4, "qux", Type::Primitive(PrimitiveType::Float)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_collect_field_id() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux").is_null();
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_collect_field_id_with_and() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux")
            .is_null()
            .and(Reference::new("baz").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);
        expected.insert(3);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_collect_field_id_with_or() {
        let schema = table_schema_simple();
        let expr = Reference::new("qux")
            .is_null()
            .or(Reference::new("baz").is_null());
        let bound_expr = expr.bind(schema, true).unwrap();

        let mut visitor = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut visitor, &bound_expr).unwrap();

        let mut expected = HashSet::default();
        expected.insert(4_i32);
        expected.insert(3);

        assert_eq!(visitor.field_ids, expected);
    }

    #[test]
    fn test_constant_bool_array() {
        for len in [0, 8192] {
            let all_true = constant_bool_array(true, len);
            assert_eq!(all_true.len(), len);
            assert_eq!(all_true.null_count(), 0);
            assert!(all_true.iter().all(|v| v == Some(true)));

            let all_false = constant_bool_array(false, len);
            assert_eq!(all_false.len(), len);
            assert_eq!(all_false.null_count(), 0);
            assert!(all_false.iter().all(|v| v == Some(false)));
        }
    }

    fn apply_predicate_to_batch(
        predicate: Predicate,
        schema: SchemaRef,
        batch: RecordBatch,
    ) -> BooleanArray {
        let bound = predicate.bind(schema, true).unwrap();

        // Build a trivial Parquet schema with one float column at field id 4
        let message_type = "
            message schema {
              optional float qux = 4;
            }
        ";
        let parquet_type = parse_message_type(message_type).expect("parse schema");
        let parquet_schema = SchemaDescriptor::new(Arc::new(parquet_type));

        let column_map = HashMap::from([(4i32, 0usize)]);
        let column_indices = vec![0usize];

        let mut converter = PredicateConverter {
            parquet_schema: &parquet_schema,
            column_map: &column_map,
            column_indices: &column_indices,
        };

        let mut predicate_fn = visit(&mut converter, &bound).unwrap();
        predicate_fn(batch).unwrap()
    }

    #[test]
    fn test_predicate_converter_nan() {
        use arrow_array::Float32Array;

        let schema = table_schema_simple();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "qux",
            DataType::Float32,
            true,
        )]));
        let values = vec![Some(1.0f32), Some(f32::NAN), None, Some(0.0f32)];

        // is_nan: non-null-propagating per Java's implementation - NULL → false
        let batch = RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(Float32Array::from(
            values.clone(),
        ))])
        .unwrap();
        let result =
            apply_predicate_to_batch(Reference::new("qux").is_nan(), schema.clone(), batch);
        assert_eq!(
            [
                result.value(0),
                result.value(1),
                result.value(2),
                result.value(3)
            ],
            [false, true, false, false]
        );
        assert!(!result.is_null(2));

        // not_nan: non-null-propagating per Java's implementation - NULL → true
        let batch =
            RecordBatch::try_new(arrow_schema, vec![Arc::new(Float32Array::from(values))]).unwrap();
        let result = apply_predicate_to_batch(Reference::new("qux").is_not_nan(), schema, batch);
        assert_eq!(
            [
                result.value(0),
                result.value(1),
                result.value(2),
                result.value(3)
            ],
            [true, false, true, true]
        );
        assert!(!result.is_null(2));
    }

    #[test]
    fn in_set_matches_the_comparison_kernels() {
        use arrow_array::{ArrayRef, Datum as ArrowDatum, Int64Array, StringArray};

        use super::InSet;
        use crate::arrow::get_arrow_datum;
        use crate::spec::Datum;

        let longs: Vec<Arc<dyn ArrowDatum + Send + Sync>> = (0..20)
            .map(|value| get_arrow_datum(&Datum::long(value * 3)).unwrap())
            .collect();
        let column: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(0),
            Some(1),
            None,
            Some(57),
            Some(58),
        ]));
        let set = InSet::build(&DataType::Int64, &longs).unwrap().unwrap();
        assert_eq!(
            set.mask(&column).unwrap(),
            BooleanArray::from(vec![Some(true), Some(false), None, Some(true), Some(false)])
        );
        // A set built for one column type never answers for another.
        let other: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        assert!(set.mask(&other).is_none());

        let strings: Vec<Arc<dyn ArrowDatum + Send + Sync>> = ["k1", "k2", "k3"]
            .iter()
            .map(|value| get_arrow_datum(&Datum::string(*value)).unwrap())
            .collect();
        let column: ArrayRef = Arc::new(StringArray::from(vec![Some("k2"), None, Some("k9")]));
        let set = InSet::build(&DataType::Utf8, &strings).unwrap().unwrap();
        assert_eq!(
            set.mask(&column).unwrap(),
            BooleanArray::from(vec![Some(true), None, Some(false)])
        );
        // Types without a hash path keep the comparison kernels.
        assert!(InSet::build(&DataType::Float64, &longs).unwrap().is_none());
    }

    #[test]
    fn membership_predicates_match_kernels_for_small_and_large_sets() {
        use arrow_array::{
            ArrayRef, Int32Array, Int64Array, StringArray, Time64MicrosecondArray,
            TimestampMicrosecondArray, TimestampNanosecondArray,
        };
        use arrow_cast::cast;
        use fnv::FnvHashSet;

        use super::{and, eq, neq, or, promote_column_for_literal, try_cast_literal};
        use crate::arrow::get_arrow_datum;
        use crate::spec::Datum;

        let parquet_schema = SchemaDescriptor::new(Arc::new(
            parse_message_type("message schema { optional int64 value = 1; }").unwrap(),
        ));
        let column_map = HashMap::from([(1, 0)]);
        let column_indices = vec![0];
        let converter = PredicateConverter {
            parquet_schema: &parquet_schema,
            column_map: &column_map,
            column_indices: &column_indices,
        };
        let strings: ArrayRef = Arc::new(StringArray::from(vec![
            Some("key-1-冰"),
            None,
            Some("key-19-冰"),
            Some("absent"),
        ]));
        let longs: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(19), Some(-1)]));
        let narrowed: ArrayRef =
            Arc::new(Int32Array::from(vec![Some(1), None, Some(19), Some(-1)]));
        let cases = [
            (longs, (0..20).map(Datum::long).collect::<Vec<_>>()),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(1),
                    None,
                    Some(19),
                    Some(-1),
                ])) as ArrayRef,
                (0..20).map(Datum::timestamp_micros).collect(),
            ),
            (
                Arc::new(
                    TimestampMicrosecondArray::from(vec![Some(1), None, Some(19), Some(-1)])
                        .with_timezone("UTC"),
                ) as ArrayRef,
                (0..20).map(Datum::timestamptz_micros).collect(),
            ),
            (
                Arc::new(
                    TimestampMicrosecondArray::from(vec![Some(1), None, Some(19), Some(-1)])
                        .with_timezone("+00:00"),
                ) as ArrayRef,
                (0..20).map(Datum::timestamptz_micros).collect(),
            ),
            (
                Arc::new(TimestampNanosecondArray::from(vec![
                    Some(1),
                    None,
                    Some(19),
                    Some(-1),
                ])) as ArrayRef,
                (0..20).map(Datum::timestamp_nanos).collect(),
            ),
            (
                Arc::new(Time64MicrosecondArray::from(vec![
                    Some(1),
                    None,
                    Some(19),
                    Some(0),
                ])) as ArrayRef,
                (0..20)
                    .map(|value| Datum::time_micros(value).unwrap())
                    .collect(),
            ),
            (
                narrowed,
                std::iter::once(Datum::long(i64::from(i32::MAX) + 1))
                    .chain((1..20).map(Datum::long))
                    .collect(),
            ),
            (
                strings.clone(),
                (0..20)
                    .map(|value| Datum::string(format!("key-{value}-冰")))
                    .collect(),
            ),
            (
                cast(&strings, &DataType::LargeUtf8).unwrap(),
                (0..20)
                    .map(|value| Datum::string(format!("key-{value}-冰")))
                    .collect(),
            ),
            (
                cast(&strings, &DataType::Utf8View).unwrap(),
                (0..20)
                    .map(|value| Datum::string(format!("key-{value}-冰")))
                    .collect(),
            ),
        ];
        for (column, literals) in cases {
            for count in [5, 20] {
                let literals: FnvHashSet<_> = literals.iter().take(count).cloned().collect();
                for negate in [false, true] {
                    let mut expected = constant_bool_array(negate, column.len());
                    for literal in &literals {
                        let literal = get_arrow_datum(literal).unwrap();
                        let promoted =
                            promote_column_for_literal(column.clone(), literal.get().0.data_type())
                                .unwrap();
                        let literal = try_cast_literal(&literal, promoted.data_type()).unwrap();
                        expected = if negate {
                            and(&expected, &neq(&promoted, literal.as_ref()).unwrap()).unwrap()
                        } else {
                            or(&expected, &eq(&promoted, literal.as_ref()).unwrap()).unwrap()
                        };
                    }
                    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                        "value",
                        column.data_type().clone(),
                        true,
                    )]));
                    let batch = RecordBatch::try_new(schema, vec![column.clone()]).unwrap();
                    let mut predicate =
                        converter.build_set_predicate(0, &literals, negate).unwrap();
                    // Exercise both first-use set compilation and cached reuse.
                    for _ in 0..2 {
                        assert_eq!(
                            predicate(batch.clone()).unwrap(),
                            expected,
                            "{} with {count} literals, negate={negate}",
                            column.data_type()
                        );
                    }
                    assert!(expected.is_null(1));
                }
            }
        }
    }
}
