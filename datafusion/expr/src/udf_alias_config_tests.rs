// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Configuration refresh through scalar UDF alias wrappers.

use super::*;
use crate::ptr_eq::PtrEq;
use arrow::datatypes::TimeUnit;
use datafusion_expr_common::signature::Volatility;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConfiguredScalar {
    signature: Signature,
    parameter: i64,
    time_zone: Option<String>,
    aliases: Vec<String>,
    updates_enabled: bool,
    calls: PtrEq<Arc<AtomicUsize>>,
}

impl ConfiguredScalar {
    fn new(parameter: i64, updates_enabled: bool, calls: Arc<AtomicUsize>) -> Self {
        Self {
            signature: Signature::nullary(Volatility::Stable),
            parameter,
            time_zone: Some("UTC".into()),
            aliases: vec!["native_UTC".into()],
            updates_enabled,
            calls: calls.into(),
        }
    }
}

impl ScalarUDFImpl for ConfiguredScalar {
    fn name(&self) -> &str {
        "configured_scalar"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Timestamp(
            TimeUnit::Nanosecond,
            self.time_zone.clone().map(Into::into),
        ))
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !self.updates_enabled
            || self.time_zone.as_deref()
                == config.execution.time_zone.as_ref().map(|tz| tz.as_str())
        {
            return None;
        }
        let mut updated = self.clone();
        updated.time_zone = config.execution.time_zone.as_ref().map(ToString::to_string);
        updated.aliases = vec![format!(
            "native_{}",
            updated.time_zone.as_deref().unwrap_or("none")
        )];
        Some(ScalarUDF::from(updated))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Scalar(ScalarValue::TimestampNanosecond(
            Some(self.parameter),
            self.time_zone.clone().map(Into::into),
        )))
    }
}

fn config(time_zone: Option<&str>) -> ConfigOptions {
    let mut config = ConfigOptions::default();
    config.execution.time_zone = time_zone.map(|tz| tz.parse().unwrap());
    config
}

#[test]
fn refresh_preserves_custom_parameters_and_wrapper_aliases() -> Result<()> {
    for parameter in [17, 91] {
        let calls = Arc::new(AtomicUsize::new(0));
        let original =
            ScalarUDF::from(ConfiguredScalar::new(parameter, true, Arc::clone(&calls)))
                .with_aliases(["extra", "extra"]);
        let config = config(Some("America/Los_Angeles"));
        let updated = original
            .inner()
            .with_updated_config(&config)
            .expect("the wrapped function changes with the timezone");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(updated.name(), original.name());
        assert_eq!(updated.signature(), original.signature());
        assert_eq!(
            updated.aliases(),
            ["native_America/Los_Angeles", "extra", "extra"]
        );
        let expected_type =
            DataType::Timestamp(TimeUnit::Nanosecond, Some("America/Los_Angeles".into()));
        assert_eq!(updated.return_type(&[])?, expected_type);
        let value = updated.invoke_with_args(ScalarFunctionArgs {
            args: vec![],
            arg_fields: vec![],
            number_rows: 1,
            return_field: Arc::new(Field::new("result", expected_type, true)),
            config_options: Arc::new(config),
        })?;
        let ColumnarValue::Scalar(value) = value else {
            panic!("expected a scalar result");
        };
        assert_eq!(
            value,
            ScalarValue::TimestampNanosecond(
                Some(parameter),
                Some("America/Los_Angeles".into())
            )
        );
        assert_eq!(original.aliases(), ["native_UTC", "extra", "extra"]);
        assert_eq!(
            original.return_type(&[])?,
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
        );
    }
    Ok(())
}

#[test]
fn refresh_none_forwards_hook_and_preserves_caller_identity() {
    for (updates_enabled, time_zone) in [(false, "Asia/Tokyo"), (true, "UTC")] {
        let calls = Arc::new(AtomicUsize::new(0));
        let original = Arc::new(
            ScalarUDF::from(ConfiguredScalar::new(
                17,
                updates_enabled,
                Arc::clone(&calls),
            ))
            .with_aliases(["extra"]),
        );
        let replacement = original
            .inner()
            .with_updated_config(&config(Some(time_zone)));
        assert!(replacement.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let retained = replacement
            .map(Arc::new)
            .unwrap_or_else(|| Arc::clone(&original));
        assert!(Arc::ptr_eq(&original, &retained));
        assert!(Arc::ptr_eq(original.inner(), retained.inner()));
        assert_eq!(retained.aliases(), ["native_UTC", "extra"]);
    }
}

#[test]
fn repeated_refresh_replaces_inner_aliases_without_accumulating_wrappers() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut function =
        ScalarUDF::from(ConfiguredScalar::new(17, true, Arc::clone(&calls)))
            .with_aliases(["native_UTC", "shared"])
            .with_aliases(["outer", "shared"]);
    for (index, time_zone) in [Some("Europe/London"), None, Some("UTC")]
        .into_iter()
        .enumerate()
    {
        function = function
            .inner()
            .with_updated_config(&config(time_zone))
            .expect("each timezone changes the wrapped function");
        assert_eq!(calls.load(Ordering::SeqCst), index + 1);
        // An explicitly added duplicate of an old native alias is still an alias.
        assert_eq!(
            function.aliases(),
            [
                format!("native_{}", time_zone.unwrap_or("none")),
                "native_UTC".into(),
                "shared".into(),
                "outer".into(),
                "shared".into(),
            ]
        );
        assert_eq!(
            function.return_type(&[]).unwrap(),
            DataType::Timestamp(TimeUnit::Nanosecond, time_zone.map(Into::into))
        );
    }
}
