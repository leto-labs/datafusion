// Modified by the Longbow authors; see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Optional object validators preserve presence on the native protobuf wire.
use crate::protobuf::PartitionedFile;
use prost::Message;

#[test]
fn identity_wire_roundtrip_preserves_absent_empty_and_present_values() {
    for (e_tag, version) in [
        (None, None),
        (Some(""), Some("")),
        (Some("\"etag-2\""), None),
        (None, Some("version/with+escapes=")),
        (Some("W/weak"), Some("null")),
    ] {
        let file = PartitionedFile {
            path: "data.parquet".into(),
            size: 1234,
            last_modified_ns: 1_234_567_891,
            e_tag: e_tag.map(str::to_owned),
            version: version.map(str::to_owned),
            ..Default::default()
        };
        let bytes = file.encode_to_vec();
        assert_eq!(PartitionedFile::decode(bytes.as_slice()).unwrap(), file);
    }
}

/// The validators use the fork's reserved tag range (900 and 901), so a
/// later upstream field cannot collide with them.
#[test]
fn identity_wire_roundtrip_uses_reserved_fork_tags() {
    let only = |e_tag: Option<&str>, version: Option<&str>| {
        PartitionedFile {
            e_tag: e_tag.map(str::to_owned),
            version: version.map(str::to_owned),
            ..Default::default()
        }
        .encode_to_vec()
    };
    // Key varints: (900 << 3) | 2 = 7202 and (901 << 3) | 2 = 7210.
    assert_eq!(only(Some("e"), None), vec![0xa2, 0x38, 1, b'e']);
    assert_eq!(only(None, Some("v")), vec![0xaa, 0x38, 1, b'v']);
}
