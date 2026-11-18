// Added by the Longbow authors (family AJ); see third_party/README.md.

// SPDX-License-Identifier: Apache-2.0

//! DuckDB's ASOF join forms, rewritten to the form that `sqlparser` parses.
//!
//! `sqlparser` accepts only Snowflake's form,
//! `l ASOF JOIN r MATCH_CONDITION (l.t >= r.t) [ON ... | USING (...)]`, in
//! which every left row is kept. DuckDB writes
//!
//! ```text
//! l ASOF [INNER | LEFT [OUTER]] JOIN r ON l.k = r.k AND l.t >= r.t
//! l ASOF [INNER | LEFT [OUTER]] JOIN r USING (k, ..., t)
//! ```
//!
//! with the inequality inside `ON`, or as the last `USING` column, and with
//! inner semantics unless `LEFT` is written. [`rewrite_duckdb_asof`] runs on
//! the token stream before the parser: after `ASOF [INNER | LEFT [OUTER]] JOIN
//! <relation>`, when the next token is `ON` or `USING` (not
//! `MATCH_CONDITION`), it removes the join kind and inserts
//! `MATCH_CONDITION (<marker>)`, where the marker is an identifier that names
//! the join kind ([`ASOF_INNER_MARKER`], [`ASOF_LEFT_MARKER`]). The SQL planner
//! recognizes the marker and derives the ordered comparison and the equality
//! keys from the constraint, with the schemas of both sides in hand. A
//! relation without an alias gets its own last name part as an alias, so the
//! parser does not read `MATCH_CONDITION` as the alias.
//!
//! The pass needs no schema and leaves every other token unchanged. It is
//! removed when `sqlparser` parses DuckDB's form.

use sqlparser::ast::TableFactor;
use sqlparser::dialect::Dialect;
use sqlparser::dialect::keywords::Keyword;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Token, TokenWithSpan, Whitespace};

/// Marker of DuckDB's `ASOF [INNER] JOIN`: unmatched left rows are dropped.
pub const ASOF_INNER_MARKER: &str = "__datafusion_asof_inner";
/// Marker of DuckDB's `ASOF LEFT [OUTER] JOIN`: every left row is kept.
pub const ASOF_LEFT_MARKER: &str = "__datafusion_asof_left";

fn is_keyword(token: &Token, keyword: Keyword) -> bool {
    matches!(token, Token::Word(word) if word.quote_style.is_none() && word.keyword == keyword)
}

fn is_blank(token: &Token) -> bool {
    matches!(token, Token::Whitespace(_))
}

/// The index of the first token at or after `from` that is not whitespace.
fn significant(tokens: &[TokenWithSpan], from: usize) -> Option<usize> {
    (from..tokens.len()).find(|&i| !is_blank(&tokens[i].token))
}

/// Rewrites every DuckDB-form ASOF join in `tokens` to Snowflake's form with
/// a marker (see the module documentation). Tokens of other SQL, and of
/// Snowflake's form, are returned unchanged.
pub fn rewrite_duckdb_asof(
    dialect: &dyn Dialect,
    mut tokens: Vec<TokenWithSpan>,
) -> Vec<TokenWithSpan> {
    let starts: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| is_keyword(&t.token, Keyword::ASOF))
        .map(|(i, _)| i)
        .collect();
    // From the last ASOF to the first: the relation of a join contains only
    // ASOF joins that start after it, which are then already rewritten.
    for &start in starts.iter().rev() {
        rewrite_one(dialect, &mut tokens, start);
    }
    tokens
}

fn rewrite_one(dialect: &dyn Dialect, tokens: &mut Vec<TokenWithSpan>, start: usize) {
    let Some(mut next) = significant(tokens, start + 1) else {
        return;
    };
    let mut kind_tokens = Vec::new();
    let mut inner = true;
    if is_keyword(&tokens[next].token, Keyword::INNER) {
        kind_tokens.push(next);
    } else if is_keyword(&tokens[next].token, Keyword::LEFT) {
        inner = false;
        kind_tokens.push(next);
        let Some(after) = significant(tokens, next + 1) else {
            return;
        };
        if is_keyword(&tokens[after].token, Keyword::OUTER) {
            kind_tokens.push(after);
            next = after;
        }
    }
    if !kind_tokens.is_empty() {
        let Some(after) = significant(tokens, next + 1) else {
            return;
        };
        next = after;
    }
    if !is_keyword(&tokens[next].token, Keyword::JOIN) {
        return;
    }
    let relation_start = next + 1;
    let mut parser = Parser::new(dialect)
        .with_tokens_with_locations(tokens[relation_start..].to_vec());
    let Ok(relation) = parser.parse_table_factor() else {
        return;
    };
    let relation_end = relation_start + parser.index();
    let Some(constraint) = significant(tokens, relation_end) else {
        return;
    };
    if !is_keyword(&tokens[constraint].token, Keyword::ON)
        && !is_keyword(&tokens[constraint].token, Keyword::USING)
    {
        return;
    }

    let span = tokens[constraint].span;
    let mut insert = Vec::new();
    let mut push = |token: Token| insert.push(TokenWithSpan::new(token, span));
    if let TableFactor::Table {
        name, alias: None, ..
    } = &relation
        && let Some(last) = name.0.last().and_then(|part| part.as_ident())
    {
        push(Token::make_keyword("AS"));
        push(Token::Whitespace(Whitespace::Space));
        push(Token::make_word(&last.value, last.quote_style));
        push(Token::Whitespace(Whitespace::Space));
    }
    push(Token::make_keyword("MATCH_CONDITION"));
    push(Token::Whitespace(Whitespace::Space));
    push(Token::LParen);
    push(Token::make_word(
        if inner {
            ASOF_INNER_MARKER
        } else {
            ASOF_LEFT_MARKER
        },
        None,
    ));
    push(Token::RParen);
    push(Token::Whitespace(Whitespace::Space));
    tokens.splice(constraint..constraint, insert);
    // The kind keywords are before the insertion point, so their indices hold.
    for index in kind_tokens {
        tokens[index].token = Token::Whitespace(Whitespace::Space);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlparser::dialect::GenericDialect;
    use sqlparser::tokenizer::Tokenizer;

    fn rewrite(sql: &str) -> String {
        let dialect = GenericDialect {};
        let tokens = Tokenizer::new(&dialect, sql)
            .tokenize_with_location()
            .unwrap();
        let out = rewrite_duckdb_asof(&dialect, tokens);
        let text: String = out.iter().map(|t| t.token.to_string()).collect();
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn left_on_and_using() {
        assert_eq!(
            rewrite("SELECT * FROM l ASOF LEFT JOIN r ON l.k = r.k AND l.t >= r.t"),
            "SELECT * FROM l ASOF JOIN r AS r MATCH_CONDITION (__datafusion_asof_left) ON l.k = r.k AND l.t >= r.t"
        );
        assert_eq!(
            rewrite(
                "SELECT * FROM l ASOF LEFT OUTER JOIN (SELECT * FROM x) q USING (k, t)"
            ),
            "SELECT * FROM l ASOF JOIN (SELECT * FROM x) q MATCH_CONDITION (__datafusion_asof_left) USING (k, t)"
        );
    }

    #[test]
    fn inner_forms_and_unaliased_tables() {
        assert_eq!(
            rewrite("FROM l ASOF JOIN s.timeBuckets ON l.t >= timeBuckets.bound"),
            "FROM l ASOF JOIN s.timeBuckets AS timeBuckets MATCH_CONDITION (__datafusion_asof_inner) ON l.t >= timeBuckets.bound"
        );
        assert_eq!(
            rewrite(r#"FROM l ASOF INNER JOIN "R" USING (t)"#),
            r#"FROM l ASOF JOIN "R" AS "R" MATCH_CONDITION (__datafusion_asof_inner) USING (t)"#
        );
    }

    #[test]
    fn nested_joins_rewrite_inside_out() {
        assert_eq!(
            rewrite(
                "FROM a ASOF LEFT JOIN (SELECT * FROM b ASOF JOIN c ON b.t > c.t) x ON a.t >= x.t"
            ),
            "FROM a ASOF JOIN (SELECT * FROM b ASOF JOIN c AS c MATCH_CONDITION (__datafusion_asof_inner) ON b.t > c.t) x MATCH_CONDITION (__datafusion_asof_left) ON a.t >= x.t"
        );
    }

    #[test]
    fn other_sql_is_unchanged() {
        for sql in [
            "SELECT * FROM l ASOF JOIN r MATCH_CONDITION (l.t >= r.t) ON l.k = r.k",
            "SELECT asof FROM t",
            "SELECT 'ASOF LEFT JOIN r ON' FROM t",
            "SELECT * FROM l ASOF LEFT JOIN r",
            "SELECT * FROM l LEFT JOIN r ON l.k = r.k",
        ] {
            assert_eq!(
                rewrite(sql),
                sql.split_whitespace().collect::<Vec<_>>().join(" ")
            );
        }
    }
}
