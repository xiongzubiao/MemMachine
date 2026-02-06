use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterExpr {
    Comparison(Comparison),
    And {
        left: Box<FilterExpr>,
        right: Box<FilterExpr>,
    },
    Or {
        left: Box<FilterExpr>,
        right: Box<FilterExpr>,
    },
}

impl FilterExpr {
    pub fn and(self, other: FilterExpr) -> Self {
        FilterExpr::And {
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    pub fn or(self, other: FilterExpr) -> Self {
        FilterExpr::Or {
            left: Box::new(self),
            right: Box::new(other),
        }
    }

    pub fn eq(field: impl Into<String>, value: impl Into<FilterValue>) -> Self {
        FilterExpr::Comparison(Comparison {
            field: field.into(),
            op: ComparisonOp::Eq,
            value: value.into(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub field: String,
    pub op: ComparisonOp,
    pub value: FilterValue,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,
    In,
    Gt,
    Lt,
    Gte,
    Lte,
    IsNull,
    IsNotNull,
}

impl ComparisonOp {
    pub fn as_sql(&self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::In => "IN",
            Self::Gt => ">",
            Self::Lt => "<",
            Self::Gte => ">=",
            Self::Lte => "<=",
            Self::IsNull => "IS NULL",
            Self::IsNotNull => "IS NOT NULL",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<FilterValue>),
    Null,
}

impl From<String> for FilterValue {
    fn from(s: String) -> Self {
        FilterValue::String(s)
    }
}

impl From<&str> for FilterValue {
    fn from(s: &str) -> Self {
        FilterValue::String(s.to_string())
    }
}

impl From<i64> for FilterValue {
    fn from(i: i64) -> Self {
        FilterValue::Int(i)
    }
}

impl From<i32> for FilterValue {
    fn from(i: i32) -> Self {
        FilterValue::Int(i64::from(i))
    }
}

impl From<f64> for FilterValue {
    fn from(f: f64) -> Self {
        FilterValue::Float(f)
    }
}

impl From<bool> for FilterValue {
    fn from(b: bool) -> Self {
        FilterValue::Bool(b)
    }
}

impl<T: Into<FilterValue>> From<Vec<T>> for FilterValue {
    fn from(v: Vec<T>) -> Self {
        FilterValue::List(v.into_iter().map(Into::into).collect())
    }
}
