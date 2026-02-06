//! Python type wrappers for MemMachine core types.
//!
//! These types mirror the Rust types from memmachine-core but with PyO3 bindings.

use memmachine_core::{
    Episode, EpisodeEntry, EpisodeRole, FeatureMetadata, FilterExpr, MemoryType, SemanticFeature,
    Session, SessionData, SessionInfo,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};

/// Python wrapper for EpisodeRole
#[pyclass(name = "EpisodeRole", eq)]
#[derive(Debug, Clone, PartialEq)]
pub enum PyEpisodeRole {
    User,
    Assistant,
    System,
}

#[pymethods]
impl PyEpisodeRole {
    #[new]
    fn new(role: &str) -> PyResult<Self> {
        match role.to_lowercase().as_str() {
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "system" => Ok(Self::System),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid role: {}. Must be 'user', 'assistant', or 'system'",
                role
            ))),
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }

    fn __repr__(&self) -> String {
        format!("EpisodeRole.{}", self.__str__())
    }
}

impl From<EpisodeRole> for PyEpisodeRole {
    fn from(role: EpisodeRole) -> Self {
        match role {
            EpisodeRole::User => Self::User,
            EpisodeRole::Assistant => Self::Assistant,
            EpisodeRole::System => Self::System,
        }
    }
}

impl From<PyEpisodeRole> for EpisodeRole {
    fn from(role: PyEpisodeRole) -> Self {
        match role {
            PyEpisodeRole::User => Self::User,
            PyEpisodeRole::Assistant => Self::Assistant,
            PyEpisodeRole::System => Self::System,
        }
    }
}

/// Python wrapper for MemoryType
#[pyclass(name = "MemoryType", eq)]
#[derive(Debug, Clone, PartialEq)]
pub enum PyMemoryType {
    Episodic,
    Semantic,
}

#[pymethods]
impl PyMemoryType {
    #[new]
    fn new(memory_type: &str) -> PyResult<Self> {
        match memory_type.to_lowercase().as_str() {
            "episodic" => Ok(Self::Episodic),
            "semantic" => Ok(Self::Semantic),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid memory type: {}. Must be 'episodic' or 'semantic'",
                memory_type
            ))),
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
        }
    }

    fn __repr__(&self) -> String {
        format!("MemoryType.{}", self.__str__())
    }
}

impl From<MemoryType> for PyMemoryType {
    fn from(mt: MemoryType) -> Self {
        match mt {
            MemoryType::Episodic => Self::Episodic,
            MemoryType::Semantic => Self::Semantic,
        }
    }
}

impl From<PyMemoryType> for MemoryType {
    fn from(mt: PyMemoryType) -> Self {
        match mt {
            PyMemoryType::Episodic => Self::Episodic,
            PyMemoryType::Semantic => Self::Semantic,
        }
    }
}

/// Python wrapper for EpisodeEntry
#[pyclass(name = "EpisodeEntry")]
#[derive(Debug)]
pub struct PyEpisodeEntry {
    #[pyo3(get, set)]
    pub role: PyEpisodeRole,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub name: Option<String>,
    #[pyo3(get, set)]
    pub metadata: Option<PyObject>,
}

#[pymethods]
impl PyEpisodeEntry {
    #[new]
    #[pyo3(signature = (role, content, name=None, metadata=None))]
    fn new(
        role: PyEpisodeRole,
        content: String,
        name: Option<String>,
        metadata: Option<PyObject>,
    ) -> Self {
        Self {
            role,
            content,
            name,
            metadata,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EpisodeEntry(role={}, content='{}', name={:?})",
            self.role.__str__(),
            &self.content[..self.content.len().min(50)],
            self.name
        )
    }
}

impl PyEpisodeEntry {
    pub fn to_core(&self, py: Python<'_>) -> PyResult<EpisodeEntry> {
        let metadata = if let Some(obj) = self.metadata.as_ref() {
            Some(py_to_json(py, obj.bind(py))?)
        } else {
            None
        };

        Ok(EpisodeEntry {
            role: self.role.clone().into(),
            content: self.content.clone(),
            name: self.name.clone(),
            metadata,
        })
    }
}

/// Python wrapper for Episode
#[pyclass(name = "Episode")]
#[derive(Debug, Clone)]
pub struct PyEpisode {
    #[pyo3(get)]
    pub uid: String,
    #[pyo3(get)]
    pub session_key: String,
    #[pyo3(get)]
    pub role: PyEpisodeRole,
    #[pyo3(get)]
    pub content: String,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
    metadata_value: Option<serde_json::Value>,
}

#[pymethods]
impl PyEpisode {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &self.metadata_value {
            Some(v) => json_to_py(py, v).map(Some),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Episode(uid='{}', role={}, content='{}...')",
            self.uid,
            self.role.__str__(),
            &self.content[..self.content.len().min(30)]
        )
    }
}

impl From<Episode> for PyEpisode {
    fn from(ep: Episode) -> Self {
        Self {
            uid: ep.uid,
            session_key: ep.session_key,
            role: ep.role.into(),
            content: ep.content,
            name: ep.name,
            created_at: ep.created_at.to_rfc3339(),
            updated_at: ep.updated_at.to_rfc3339(),
            metadata_value: ep.metadata,
        }
    }
}

/// Python wrapper for SessionData
#[pyclass(name = "SessionData")]
#[derive(Debug, Clone)]
pub struct PySessionData {
    #[pyo3(get, set)]
    pub session_key: String,
    #[pyo3(get, set)]
    pub user_profile_id: Option<String>,
    #[pyo3(get, set)]
    pub role_profile_id: Option<String>,
    #[pyo3(get, set)]
    pub session_id: Option<String>,
}

#[pymethods]
impl PySessionData {
    #[new]
    #[pyo3(signature = (session_key, user_profile_id=None, role_profile_id=None, session_id=None))]
    fn new(
        session_key: String,
        user_profile_id: Option<String>,
        role_profile_id: Option<String>,
        session_id: Option<String>,
    ) -> Self {
        Self {
            session_key,
            user_profile_id,
            role_profile_id,
            session_id,
        }
    }

    fn __repr__(&self) -> String {
        format!("SessionData(session_key='{}')", self.session_key)
    }
}

impl From<PySessionData> for SessionData {
    fn from(sd: PySessionData) -> Self {
        SessionData {
            session_key: sd.session_key,
            user_profile_id: sd.user_profile_id,
            role_profile_id: sd.role_profile_id,
            session_id: sd.session_id,
        }
    }
}

impl From<SessionData> for PySessionData {
    fn from(sd: SessionData) -> Self {
        Self {
            session_key: sd.session_key,
            user_profile_id: sd.user_profile_id,
            role_profile_id: sd.role_profile_id,
            session_id: sd.session_id,
        }
    }
}

/// Python wrapper for SessionInfo
#[pyclass(name = "SessionInfo")]
#[derive(Debug, Clone)]
pub struct PySessionInfo {
    #[pyo3(get)]
    pub session_key: String,
    #[pyo3(get)]
    pub description: String,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
}

#[pymethods]
impl PySessionInfo {
    fn __repr__(&self) -> String {
        format!(
            "SessionInfo(session_key='{}', description='{}')",
            self.session_key, self.description
        )
    }
}

impl From<SessionInfo> for PySessionInfo {
    fn from(si: SessionInfo) -> Self {
        Self {
            session_key: si.session_key,
            description: si.description,
            created_at: si.created_at.to_rfc3339(),
            updated_at: si.updated_at.to_rfc3339(),
        }
    }
}

/// Python wrapper for Session
#[pyclass(name = "Session")]
#[derive(Debug, Clone)]
pub struct PySession {
    #[pyo3(get)]
    pub session_key: String,
    #[pyo3(get)]
    pub description: Option<String>,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
    configuration_value: Option<serde_json::Value>,
    metadata_value: Option<serde_json::Value>,
}

#[pymethods]
impl PySession {
    #[getter]
    fn configuration(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &self.configuration_value {
            Some(v) => json_to_py(py, v).map(Some),
            None => Ok(None),
        }
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &self.metadata_value {
            Some(v) => json_to_py(py, v).map(Some),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Session(session_key='{}', description={:?})",
            self.session_key, self.description
        )
    }
}

impl From<Session> for PySession {
    fn from(s: Session) -> Self {
        Self {
            session_key: s.session_key,
            description: s.description,
            created_at: s.created_at.to_rfc3339(),
            updated_at: s.updated_at.to_rfc3339(),
            configuration_value: s.configuration,
            metadata_value: s.metadata,
        }
    }
}

/// Python wrapper for SemanticFeature
#[pyclass(name = "SemanticFeature")]
#[derive(Debug, Clone)]
pub struct PySemanticFeature {
    #[pyo3(get)]
    pub set_id: String,
    #[pyo3(get)]
    pub category: String,
    #[pyo3(get)]
    pub tag: String,
    #[pyo3(get)]
    pub feature_name: String,
    #[pyo3(get)]
    pub value: String,
    #[pyo3(get)]
    pub metadata: PyFeatureMetadata,
}

#[pymethods]
impl PySemanticFeature {
    fn __repr__(&self) -> String {
        format!(
            "SemanticFeature(category='{}', tag='{}', feature_name='{}', value='{}')",
            self.category, self.tag, self.feature_name, self.value
        )
    }
}

impl From<SemanticFeature> for PySemanticFeature {
    fn from(sf: SemanticFeature) -> Self {
        Self {
            set_id: sf.set_id,
            category: sf.category,
            tag: sf.tag,
            feature_name: sf.feature_name,
            value: sf.value,
            metadata: sf.metadata.into(),
        }
    }
}

/// Python wrapper for FeatureMetadata
#[pyclass(name = "FeatureMetadata")]
#[derive(Debug, Clone)]
pub struct PyFeatureMetadata {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub citations: Option<Vec<String>>,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
    other_value: Option<serde_json::Value>,
}

#[pymethods]
impl PyFeatureMetadata {
    #[getter]
    fn other(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &self.other_value {
            Some(v) => json_to_py(py, v).map(Some),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!("FeatureMetadata(id='{}')", self.id)
    }
}

impl From<FeatureMetadata> for PyFeatureMetadata {
    fn from(fm: FeatureMetadata) -> Self {
        Self {
            id: fm.id,
            citations: fm.citations,
            created_at: fm.created_at.to_rfc3339(),
            updated_at: fm.updated_at.to_rfc3339(),
            other_value: fm.other,
        }
    }
}

/// Python wrapper for search results
#[pyclass(name = "EpisodicSearchResult")]
#[derive(Debug, Clone)]
pub struct PyEpisodicSearchResult {
    #[pyo3(get)]
    pub episodes: Vec<PyEpisode>,
    #[pyo3(get)]
    pub scores: Vec<f32>,
}

#[pymethods]
impl PyEpisodicSearchResult {
    fn __repr__(&self) -> String {
        format!("EpisodicSearchResult(count={})", self.episodes.len())
    }

    fn __len__(&self) -> usize {
        self.episodes.len()
    }
}

/// Python wrapper for SearchResponse
#[pyclass(name = "SearchResponse")]
#[derive(Debug, Clone)]
pub struct PySearchResponse {
    #[pyo3(get)]
    pub episodic_memory: Option<PyEpisodicSearchResult>,
    #[pyo3(get)]
    pub semantic_memory: Option<Vec<(PySemanticFeature, f32)>>,
}

#[pymethods]
impl PySearchResponse {
    fn __repr__(&self) -> String {
        let episodic = self
            .episodic_memory
            .as_ref()
            .map(|e| e.episodes.len())
            .unwrap_or(0);
        let semantic = self
            .semantic_memory
            .as_ref()
            .map(|s| s.len())
            .unwrap_or(0);
        format!(
            "SearchResponse(episodic_count={}, semantic_count={})",
            episodic, semantic
        )
    }
}

/// Python wrapper for ListResults
#[pyclass(name = "ListResults")]
#[derive(Debug, Clone)]
pub struct PyListResults {
    #[pyo3(get)]
    pub episodic_memory: Option<Vec<PyEpisode>>,
    #[pyo3(get)]
    pub semantic_memory: Option<Vec<PySemanticFeature>>,
}

#[pymethods]
impl PyListResults {
    fn __repr__(&self) -> String {
        let episodic = self
            .episodic_memory
            .as_ref()
            .map(|e| e.len())
            .unwrap_or(0);
        let semantic = self
            .semantic_memory
            .as_ref()
            .map(|s| s.len())
            .unwrap_or(0);
        format!(
            "ListResults(episodic_count={}, semantic_count={})",
            episodic, semantic
        )
    }
}

// ============================================================================
// Filter Expression Helpers
// ============================================================================

/// Parse a Python dict filter into FilterExpr
pub fn parse_filter_dict(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<FilterExpr> {
    use memmachine_core::{Comparison, ComparisonOp, FilterValue};

    // Simple format: {"field": "value"} -> eq comparison
    // Extended format: {"field": {"$eq": "value"}, ...}
    // Logical: {"$and": [...], "$or": [...]}

    let mut exprs: Vec<FilterExpr> = Vec::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;

        if key_str == "$and" {
            let list: &Bound<'_, PyList> = value.downcast()?;
            let mut sub_exprs: Vec<FilterExpr> = Vec::new();
            for item in list.iter() {
                let sub_dict: &Bound<'_, PyDict> = item.downcast()?;
                sub_exprs.push(parse_filter_dict(py, sub_dict)?);
            }
            if !sub_exprs.is_empty() {
                let combined = sub_exprs
                    .into_iter()
                    .reduce(|a, b| a.and(b))
                    .unwrap();
                exprs.push(combined);
            }
        } else if key_str == "$or" {
            let list: &Bound<'_, PyList> = value.downcast()?;
            let mut sub_exprs: Vec<FilterExpr> = Vec::new();
            for item in list.iter() {
                let sub_dict: &Bound<'_, PyDict> = item.downcast()?;
                sub_exprs.push(parse_filter_dict(py, sub_dict)?);
            }
            if !sub_exprs.is_empty() {
                let combined = sub_exprs
                    .into_iter()
                    .reduce(|a, b| a.or(b))
                    .unwrap();
                exprs.push(combined);
            }
        } else {
            // Field comparison
            let (op, val) = if let Ok(sub_dict) = value.downcast::<PyDict>() {
                // Extended format: {"field": {"$eq": value}}
                let mut op = ComparisonOp::Eq;
                let mut val = FilterValue::Null;

                for (op_key, op_val) in sub_dict.iter() {
                    let op_str: String = op_key.extract()?;
                    op = match op_str.as_str() {
                        "$eq" => ComparisonOp::Eq,
                        "$in" => ComparisonOp::In,
                        "$gt" => ComparisonOp::Gt,
                        "$lt" => ComparisonOp::Lt,
                        "$gte" => ComparisonOp::Gte,
                        "$lte" => ComparisonOp::Lte,
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Unknown operator: {}",
                                op_str
                            )));
                        }
                    };
                    val = py_to_filter_value(py, &op_val)?;
                    break; // Only support one operator per field
                }
                (op, val)
            } else {
                // Simple format: {"field": value}
                (ComparisonOp::Eq, py_to_filter_value(py, &value)?)
            };

            exprs.push(FilterExpr::Comparison(Comparison {
                field: key_str,
                op,
                value: val,
            }));
        }
    }

    if exprs.is_empty() {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Empty filter expression",
        ))
    } else {
        Ok(exprs.into_iter().reduce(|a, b| a.and(b)).unwrap())
    }
}

fn py_to_filter_value(
    py: Python<'_>,
    obj: &Bound<'_, pyo3::PyAny>,
) -> PyResult<memmachine_core::FilterValue> {
    use memmachine_core::FilterValue;

    if obj.is_none() {
        Ok(FilterValue::Null)
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(FilterValue::String(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(FilterValue::Int(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(FilterValue::Float(f))
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(FilterValue::Bool(b))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let values: PyResult<Vec<FilterValue>> = list
            .iter()
            .map(|item| py_to_filter_value(py, &item))
            .collect();
        Ok(FilterValue::List(values?))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported filter value type",
        ))
    }
}

// ============================================================================
// JSON Conversion Helpers
// ============================================================================

/// Convert a Python object to serde_json::Value
pub fn py_to_json(py: Python<'_>, obj: &Bound<'_, pyo3::PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let values: PyResult<Vec<serde_json::Value>> =
            list.iter().map(|item| py_to_json(py, &item)).collect();
        Ok(serde_json::Value::Array(values?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            let value = py_to_json(py, &v)?;
            map.insert(key, value);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Cannot convert Python object to JSON",
        ))
    }
}

/// Convert serde_json::Value to a Python object
pub fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid JSON number",
                ))
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in obj {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}
