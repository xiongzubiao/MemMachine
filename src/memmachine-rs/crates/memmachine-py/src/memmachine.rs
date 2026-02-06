use crate::types::{
    parse_filter_dict, PyEpisode, PyEpisodeEntry, PyEpisodicSearchResult, PyListResults,
    PyMemoryType, PySearchResponse, PySemanticFeature, PySession, PySessionData, PySessionInfo,
};
use memmachine_core::{Configuration, MemMachine, MemoryType};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[pyclass(name = "MemMachine")]
pub struct PyMemMachine {
    inner: Arc<Mutex<Option<MemMachine>>>,
    runtime: tokio::runtime::Handle,
}

#[pymethods]
impl PyMemMachine {
    #[new]
    #[pyo3(signature = (storage_path, config_path=None))]
    fn new(storage_path: String, config_path: Option<String>) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create tokio runtime: {}",
                    e
                ))
            })?;

        let config = if let Some(path) = config_path {
            Configuration::from_yaml_file(&path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to load config from {}: {}",
                    path, e
                ))
            })?
        } else {
            Configuration::new(PathBuf::from(&storage_path))
        };

        let mm = runtime.block_on(async { MemMachine::new(config).await }).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create MemMachine: {}", e))
        })?;

        Ok(Self {
            inner: Arc::new(Mutex::new(Some(mm))),
            runtime: runtime.handle().clone(),
        })
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            if let Some(mm) = guard.as_mut() {
                mm.start().await.map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to start MemMachine: {}",
                        e
                    ))
                })?;
            }
            Ok(())
        })
    }

    fn stop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            if let Some(mm) = guard.as_mut() {
                mm.stop().await.map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to stop MemMachine: {}",
                        e
                    ))
                })?;
            }
            Ok(())
        })
    }

    fn is_started(&self) -> PyResult<bool> {
        let inner = self.inner.clone();
        self.runtime.block_on(async {
            let guard = inner.lock().await;
            Ok(guard.as_ref().is_some_and(|mm| mm.is_started()))
        })
    }

    fn create_session<'py>(
        &self,
        py: Python<'py>,
        session_key: String,
        description: String,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let info = mm.create_session(&session_key, &description).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create session: {}", e))
            })?;

            Ok(PySessionInfo::from(info))
        })
    }

    fn get_session<'py>(
        &self,
        py: Python<'py>,
        session_key: String,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let session = mm.get_session(&session_key).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get session: {}", e))
            })?;

            Ok(session.map(PySession::from))
        })
    }

    fn delete_session<'py>(
        &self,
        py: Python<'py>,
        session_data: PySessionData,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let sd = session_data.into();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            mm.delete_session(&sd).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to delete session: {}",
                    e
                ))
            })?;

            Ok(())
        })
    }

    #[pyo3(signature = (filter=None))]
    fn search_sessions<'py>(
        &self,
        py: Python<'py>,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let filter_expr = if let Some(f) = filter {
            Some(parse_filter_dict(py, &f)?)
        } else {
            None
        };

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let sessions = mm
                .search_sessions(filter_expr.as_ref())
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to search sessions: {}",
                        e
                    ))
                })?;

            Ok(sessions)
        })
    }

    #[pyo3(signature = (session_data, entries, target_memories=None))]
    fn add_episodes<'py>(
        &self,
        py: Python<'py>,
        session_data: PySessionData,
        entries: Vec<Py<PyEpisodeEntry>>,
        target_memories: Option<Vec<PyMemoryType>>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let sd = session_data.into();
        let core_entries: Result<Vec<_>, _> =
            entries
                .into_iter()
                .map(|entry| entry.borrow(py).to_core(py))
                .collect();
        let core_entries = core_entries?;

        let memories: Vec<MemoryType> = target_memories
            .unwrap_or_else(|| vec![PyMemoryType::Episodic])
            .into_iter()
            .map(|m| m.into())
            .collect();

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let ids = mm
                .add_episodes(&sd, core_entries, &memories)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to add episodes: {}",
                        e
                    ))
                })?;

            Ok(ids)
        })
    }

    #[pyo3(signature = (session_data, query, target_memories=None, limit=None, min_score=None))]
    fn query_search<'py>(
        &self,
        py: Python<'py>,
        session_data: PySessionData,
        query: String,
        target_memories: Option<Vec<PyMemoryType>>,
        limit: Option<u32>,
        min_score: Option<f32>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let sd = session_data.into();
        let memories: Vec<MemoryType> = target_memories
            .unwrap_or_else(|| vec![PyMemoryType::Episodic, PyMemoryType::Semantic])
            .into_iter()
            .map(|m| m.into())
            .collect();

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let response = mm
                .query_search(&sd, &query, &memories, limit, min_score)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to query search: {}",
                        e
                    ))
                })?;

            let episodic = response.episodic_memory.map(|em| PyEpisodicSearchResult {
                episodes: em.episodes.into_iter().map(PyEpisode::from).collect(),
                scores: em.scores,
            });

            let semantic = response.semantic_memory.map(|sm| {
                sm.into_iter()
                    .map(|(f, s)| (PySemanticFeature::from(f), s))
                    .collect()
            });

            Ok(PySearchResponse {
                episodic_memory: episodic,
                semantic_memory: semantic,
            })
        })
    }

    #[pyo3(signature = (session_data, target_memories=None, filter=None, page_size=None, page_num=None))]
    fn list_search<'py>(
        &self,
        py: Python<'py>,
        session_data: PySessionData,
        target_memories: Option<Vec<PyMemoryType>>,
        filter: Option<Bound<'py, PyDict>>,
        page_size: Option<u32>,
        page_num: Option<u32>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let sd = session_data.into();
        let memories: Vec<MemoryType> = target_memories
            .unwrap_or_else(|| vec![PyMemoryType::Episodic, PyMemoryType::Semantic])
            .into_iter()
            .map(|m| m.into())
            .collect();

        let filter_expr = if let Some(f) = filter {
            Some(parse_filter_dict(py, &f)?)
        } else {
            None
        };

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let results = mm
                .list_search(&sd, &memories, filter_expr.as_ref(), page_size, page_num)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to list search: {}",
                        e
                    ))
                })?;

            Ok(PyListResults {
                episodic_memory: results
                    .episodic_memory
                    .map(|eps| eps.into_iter().map(PyEpisode::from).collect()),
                semantic_memory: results
                    .semantic_memory
                    .map(|sfs| sfs.into_iter().map(PySemanticFeature::from).collect()),
            })
        })
    }

    #[pyo3(signature = (session_data, filter=None))]
    fn episodes_count<'py>(
        &self,
        py: Python<'py>,
        session_data: PySessionData,
        filter: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let sd = session_data.into();
        let filter_expr = if let Some(f) = filter {
            Some(parse_filter_dict(py, &f)?)
        } else {
            None
        };

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let count = mm
                .episodes_count(&sd, filter_expr.as_ref())
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to get episodes count: {}",
                        e
                    ))
                })?;

            Ok(count)
        })
    }

    #[pyo3(signature = (episode_ids, session_data=None))]
    fn delete_episodes<'py>(
        &self,
        py: Python<'py>,
        episode_ids: Vec<String>,
        session_data: Option<PySessionData>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let sd = session_data.map(|s| s.into());

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let count = mm
                .delete_episodes(&episode_ids, sd.as_ref())
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to delete episodes: {}",
                        e
                    ))
                })?;

            Ok(count)
        })
    }

    fn delete_features<'py>(
        &self,
        py: Python<'py>,
        feature_ids: Vec<String>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let count = mm.delete_features(&feature_ids).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to delete features: {}",
                    e
                ))
            })?;

            Ok(count)
        })
    }

    fn add_semantic_feature<'py>(
        &self,
        py: Python<'py>,
        session_data: PySessionData,
        category: String,
        tag: String,
        feature_name: String,
        value: String,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let sd = session_data.into();

        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let guard = inner.lock().await;
            let mm = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("MemMachine not initialized")
            })?;

            let feature_id = mm
                .add_semantic_feature(&sd, &category, &tag, &feature_name, &value)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to add semantic feature: {}",
                        e
                    ))
                })?;

            Ok(feature_id)
        })
    }

    fn __repr__(&self) -> String {
        let started = self.is_started().unwrap_or(false);
        format!("MemMachine(started={})", started)
    }
}
