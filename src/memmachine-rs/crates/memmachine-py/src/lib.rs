mod memmachine;
mod types;

use pyo3::prelude::*;

use crate::memmachine::PyMemMachine;
use crate::types::{
    PyEpisode, PyEpisodeEntry, PyEpisodeRole, PyEpisodicSearchResult, PyFeatureMetadata,
    PyListResults, PyMemoryType, PySearchResponse, PySemanticFeature, PySession, PySessionData,
    PySessionInfo,
};

#[pymodule]
fn memmachine_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyMemMachine>()?;
    m.add_class::<PyEpisodeRole>()?;
    m.add_class::<PyMemoryType>()?;
    m.add_class::<PyEpisodeEntry>()?;
    m.add_class::<PyEpisode>()?;
    m.add_class::<PySessionData>()?;
    m.add_class::<PySessionInfo>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PySemanticFeature>()?;
    m.add_class::<PyFeatureMetadata>()?;
    m.add_class::<PyEpisodicSearchResult>()?;
    m.add_class::<PySearchResponse>()?;
    m.add_class::<PyListResults>()?;

    Ok(())
}
