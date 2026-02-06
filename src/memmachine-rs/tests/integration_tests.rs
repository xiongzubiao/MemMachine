use memmachine_core::{
    Configuration, EpisodeEntry, EpisodeRole, MemMachine, MemoryType, SessionData, StorageConfig,
};
use std::path::PathBuf;
use tempfile::tempdir;

async fn create_test_memmachine() -> (MemMachine, tempfile::TempDir) {
    let dir = tempdir().expect("Failed to create temp dir");
    let db_path = dir.path().join("test.db");

    let config = Configuration::new(db_path);
    let mm = MemMachine::new(config)
        .await
        .expect("Failed to create MemMachine");

    (mm, dir)
}

#[tokio::test]
async fn test_create_session() {
    let (mm, _dir) = create_test_memmachine().await;

    let info = mm
        .create_session("test-session", "Test session description")
        .await
        .expect("Failed to create session");

    assert_eq!(info.session_key, "test-session");
    assert_eq!(info.description, "Test session description");
}

#[tokio::test]
async fn test_get_session() {
    let (mm, _dir) = create_test_memmachine().await;

    mm.create_session("test-session", "Test description")
        .await
        .expect("Failed to create session");

    let session = mm
        .get_session("test-session")
        .await
        .expect("Failed to get session");

    assert!(session.is_some());
    let session = session.unwrap();
    assert_eq!(session.session_key, "test-session");
}

#[tokio::test]
async fn test_get_nonexistent_session() {
    let (mm, _dir) = create_test_memmachine().await;

    let session = mm
        .get_session("nonexistent")
        .await
        .expect("Failed to query session");

    assert!(session.is_none());
}

#[tokio::test]
async fn test_add_episodes() {
    let (mm, _dir) = create_test_memmachine().await;

    mm.create_session("test-session", "Test")
        .await
        .expect("Failed to create session");

    let session_data = SessionData::new("test-session");

    let entries = vec![
        EpisodeEntry::new(EpisodeRole::User, "Hello, how are you?"),
        EpisodeEntry::new(EpisodeRole::Assistant, "I'm doing well, thank you!"),
    ];

    let ids = mm
        .add_episodes(&session_data, entries, &[MemoryType::Episodic])
        .await
        .expect("Failed to add episodes");

    assert_eq!(ids.len(), 2);
    assert!(ids[0].starts_with("ep-"));
    assert!(ids[1].starts_with("ep-"));
}

#[tokio::test]
async fn test_episodes_count() {
    let (mm, _dir) = create_test_memmachine().await;

    mm.create_session("test-session", "Test")
        .await
        .expect("Failed to create session");

    let session_data = SessionData::new("test-session");

    let entries = vec![
        EpisodeEntry::new(EpisodeRole::User, "Message 1"),
        EpisodeEntry::new(EpisodeRole::Assistant, "Message 2"),
        EpisodeEntry::new(EpisodeRole::User, "Message 3"),
    ];

    mm.add_episodes(&session_data, entries, &[MemoryType::Episodic])
        .await
        .expect("Failed to add episodes");

    let count = mm
        .episodes_count(&session_data, None)
        .await
        .expect("Failed to get count");

    assert_eq!(count, 3);
}

#[tokio::test]
async fn test_delete_episodes() {
    let (mm, _dir) = create_test_memmachine().await;

    mm.create_session("test-session", "Test")
        .await
        .expect("Failed to create session");

    let session_data = SessionData::new("test-session");

    let entries = vec![
        EpisodeEntry::new(EpisodeRole::User, "Message 1"),
        EpisodeEntry::new(EpisodeRole::Assistant, "Message 2"),
    ];

    let ids = mm
        .add_episodes(&session_data, entries, &[MemoryType::Episodic])
        .await
        .expect("Failed to add episodes");

    let deleted = mm
        .delete_episodes(&ids[..1], Some(&session_data))
        .await
        .expect("Failed to delete episodes");

    assert_eq!(deleted, 1);

    let count = mm
        .episodes_count(&session_data, None)
        .await
        .expect("Failed to get count");

    assert_eq!(count, 1);
}

#[tokio::test]
async fn test_start_stop() {
    let (mut mm, _dir) = create_test_memmachine().await;

    assert!(!mm.is_started());

    mm.start().await.expect("Failed to start");
    assert!(mm.is_started());

    mm.stop().await.expect("Failed to stop");
    assert!(!mm.is_started());
}

#[tokio::test]
async fn test_list_search() {
    let (mm, _dir) = create_test_memmachine().await;

    mm.create_session("test-session", "Test")
        .await
        .expect("Failed to create session");

    let session_data = SessionData::new("test-session");

    let entries = vec![
        EpisodeEntry::new(EpisodeRole::User, "First message"),
        EpisodeEntry::new(EpisodeRole::Assistant, "Second message"),
        EpisodeEntry::new(EpisodeRole::User, "Third message"),
    ];

    mm.add_episodes(&session_data, entries, &[MemoryType::Episodic])
        .await
        .expect("Failed to add episodes");

    let results = mm
        .list_search(&session_data, &[MemoryType::Episodic], None, Some(2), Some(0))
        .await
        .expect("Failed to list search");

    assert!(results.episodic_memory.is_some());
    let episodes = results.episodic_memory.unwrap();
    assert_eq!(episodes.len(), 2);
}

#[tokio::test]
async fn test_delete_session() {
    let (mm, _dir) = create_test_memmachine().await;

    mm.create_session("test-session", "Test")
        .await
        .expect("Failed to create session");

    let session_data = SessionData::new("test-session");

    let entries = vec![EpisodeEntry::new(EpisodeRole::User, "Message")];

    mm.add_episodes(&session_data, entries, &[MemoryType::Episodic])
        .await
        .expect("Failed to add episodes");

    mm.delete_session(&session_data)
        .await
        .expect("Failed to delete session");

    let session = mm
        .get_session("test-session")
        .await
        .expect("Failed to query session");

    assert!(session.is_none());
}
