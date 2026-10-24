//! katago-ws src/error.rs's AppError, minus the axum/aide/diesel parts (same variants and messages).
#[derive(Debug)]
pub enum AppError {
    Sgf(String),
    NotFound,
    BadRequest(String),
    Unauthorized,
    Inference(String),
    ModelLoad(String),
    Db(String),
    Queue(String),
    Internal(String),
}
impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sgf(m) => write!(f, "invalid SGF: {m}"),
            Self::NotFound => write!(f, "not found"),
            Self::BadRequest(m) => write!(f, "bad request: {m}"),
            Self::Unauthorized => write!(f, "unauthorized"),
            Self::Inference(m) => write!(f, "inference error: {m}"),
            Self::ModelLoad(m) => write!(f, "model load error: {m}"),
            Self::Db(m) => write!(f, "database error: {m}"),
            Self::Queue(m) => write!(f, "queue error: {m}"),
            Self::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}
impl std::error::Error for AppError {}
pub type AppResult<T> = Result<T, AppError>;
