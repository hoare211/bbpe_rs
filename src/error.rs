use thiserror::Error;

#[derive(Error, Debug)]
pub enum BbpeError {
    #[allow(dead_code)]
    #[error("词表大小必须大于 {0}")]
    VocabSizeTooSmall(usize),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON 序列化/反序列化错误: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("无效的 Token ID: {0}")]
    InvalidTokenId(i32),

    #[error("正则编译失败: {0}")]
    RegexError(#[from] regex::Error),

    // ---------- 新增 ----------
    #[error("Aho-Corasick 自动机编译失败: {0}")]
    AhoCorasickError(String),

    #[error("模型文件格式损坏: {0}")]
    InvalidModelFormat(String),

    // ---------- 模型契约 v1（docs/v2输入输出标准规范技术方案.md §7）----------
    #[error("模型 format_version={0} 高于本程序支持的 {1}，请升级 bbpe_rs")]
    UnsupportedFormatVersion(u32, u32),

    #[error(
        "模型缺少 pre_tokenizer_pattern（v0 旧格式）。\
         请显式指定训练时使用的 pattern 并迁移（inspect_model --migrate），\
         或以 allow_legacy=true / --allow-legacy-pattern 加载（将回退 DEFAULT_PATTERN，可能不正确）"
    )]
    MissingPreTokenizerPattern,

    #[error("模型自洽性校验失败: {0}")]
    ModelValidation(String),

    // ---------- M5：容量上界（把"静默截断"变成"显式失败"）----------
    #[error("容量超限：{0}")]
    CapacityExceeded(String),

    // ---------- A1：train 幂等性守卫（把"静默产出损坏模型"变成显式失败）----------
    #[error(
        "该 trainer 已训练过（现有 {} 条 merges），不能直接再次训练。\n\
         原因：训练产出的 merges/vocab 会**残留**在本实例里，而 new_id = 256 + i \
         每轮从 256 重启、与残留 id 撞车，累积后词表不再是合法的连续区间 \
         （V3 违反），产出的模型连自身 inspect --validate-only 都过不了。\n\
         处置：新建一个 trainer；或显式调用 reset() 回到初始状态后重训",
        .0
    )]
    AlreadyTrained(usize),

    // ---------- A2：预分词阶段的静默丢失（正则运行期错误等）----------
    #[error("预分词失败：{0}")]
    PreTokenize(String),

    // ---------- R4-8：运行参数非法（把除零 panic 变成可操作错误）----------
    #[error("运行参数非法：{0}")]
    InvalidRuntimeConfig(String),

    // ---------- R4-20：special token 列表非法（空串 / 重复）----------
    #[error("special token 配置非法：{0}")]
    InvalidSpecialTokens(String),

    // ---------- R6-1：训练被取消（Ctrl+C / cancel_token / timeout）----------
    // 必须走 Err 而不是 break：train_inner 末尾会无条件写 meta.train 并返回
    // Ok(())，break 会让调用方把一个半截模型当成正常产物保存。
    #[error("训练已取消（{reason}，已完成 {merges_done} 次合并）")]
    Cancelled {
        merges_done: usize,
        reason: crate::cancel::CancelReason,
    },

    // ---------- R6-1 P3：OOM 预检在 worker 内失败时把诊断信息带回主线程 ----------
    #[error("{0}")]
    MemoryBudgetExceeded(String),

    // ---------- G4：语料整体不可用（把"静默产出空模型"变成显式失败）----------
    // 此前 skipped_lines == total_lines 时训练照常"成功"：产出一个只有 256 个
    // 基础字节、0 条 merges 的模型，且 inspect --validate-only 也能过（V1–V7
    // 对空 merges 成立）。最典型的触发是把纯文本按默认 --format jsonl 读入 ——
    // 每一行都解析失败被跳过，用户拿到的却是一个"看起来正常"的模型文件。
    #[error("语料不可用：{0}")]
    EmptyCorpus(String),
}
