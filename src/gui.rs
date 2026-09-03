//! Desktop window (egui / eframe, glow backend).
//!
//! This is a *front end*, not a second implementation: it subscribes to the same
//! `UiEvent` stream the terminal renders and sends the same `UiCommand`s that
//! speech produces, so a button and a spoken phrase take one code path
//! (`from_command` → `Intent` → `act` in main.rs). Nothing about the pipeline
//! changes when the window is attached — v0.10.0 exists so that this file can be
//! purely additive.
//!
//! Two platform facts shape the structure:
//!
//! - **winit owns the main thread.** So the window runs on it and the voice
//!   pipeline moves to a worker thread; the pipeline reports its own death
//!   through the event stream rather than taking the process down silently.
//! - **egui ships no CJK glyphs.** Without a system font every Chinese character
//!   renders as tofu, so `install_cjk_font` probes the platform's font paths.
//!   Tauri was rejected for this window (WebKitGTK bugs and no wlr-layer-shell on
//!   GNOME/Wayland, as documented by SpeakoFlow); glow avoids that whole stack.

use crate::ui::{ToolState, Ui, UiCommand, UiEvent};
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use std::sync::{Arc, Mutex};
use std::thread;

/// Launch the window. Blocks until it is closed, at which point the dropped
/// command sender tells the pipeline to shut down (see `run_with`).
pub fn run(cfg: crate::Config) -> anyhow::Result<()> {
    let (ui, events) = Ui::channel();
    let (cmd_tx, cmd_rx) = unbounded::<UiCommand>();

    // The pipeline cannot have the main thread, so it gets a worker and its own
    // copy of the config. A failure there must be visible in the window.
    let pipeline_cfg = cfg.clone();
    let reporter = ui.clone();
    thread::spawn(move || {
        if let Err(e) = crate::run_with(&pipeline_cfg, ui, cmd_rx) {
            reporter.error(format!("语音流水线已停止: {e}"));
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([880.0, 620.0])
            .with_min_inner_size([560.0, 380.0])
            .with_title(format!("{} · voice assistant", cfg.persona)),
        ..Default::default()
    };
    eframe::run_native(
        "voice-assistant",
        options,
        Box::new(move |cc| {
            install_cjk_font(&cc.egui_ctx);
            Ok(Box::new(App::new(cc.egui_ctx.clone(), events, cmd_tx, &cfg)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("窗口启动失败: {e}"))
}

/// Give egui a font with Chinese glyphs, or leave it Latin-only.
///
/// The size check is not paranoia: macOS ships zero-byte "on demand" stubs (this
/// machine has a 52-byte `Arial Unicode.ttf`), and handing one to egui yields an
/// unusable font rather than an error.
fn install_cjk_font(ctx: &egui::Context) {
    const CANDIDATES: &[&str] = &[
        // macOS — PingFang moved between releases, Hiragino Sans GB is the
        // reliable fallback; both are collections (.ttc), index 0.
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        // Windows
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        // Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ];
    for path in CANDIDATES {
        let Ok(bytes) = std::fs::read(path) else { continue };
        if bytes.len() < 100_000 {
            continue; // a placeholder stub, not a real font
        }
        let mut fonts = egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("cjk".to_owned(), Arc::new(egui::FontData::from_owned(bytes)));
        // Front of the proportional list so CJK wins, appended to monospace so
        // code still looks like code.
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "cjk".to_owned());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .push("cjk".to_owned());
        ctx.set_fonts(fonts);
        return;
    }
    eprintln!("[gui] 未找到中文字体，界面中文会显示为方块（装 Noto Sans CJK 可解决）");
}

/// What the assistant is doing, as far as the event stream reveals.
#[derive(PartialEq, Clone, Copy)]
enum Phase {
    Idle,
    Awake,
    Thinking,
    Restarting,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Phase::Idle => "待机中",
            Phase::Awake => "在听",
            Phase::Thinking => "思考中",
            Phase::Restarting => "重启中",
        }
    }

    fn color(self) -> egui::Color32 {
        match self {
            Phase::Idle => egui::Color32::from_rgb(120, 130, 140),
            Phase::Awake => egui::Color32::from_rgb(70, 190, 120),
            Phase::Thinking => egui::Color32::from_rgb(90, 160, 240),
            Phase::Restarting => egui::Color32::from_rgb(230, 170, 60),
        }
    }
}

/// One line in the conversation stream.
enum Row {
    You(String),
    /// Grows as `Reply` chunks arrive.
    Assistant(String),
    /// Thinking, streamed; grows the same way.
    Thought(String),
    Tool(String),
    Notice(String),
    Error(String),
    /// The assistant spoke unprompted (the sign-off).
    Spoken(String),
}

struct App {
    /// Events land here from a waker thread, which also nudges egui to repaint —
    /// so the window sleeps when nothing is happening instead of polling.
    inbox: Arc<Mutex<Vec<UiEvent>>>,
    commands: Sender<UiCommand>,
    rows: Vec<Row>,
    /// True while a reply is still streaming (draws the caret).
    streaming: bool,
    phase: Phase,
    wake_score: f32,
    threshold: f32,
    wake_word: String,
    persona: String,
    input: String,
    /// Thinking and tool lines are noise until you want them.
    show_details: bool,
    thoughts: usize,
    tools: usize,
}

impl App {
    fn new(
        ctx: egui::Context,
        events: Receiver<UiEvent>,
        commands: Sender<UiCommand>,
        cfg: &crate::Config,
    ) -> Self {
        let inbox: Arc<Mutex<Vec<UiEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = inbox.clone();
        thread::spawn(move || {
            for ev in events.iter() {
                if let Ok(mut q) = sink.lock() {
                    q.push(ev);
                }
                ctx.request_repaint();
            }
        });
        App {
            inbox,
            commands,
            rows: Vec::new(),
            streaming: false,
            phase: Phase::Idle,
            wake_score: 0.0,
            threshold: cfg.wake_threshold,
            wake_word: cfg.wake_display.clone(),
            persona: cfg.persona.clone(),
            input: String::new(),
            show_details: false,
            thoughts: 0,
            tools: 0,
        }
    }

    fn send(&self, cmd: UiCommand) {
        let _ = self.commands.send(cmd);
    }

    fn drain(&mut self) {
        let batch: Vec<UiEvent> = match self.inbox.lock() {
            Ok(mut q) => q.drain(..).collect(),
            Err(_) => return,
        };
        for ev in batch {
            self.apply(ev);
        }
    }

    fn apply(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::Ready { wake_word } => {
                self.wake_word = wake_word;
                self.phase = Phase::Idle;
            }
            UiEvent::WakeScore(s) => self.wake_score = s,
            UiEvent::Wake { .. } => self.phase = Phase::Awake,
            UiEvent::NoSpeech => {
                self.phase = Phase::Idle;
                self.rows.push(Row::Notice("没听到指令，回到待机".into()));
            }
            UiEvent::Transcript(text) => {
                self.rows.push(Row::You(text));
                self.streaming = false;
                self.thoughts = 0;
                self.tools = 0;
            }
            UiEvent::Notice(text) => self.rows.push(Row::Notice(text)),
            UiEvent::Error(text) => self.rows.push(Row::Error(text)),
            UiEvent::Spoken(text) => self.rows.push(Row::Spoken(text)),
            // Chunks extend the row they belong to; a new one starts only when
            // the previous row is something else.
            UiEvent::Reply(text) => {
                match self.rows.last_mut() {
                    Some(Row::Assistant(buf)) if self.streaming => buf.push_str(&text),
                    _ => self.rows.push(Row::Assistant(text)),
                }
                self.streaming = true;
            }
            UiEvent::Thought(text) => {
                match self.rows.last_mut() {
                    Some(Row::Thought(buf)) => buf.push_str(&text),
                    _ => {
                        self.thoughts += 1;
                        self.rows.push(Row::Thought(text));
                    }
                }
                self.streaming = false; // thinking interrupts a reply stream
            }
            UiEvent::Tool { title, state } => {
                let line = match state {
                    ToolState::Started => {
                        self.tools += 1;
                        format!("· {}...", crate::ui::truncate(&title))
                    }
                    ToolState::Completed => format!("✓ {}", crate::ui::truncate(&title)),
                    ToolState::Failed => format!("✗ {} 失败", crate::ui::truncate(&title)),
                    ToolState::Permission { approved: true } => {
                        format!("· {} 请求授权 → 已批准", crate::ui::truncate(&title))
                    }
                    ToolState::Permission { approved: false } => {
                        format!("· {} 请求授权 → 已拒绝（权限模式）", crate::ui::truncate(&title))
                    }
                };
                self.rows.push(Row::Tool(line));
                self.streaming = false;
            }
            UiEvent::TurnEnd { reason } => {
                self.streaming = false;
                self.phase = Phase::Idle;
                if reason == "cancelled" {
                    self.rows.push(Row::Notice("已取消".into()));
                }
            }
            UiEvent::AgentRestarting(why) => {
                self.phase = Phase::Restarting;
                self.rows.push(Row::Error(format!("agent 重启中: {why}")));
            }
            UiEvent::Busy(busy) => {
                self.phase = if busy {
                    Phase::Thinking
                } else if self.phase == Phase::Thinking {
                    Phase::Idle
                } else {
                    self.phase
                };
            }
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.colored_label(self.phase.color(), "●");
            ui.label(self.phase.label());
            ui.separator();
            ui.weak(format!("“{}”", self.wake_word));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_details, "显示思考/工具");
                if self.thoughts > 0 || self.tools > 0 {
                    ui.weak(format!("思考 {} · 工具 {}", self.thoughts, self.tools));
                }
            });
        });
        ui.horizontal(|ui| {
            ui.weak("唤醒");
            // The meter shows every evaluated window, including the quiet ones,
            // which is what makes a wrong threshold obvious.
            ui.add(
                egui::ProgressBar::new(self.wake_score.clamp(0.0, 1.0))
                    .desired_width(220.0)
                    .text(format!("{:.2}", self.wake_score)),
            );
            ui.weak(format!("阈值 {:.2}", self.threshold));
        });
    }

    fn stream(&self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for row in &self.rows {
                    match row {
                        Row::You(text) => Self::bubble(ui, "你", egui::Color32::from_rgb(90, 160, 240), text),
                        Row::Assistant(text) => {
                            Self::bubble(ui, &self.persona, egui::Color32::from_rgb(70, 190, 120), text)
                        }
                        Row::Spoken(text) => Self::bubble(
                            ui,
                            &format!("🔊 {}", self.persona),
                            egui::Color32::from_rgb(70, 190, 120),
                            text,
                        ),
                        Row::Notice(text) => {
                            ui.weak(format!("— {text}"));
                        }
                        Row::Error(text) => {
                            ui.colored_label(egui::Color32::from_rgb(230, 110, 100), format!("! {text}"));
                        }
                        Row::Thought(text) if self.show_details => {
                            ui.weak(format!("思考: {text}"));
                        }
                        Row::Tool(line) if self.show_details => {
                            ui.weak(line);
                        }
                        _ => {} // details collapsed; the counters in the bar remain
                    }
                }
                if self.streaming {
                    ui.weak("▌");
                }
            });
    }

    fn bubble(ui: &mut egui::Ui, who: &str, color: egui::Color32, text: &str) {
        ui.add_space(4.0);
        ui.horizontal_top(|ui| {
            ui.colored_label(color, who);
            ui.add(egui::Label::new(text).wrap());
        });
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Same three intents speech has, so nothing can drift apart.
            if ui.button("⏸ 暂停").clicked() {
                self.send(UiCommand::Pause);
            }
            if ui.button("▶ 继续").clicked() {
                self.send(UiCommand::Resume);
            }
            if ui.button("✕ 放弃").clicked() {
                self.send(UiCommand::Abandon);
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let send = ui.button("发送").clicked();
            let field = ui.add_sized(
                [ui.available_width(), 24.0],
                egui::TextEdit::singleline(&mut self.input).hint_text("也可以直接打字…"),
            );
            let entered = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (send || entered) && !self.input.trim().is_empty() {
                let text = std::mem::take(&mut self.input);
                self.rows.push(Row::You(text.clone()));
                self.send(UiCommand::Prompt(text));
                field.request_focus();
            }
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain();
        // 0.36 merged SidePanel/TopBottomPanel into one `Panel` type, and panels
        // now take `&mut Ui` rather than `&Context` — most tutorials online show
        // the older API.
        egui::Panel::top("status").show(ui, |ui| {
            ui.add_space(4.0);
            self.status_bar(ui);
            ui.add_space(4.0);
        });
        egui::Panel::bottom("controls").show(ui, |ui| {
            ui.add_space(6.0);
            self.controls(ui);
            ui.add_space(6.0);
        });
        egui::CentralPanel::default().show(ui, |ui| self.stream(ui));
    }
}
