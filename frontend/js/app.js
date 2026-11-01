/**
 * Adaptive AI Personal Life & Study Coach - Core Frontend Application
 * Interacts with the unified multi-agent FastAPI backend.
 */

const API_BASE = '/api';

// Global application state
const state = {
  user: null,
  adaptiveModel: null,
  predictions: [],
  schedule: null,
  goalsData: null,
  materials: [],
  courses: [],
  currentQuiz: null,
  quizAnswers: {},
  currentCardIndex: 0,
  flashcards: [],
};

// Initialize application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  fetchAllData();
  setupEventListeners();
  document.getElementById('course-create-form')?.addEventListener('submit', createCourse);
  document.getElementById('courses-list')?.addEventListener('submit', uploadMaterial);
});

// Navigation Handling
function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetTab = item.dataset.tab;
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));

  const activeNav = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
  const activeTab = document.getElementById(tabId);

  if (activeNav) activeNav.classList.add('active');
  if (activeTab) activeTab.classList.add('active');

  // Trigger tab-specific refresh if needed
  if (tabId === 'tab-planner') renderPlanner();
  if (tabId === 'tab-model') renderAdaptiveModel();
  if (tabId === 'tab-goals') renderGoals();
  if (tabId === 'tab-tutor') renderTutor();
  if (tabId === 'tab-courses') renderCourses();
}

// Fetch all initial data
async function fetchAllData() {
  try {
    const [modelRes, predRes, schedRes, goalsRes, coursesRes] = await Promise.all([
      fetch(`${API_BASE}/adaptive/model`).then(r => r.json()),
      fetch(`${API_BASE}/adaptive/predictions`).then(r => r.json()),
      fetch(`${API_BASE}/planning/schedule`).then(r => r.json()),
      fetch(`${API_BASE}/goals`).then(r => r.json()),
      fetch(`${API_BASE}/assessment/courses`).then(r => r.json()),
    ]);

    state.adaptiveModel = modelRes.adaptive_model;
    state.predictions = predRes.predictions || [];
    state.schedule = schedRes;
    state.goalsData = goalsRes;
    state.courses = coursesRes.courses || [];
    state.materials = state.courses.flatMap(c => (c.lectures || []).map(m => ({...m, course_name:c.course_name})));

    updateGlobalUI();
    renderDashboard();
    renderCourses();
  } catch (err) {
    console.error('Failed to load application data:', err);
    showNotification('Error connecting to backend service.', 'danger');
  }
}

// Global UI Updates
function updateGlobalUI() {
  if (!state.adaptiveModel) return;

  // Update System Mode Pill in sidebar
  const modeBadge = document.getElementById('system-mode-badge');
  const modeText = state.adaptiveModel.current_system_mode || 'NORMAL';
  if (modeBadge) {
    modeBadge.textContent = modeText.replace('_', ' ');
    modeBadge.className = 'status-badge ' + (
      modeText.includes('RESTRUCTURE') ? 'restructure' :
      modeText.includes('REMEDIAL') ? 'remedial' : 'normal'
    );
  }

  // Update Topbar streak pill
  const streakPill = document.getElementById('topbar-streak');
  if (streakPill && state.goalsData) {
    const habit = state.goalsData.habits?.[0];
    if (!habit) { streakPill.style.display = 'none'; return; }
    streakPill.style.display = 'inline-flex';
    streakPill.innerHTML = `<i class="lucide-flame"></i> ${habit.streak} أيام`;
  }
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function renderCourses() {
  const list = document.getElementById('courses-list');
  if (list) list.innerHTML = state.courses.length ? state.courses.map(c => `
    <article class="card course-card"><div class="card-header"><h4>${escapeHtml(c.course_name)}</h4><span class="badge badge-accent">${(c.lectures || []).length} ملفات</span></div>
    <form class="upload-form" data-course-id="${escapeHtml(c.course_id)}"><input type="file" name="file" accept=".pdf,.txt,.docx" required><button class="btn btn-secondary btn-sm">رفع ملف</button></form>
    <ul class="material-list">${(c.lectures || []).map(m => `<li>${escapeHtml(m.title)} <small>${m.pages || 1} صفحة</small></li>`).join('') || '<li class="empty-note">لا توجد ملفات بعد.</li>'}</ul></article>`).join('') : '<div class="empty-state"><h3>ابدأ بأول مادة</h3><p>اكتب اسم المادة ثم ارفع ملفات المحاضرات.</p></div>';
  const select = document.getElementById('quiz-topic-select');
  if (select) {
    const old = select.value;
    select.innerHTML = '<option value="">اختر المادة</option>' + state.courses.map(c => `<option value="${escapeHtml(c.course_id)}">${escapeHtml(c.course_name)}</option>`).join('');
    if (state.courses.some(c => c.course_id === old)) select.value = old;
    select.onchange = renderQuizLectures;
    renderQuizLectures();
  }
}

function renderQuizLectures() {
  const select = document.getElementById('quiz-lecture-select');
  const course = state.courses.find(c => c.course_id === document.getElementById('quiz-topic-select')?.value);
  if (!select) return;
  select.innerHTML = '<option value="">كل محاضرات المادة</option>' + (course?.lectures || []).map(m => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.title)}</option>`).join('');
}

async function createCourse(event) {
  event.preventDefault();
  const name = document.getElementById('course-name-input').value.trim();
  if (!name) return;
  const body = new FormData(); body.append('name', name);
  try {
    const res = await fetch(`${API_BASE}/assessment/courses`, {method:'POST', body});
    const data = await res.json(); if (!res.ok) throw new Error(data.detail || 'تعذر إضافة المادة');
    event.target.reset(); await fetchAllData(); showNotification('تمت إضافة المادة.', 'success');
  } catch (err) { showNotification(err.message, 'danger'); }
}

async function uploadMaterial(event) {
  if (!event.target.matches('.upload-form')) return;
  event.preventDefault();
  const file = event.target.querySelector('input[type=file]').files[0];
  if (!file) return;
  try {
    const res = await fetch(`${API_BASE}/assessment/courses/${encodeURIComponent(event.target.dataset.courseId)}/materials`, {method:'POST', body:new FormData(event.target)});
    const data = await res.json(); if (!res.ok) throw new Error(data.detail || 'تعذر رفع الملف');
    await fetchAllData(); showNotification('تم رفع الملف للمادة.', 'success');
  } catch (err) { showNotification(err.message, 'danger'); }
}

// --------------------------------------------------------------------------
// 1. DASHBOARD VIEW
// --------------------------------------------------------------------------
function renderDashboard() {
  document.getElementById('dashboard-welcome')?.classList.add('is-new-user');
  return;
  const model = state.adaptiveModel;
  if (!model) return;

  // Active predictions widget
  const predContainer = document.getElementById('dashboard-predictions');
  if (predContainer) {
    if (!state.predictions || state.predictions.length === 0) {
      predContainer.innerHTML = `<div class="pred-empty" style="color:var(--text-muted); font-size:13px;">No active struggle predictions. System is operating in equilibrium.</div>`;
    } else {
      predContainer.innerHTML = state.predictions.slice(0, 3).map(p => `
        <div class="prediction-card ${p.risk_level === 'High' ? 'danger' : ''}">
          <div class="pred-info">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="badge ${p.risk_level === 'High' ? 'badge-high' : 'badge-medium'}">${p.risk_level} Risk (${Math.round(p.risk_score * 100)}%)</span>
              <span style="font-size:12px; font-weight:600; color:#fff;">${formatTargetName(p.target)}: ${p.target_id}</span>
            </div>
            <p style="margin-top:6px;">${p.rationale}</p>
            <div class="pred-action">💡 Preemptive Action: ${p.preemptive_action}</div>
          </div>
        </div>
      `).join('');
    }
  }

  // Focus and Delay Gauges
  const focusEl = document.getElementById('dash-focus-mins');
  const delayEl = document.getElementById('dash-delay-factor');
  const windowEl = document.getElementById('dash-peak-window');
  const burnoutEl = document.getElementById('dash-burnout-risk');

  if (focusEl) focusEl.textContent = `${model.effective_focus_duration_mins}m`;
  if (delayEl) delayEl.textContent = `${model.task_delay_factor}x`;
  if (windowEl) windowEl.textContent = model.peak_productivity_window || '09:00-11:00';
  if (burnoutEl) {
    burnoutEl.textContent = model.burnout_risk_score || 'Low';
    burnoutEl.className = 'badge ' + (model.burnout_risk_score === 'High' ? 'badge-high' : model.burnout_risk_score === 'Medium' ? 'badge-medium' : 'badge-low');
  }

  // Today's Agenda Tasks
  const tasksContainer = document.getElementById('dashboard-today-tasks');
  if (tasksContainer && state.schedule) {
    const tasks = state.schedule.tasks || [];
    tasksContainer.innerHTML = tasks.slice(0, 4).map(t => `
      <div style="display:flex; justify-content:space-between; align-items:center; padding:12px; border-bottom:1px solid rgba(255,255,255,0.05);">
        <div>
          <div style="font-weight:600; font-size:14px; color:#fff;">${t.title}</div>
          <div style="font-size:12px; color:var(--text-muted); display:flex; gap:10px; margin-top:4px;">
            <span>⏱️ ${t.estimated_mins}m (adjusted: ${t.adjusted_mins || t.estimated_mins}m)</span>
            <span>📚 ${t.topic_ref || 'Coursework'}</span>
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="badge ${t.cognitive_load === 'high' ? 'badge-high' : 'badge-medium'}">${t.cognitive_load} load</span>
          <button class="btn btn-secondary btn-sm" onclick="quickCompleteTask('${t.task_id}')">Done</button>
        </div>
      </div>
    `).join('');
  }
}

function formatTargetName(target) {
  if (target === 'task_miss_risk') return 'Task Miss Risk';
  if (target === 'topic_struggle_risk') return 'Topic Struggle Risk';
  if (target === 'overload_risk') return 'Cognitive Overload Risk';
  return target;
}

// --------------------------------------------------------------------------
// 2. PERSONAL ADAPTIVE MODEL HUB (THE CORE USP)
// --------------------------------------------------------------------------
function renderAdaptiveModel() {
  const model = state.adaptiveModel;
  if (!model) return;

  // Fill in Gauges
  document.getElementById('model-peak-window').textContent = model.peak_productivity_window;
  document.getElementById('model-focus-duration').textContent = `${model.effective_focus_duration_mins} mins`;
  document.getElementById('model-delay-factor').textContent = `${model.task_delay_factor}x`;
  document.getElementById('model-learning-velocity').textContent = `${Math.round((model.learning_rate_velocity || 0.65) * 100)}%`;
  document.getElementById('model-burnout-risk').textContent = model.burnout_risk_score;
  document.getElementById('model-system-mode').textContent = (model.current_system_mode || 'NORMAL').replace('_', ' ');

  // Procrastination triggers
  const triggersContainer = document.getElementById('model-triggers-list');
  if (triggersContainer) {
    const patterns = model.procrastination_patterns || [];
    triggersContainer.innerHTML = patterns.map(p => `
      <div class="tag-item"><i class="lucide-alert-triangle"></i> ${p}</div>
    `).join('');
  }

  // Topic Mastery Breakdown
  const masteryContainer = document.getElementById('model-mastery-list');
  if (masteryContainer) {
    const masteries = model.topic_mastery || {};
    masteryContainer.innerHTML = Object.entries(masteries).map(([topic, pct]) => `
      <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
          <span style="font-weight:600; color:#fff;">${topic}</span>
          <span style="color:${pct < 65 ? 'var(--danger)' : 'var(--success)'}; font-weight:700;">${pct}%</span>
        </div>
        <div class="progress-container">
          <div class="progress-bar ${pct < 65 ? 'danger' : ''}" style="width: ${pct}%"></div>
        </div>
      </div>
    `).join('');
  }

  // Weak Concepts Tags
  const weaksContainer = document.getElementById('model-weak-topics');
  if (weaksContainer) {
    const weaks = model.weak_topics || [];
    weaksContainer.innerHTML = weaks.map(w => `
      <div class="tag-item weak"><i class="lucide-x-circle"></i> ${w}</div>
    `).join('') || '<span style="font-size:13px; color:var(--text-muted);">No severe knowledge gaps identified.</span>';
  }
}

// --------------------------------------------------------------------------
// 3. SMART PLANNER & CALENDAR
// --------------------------------------------------------------------------
function renderPlanner() {
  if (!state.schedule) return;

  // University schedule
  const lecsContainer = document.getElementById('planner-lectures-list');
  if (lecsContainer) {
    const lecs = state.schedule.university_schedule || [];
    lecsContainer.innerHTML = lecs.map(l => `
      <div style="padding:12px; background:rgba(255,255,255,0.03); border:1px solid var(--border-color); border-radius:var(--radius-sm); margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600; color:#fff;">${l.title}</span>
          <span class="badge badge-accent">${l.day} ${l.start}-${l.end}</span>
        </div>
        <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
          📍 ${l.room || 'Hall'} • 👨‍🏫 ${l.instructor || 'Faculty'}
        </div>
      </div>
    `).join('');
  }

  // Optimized blocks
  const blocksContainer = document.getElementById('planner-blocks-list');
  if (blocksContainer) {
    const blocks = state.schedule.scheduled_blocks || [];
    if (blocks.length === 0) {
      blocksContainer.innerHTML = `
        <div style="padding:20px; text-align:center; color:var(--text-muted);">
          <p>No calendar blocks generated yet.</p>
          <button class="btn btn-primary" style="margin-top:12px;" onclick="triggerScheduleOptimization()">
            <i class="lucide-sparkles"></i> Optimize Schedule with AI
          </button>
        </div>
      `;
    } else {
      blocksContainer.innerHTML = blocks.map(b => `
        <div style="padding:14px; background:rgba(99,102,241,0.06); border:1px solid var(--border-color); border-radius:var(--radius-sm); margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-weight:600; color:#fff; font-size:14px;">${b.title} ${b.part ? `(Part ${b.part})` : ''}</div>
            <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
              📅 ${b.date} • ⏰ ${b.start} - ${b.end}
            </div>
          </div>
          <button class="btn btn-secondary btn-sm" onclick="quickCompleteTask('${b.task_id}')">Mark Complete</button>
        </div>
      `).join('');
    }
  }

  // Tasks list
  const tasksContainer = document.getElementById('planner-tasks-table');
  if (tasksContainer) {
    const tasks = state.schedule.tasks || [];
    tasksContainer.innerHTML = tasks.map(t => `
      <tr>
        <td style="padding:12px; font-weight:600; color:#fff;">${t.title}</td>
        <td style="padding:12px;">${t.estimated_mins}m <small style="color:var(--text-muted);">(${t.adjusted_mins || t.estimated_mins}m)</small></td>
        <td style="padding:12px;"><span class="badge ${t.cognitive_load === 'high' ? 'badge-high' : 'badge-medium'}">${t.cognitive_load}</span></td>
        <td style="padding:12px;"><span class="badge ${t.status === 'completed' ? 'badge-success' : 'badge-warning'}">${t.status}</span></td>
        <td style="padding:12px;">
          <button class="btn btn-secondary btn-sm" onclick="deconstructTask('${t.task_id}')">Split</button>
          <button class="btn btn-outline btn-sm" onclick="quickCompleteTask('${t.task_id}')">Done</button>
        </td>
      </tr>
    `).join('');
  }
}

async function triggerScheduleOptimization() {
  showNotification('Running 5-step deterministic schedule optimizer...', 'info');
  try {
    const res = await fetch(`${API_BASE}/planning/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_date: new Date().toISOString().slice(0, 10) })
    }).then(r => r.json());

    if (res.status === 'success') {
      showNotification(`Schedule optimized! Scheduled ${res.result.total_blocks} blocks.`, 'success');
      await fetchAllData();
      renderPlanner();
    }
  } catch (err) {
    showNotification('Optimization error: ' + err.message, 'danger');
  }
}

async function deconstructTask(taskId) {
  showNotification(`Deconstructing task ${taskId} to match attention threshold...`, 'info');
  try {
    const res = await fetch(`${API_BASE}/planning/tasks/${taskId}/deconstruct`, {
      method: 'POST'
    }).then(r => r.json());

    if (res.status === 'success') {
      const titles = res.micro_tasks.map(m => `• ${m.title} (${m.estimated_mins}m)`).join('\n');
      alert(`Task Deconstructed into ${res.micro_tasks.length} Attention-Span Chunks:\n\n${titles}`);
    }
  } catch (err) {
    showNotification('Deconstruction error: ' + err.message, 'danger');
  }
}

async function quickCompleteTask(taskId) {
  try {
    const res = await fetch(`${API_BASE}/planning/tasks/${taskId}/status`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: 'completed', actual_mins: 35 })
    }).then(r => r.json());

    if (res.status === 'success') {
      showNotification(`Task marked complete! Event dispatched to Adaptive Manager.`, 'success');
      await fetchAllData();
      renderPlanner();
      renderDashboard();
    }
  } catch (err) {
    showNotification('Status update error: ' + err.message, 'danger');
  }
}

// --------------------------------------------------------------------------
// 4. AI STUDY ROOM & TUTOR (ai_depi RAG)
// --------------------------------------------------------------------------
function renderTutor() {
  // Populate material selects
  const matSelect = document.getElementById('tutor-material-select');
  if (matSelect) {
    matSelect.innerHTML = `<option value="">All Course Materials</option>` +
      state.materials.map(m => `<option value="${escapeHtml(m.filename)}">${escapeHtml(m.course_name)} — ${escapeHtml(m.title)}</option>`).join('');
  }
}

async function sendTutorQuery() {
  const queryInput = document.getElementById('tutor-chat-input');
  const chatOutput = document.getElementById('tutor-chat-output');
  const matSelect = document.getElementById('tutor-material-select');
  const query = queryInput.value.trim();
  if (!query) return;

  // Append user bubble
  chatOutput.innerHTML += `
    <div style="margin-bottom:16px; text-align:right;">
      <div style="display:inline-block; background:var(--accent-gradient); color:#fff; padding:10px 16px; border-radius:var(--radius-md); font-size:14px; max-width:80%;">
        ${escapeHtml(query)}
      </div>
    </div>
  `;
  queryInput.value = '';
  chatOutput.scrollTop = chatOutput.scrollHeight;

  // Append loader
  const loaderId = 'loader-' + Date.now();
  chatOutput.innerHTML += `
    <div id="${loaderId}" style="margin-bottom:16px; color:var(--text-muted); font-size:13px;">
      <i class="lucide-loader"></i> AI Tutor is retrieving lecture context and synthesizing answer...
    </div>
  `;

  try {
    const res = await fetch(`${API_BASE}/tutor/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, filter_doc: matSelect.value || null })
    }).then(r => r.json());

    document.getElementById(loaderId)?.remove();

    const result = res.result;
    const sourcesHtml = (result.sources || []).map(s => `
      <span class="badge badge-accent" style="font-size:10px; margin-right:4px;">
        📄 ${s.document_name} (Page ${s.page})
      </span>
    `).join('');

    chatOutput.innerHTML += `
      <div style="margin-bottom:20px;">
        <div style="display:inline-block; background:rgba(22,30,49,0.9); border:1px solid var(--border-color); color:#fff; padding:16px; border-radius:var(--radius-md); font-size:14px; max-width:85%;">
          <div style="font-weight:600; color:var(--accent-primary); margin-bottom:6px; display:flex; align-items:center; gap:6px;">
            🤖 Grounded AI Tutor Response
          </div>
          <div style="white-space:pre-wrap; line-height:1.6;">${escapeHtml(result.answer)}</div>
          ${sourcesHtml ? `<div style="margin-top:12px; border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;">${sourcesHtml}</div>` : ''}
        </div>
      </div>
    `;
    chatOutput.scrollTop = chatOutput.scrollHeight;
  } catch (err) {
    document.getElementById(loaderId)?.remove();
    showNotification('Chat error: ' + err.message, 'danger');
  }
}

async function generateExplanation() {
  const topic = prompt("Enter the topic or theorem to explain:", "Pumping Lemma for Regular Languages");
  if (!topic) return;

  showNotification(`Generating personalized pedagogical explanation for ${topic}...`, 'info');
  try {
    const res = await fetch(`${API_BASE}/tutor/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic: topic, level: "intermediate" })
    }).then(r => r.json());

    if (res.status === 'success') {
      const modal = document.getElementById('explanation-modal');
      const body = document.getElementById('explanation-modal-body');
      body.innerHTML = `
        <h3 style="color:#fff; margin-bottom:12px;">Personalized Explanation: ${res.result.topic}</h3>
        <div style="font-size:14px; line-height:1.7; white-space:pre-wrap; color:#e2e8f0;">${res.result.explanation}</div>
      `;
      modal.classList.add('active');
    }
  } catch (err) {
    showNotification('Explanation error: ' + err.message, 'danger');
  }
}

async function loadFlashcards() {
  showNotification('Synthesizing active recall flashcards from course materials...', 'info');
  try {
    const res = await fetch(`${API_BASE}/tutor/flashcards`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic: "Automata & Formal Languages", count: 5 })
    }).then(r => r.json());

    if (res.status === 'success' && res.flashcards.length > 0) {
      state.flashcards = res.flashcards;
      state.currentCardIndex = 0;
      renderCurrentFlashcard();
      document.getElementById('flashcard-section').style.display = 'block';
    }
  } catch (err) {
    showNotification('Flashcard error: ' + err.message, 'danger');
  }
}

function renderCurrentFlashcard() {
  const card = state.flashcards[state.currentCardIndex];
  if (!card) return;

  document.getElementById('flashcard-front-text').textContent = card.front;
  document.getElementById('flashcard-back-text').textContent = card.back;
  document.getElementById('flashcard-counter').textContent = `${state.currentCardIndex + 1} of ${state.flashcards.length}`;
  document.getElementById('flashcard-box').classList.remove('flipped');
}

function flipCurrentCard() {
  document.getElementById('flashcard-box').classList.toggle('flipped');
}

function nextFlashcard() {
  if (state.flashcards.length === 0) return;
  state.currentCardIndex = (state.currentCardIndex + 1) % state.flashcards.length;
  renderCurrentFlashcard();
}

function prevFlashcard() {
  if (state.flashcards.length === 0) return;
  state.currentCardIndex = (state.currentCardIndex - 1 + state.flashcards.length) % state.flashcards.length;
  renderCurrentFlashcard();
}

// --------------------------------------------------------------------------
// 5. ASSESSMENT & DIAGNOSTIC CENTER (Abdelrahman Quiz Agent)
// --------------------------------------------------------------------------
async function startDiagnosticQuiz() {
  const topicSelect = document.getElementById('quiz-topic-select');
  const course = state.courses.find(c => c.course_id === topicSelect?.value);
  if (!course) return showNotification('أضف مادة واخترها أولاً.', 'danger');
  const lecture = document.getElementById('quiz-lecture-select')?.value;

  showNotification('جاري تجهيز اختبار شامل…', 'info');
  try {
    const res = await fetch(`${API_BASE}/assessment/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ course_name: course.course_name, lecture_title: lecture || null,
        material_filename: state.materials.find(m => m.id === lecture)?.filename || null, num_questions: 15 })
    }).then(async r => { if (!r.ok) throw new Error((await r.json()).detail || 'تعذر إنشاء الاختبار'); return r.json(); });

    if (res.status === 'success') {
      state.currentQuiz = res.quiz;
      state.quizAnswers = {};
      renderActiveQuiz();
    }
  } catch (err) {
    showNotification('Quiz error: ' + err.message, 'danger');
  }
}

function renderActiveQuiz() {
  const quiz = state.currentQuiz;
  if (!quiz) return;

  const container = document.getElementById('active-quiz-container');
  container.style.display = 'block';
  document.getElementById('quiz-report-container').style.display = 'none';

  container.innerHTML = `
    <div class="card" style="margin-bottom:20px;">
      <div class="card-header">
        <div class="card-title">اختبار: ${escapeHtml(quiz.course_name || quiz.topic || '')} ${quiz.lecture_title ? '— ' + escapeHtml(quiz.lecture_title) : ''}</div>
        <span class="badge badge-accent">${quiz.questions.length} Questions</span>
      </div>
      <div id="quiz-questions-list">
        ${quiz.questions.map((q, idx) => `
          <div class="quiz-question-card" id="q-card-${q.id}">
            <div style="font-weight:600; font-size:15px; color:#fff; margin-bottom:12px;">
              ${idx + 1}. ${escapeHtml(q.question)}
            </div>
            <div class="quiz-options-group">
              ${Object.entries(q.options).map(([optKey, optVal]) => `
                <div class="quiz-option" onclick="selectQuizAnswer('${q.id}', '${optKey}')" id="opt-${q.id}-${optKey}">
                  <span style="font-weight:700; color:var(--accent-primary); width:20px;">${optKey})</span>
                  <span>${escapeHtml(optVal)}</span>
                </div>
              `).join('')}
            </div>
            <div class="answer-feedback" id="feedback-${q.id}" aria-live="polite"></div>
          </div>
        `).join('')}
      </div>
      <div style="margin-top:20px; display:flex; justify-content:flex-end;">
        <button class="btn btn-primary" onclick="submitDiagnosticQuiz()">
          <i class="lucide-check-circle"></i> Submit & Feed to Adaptive Model
        </button>
      </div>
    </div>
  `;
}

async function selectQuizAnswer(qid, opt) {
  state.quizAnswers[qid] = opt;
  // Mark UI
  document.querySelectorAll(`[id^="opt-${qid}-"]`).forEach(el => el.classList.remove('selected'));
  document.getElementById(`opt-${qid}-${opt}`)?.classList.add('selected');
  const q = state.currentQuiz?.questions.find(item => String(item.id) === String(qid));
  if (!q) return;
  try {
    const resp = await fetch(`${API_BASE}/assessment/check_single`, {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question_id:String(qid),selected_answer:opt,correct_answer:q.correct_answer,explanation:q.explanation || ''})});
    const result = await resp.json();
    const box = document.getElementById(`feedback-${qid}`);
    if (box) box.innerHTML = `<div class="instant-feedback ${result.is_correct ? 'correct' : 'incorrect'}"><strong>${result.is_correct ? 'إجابة صحيحة' : `إجابة غير صحيحة — الصحيحة ${escapeHtml(result.correct_answer)}`}</strong><p>${escapeHtml(result.explanation || '')}</p></div>`;
  } catch (err) { showNotification('تعذر تصحيح الإجابة الآن.', 'danger'); }
}

async function submitDiagnosticQuiz() {
  const quiz = state.currentQuiz;
  if (!quiz) return;

  // Verify all answered
  if (Object.keys(state.quizAnswers).length < quiz.questions.length) {
    if (!confirm("You have unanswered questions. Submit anyway?")) return;
  }

  showNotification('Evaluating responses and closing the feedback loop...', 'info');
  try {
    const res = await fetch(`${API_BASE}/assessment/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        quiz_id: quiz.quiz_id,
        course_name: quiz.course_name || quiz.topic,
        questions: quiz.questions,
        answers: state.quizAnswers
      })
    }).then(r => r.json());

    if (res.status === 'success') {
      showNotification(`Diagnostic submitted! Score: ${res.diagnostic_report.evaluation.score_pct}%`, 'success');
      renderQuizReport(res.diagnostic_report);
      await fetchAllData();
      renderAdaptiveModel();
      renderDashboard();
    }
  } catch (err) {
    showNotification('Evaluation error: ' + err.message, 'danger');
  }
}

function renderQuizReport(report) {
  const evalData = report.evaluation;
  const loop = report.closed_loop_adaptation;

  document.getElementById('active-quiz-container').style.display = 'none';
  const reportBox = document.getElementById('quiz-report-container');
  reportBox.style.display = 'block';

  const errorsHtml = Object.entries(evalData.error_types || {}).map(([errType, count]) => `
    <span class="badge badge-danger">${errType}: ${count}</span>
  `).join('') || '<span class="badge badge-success">Zero Conceptual Errors!</span>';

  const weakHtml = (evalData.weak_topics || []).map(w => `
    <span class="tag-item weak">⚠️ ${w}</span>
  `).join('') || '<span style="font-size:13px; color:var(--text-muted);">None detected.</span>';

  reportBox.innerHTML = `
    <div class="card" style="border-color:var(--accent-primary);">
      <div class="card-header">
        <div class="card-title">📊 Assessment Diagnostic Report</div>
        <span class="badge ${evalData.score_pct >= 65 ? 'badge-success' : 'badge-danger'}">Score: ${evalData.score_pct}%</span>
      </div>
      <div style="font-size:15px; font-weight:600; color:#fff; margin-bottom:10px;">${evalData.feedback}</div>
      
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin:20px 0;">
        <div style="background:rgba(255,255,255,0.02); padding:16px; border-radius:var(--radius-sm); border:1px solid var(--border-color);">
          <div style="font-size:12px; color:var(--text-muted); text-transform:uppercase; margin-bottom:8px;">Error Classifications</div>
          <div>${errorsHtml}</div>
        </div>
        <div style="background:rgba(255,255,255,0.02); padding:16px; border-radius:var(--radius-sm); border:1px solid var(--border-color);">
          <div style="font-size:12px; color:var(--text-muted); text-transform:uppercase; margin-bottom:8px;">Isolated Weak Concepts</div>
          <div>${weakHtml}</div>
        </div>
      </div>

      <div style="background:linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(22,30,49,0.8) 100%); padding:18px; border-radius:var(--radius-sm); border-left:4px solid var(--accent-primary); margin-top:20px;">
        <h4 style="color:#fff; font-size:14px; margin-bottom:6px;">🔄 Closed-Loop System Adaptation Applied:</h4>
        <p style="font-size:13px; color:var(--text-muted);">
          System strategy mode updated to: <strong>${loop.adaptive_model.current_system_mode}</strong>.<br/>
          Personal model topic mastery updated to: <strong>${loop.adaptive_model.topic_mastery[evalData.topic]}%</strong>.
        </p>
        <div style="margin-top:8px; font-size:12px; color:var(--accent-primary);">
          ${(loop.executed_interventions || []).map(i => `<div>✓ ${i}</div>`).join('')}
        </div>
      </div>

      <div style="margin-top:20px; text-align:right;">
        <button class="btn btn-secondary" onclick="startDiagnosticQuiz()">Take Another Quiz</button>
      </div>
    </div>
  `;
}

// --------------------------------------------------------------------------
// 6. GOALS & LIFE COACH
// --------------------------------------------------------------------------
function renderGoals() {
  if (!state.goalsData) return;

  // Goals list
  const goalsContainer = document.getElementById('goals-cards-list');
  if (goalsContainer) {
    const goals = state.goalsData.goals || [];
    goalsContainer.innerHTML = goals.map(g => `
      <div class="card" style="margin-bottom:16px;">
        <div class="card-header">
          <div>
            <span class="badge badge-accent" style="margin-bottom:6px;">${g.category}</span>
            <div style="font-size:16px; font-weight:700; color:#fff;">${g.title}</div>
            <div style="font-size:13px; color:var(--text-muted); margin-top:4px;">🎯 Why: ${g.why || 'Consistency and mastery'}</div>
          </div>
          <span class="badge badge-low">${g.status}</span>
        </div>
        <div style="margin-top:14px;">
          <div style="font-size:12px; font-weight:600; color:var(--text-dim); text-transform:uppercase; margin-bottom:8px;">Milestones Roadmap</div>
          ${(g.milestones || []).map(m => `
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-top:1px solid rgba(255,255,255,0.05);">
              <span style="font-size:13px; color:#e2e8f0;">📌 ${m.title}</span>
              <span class="badge ${m.status === 'completed' ? 'badge-success' : 'badge-warning'}">${m.status}</span>
            </div>
          `).join('') || '<div style="font-size:13px; color:var(--text-muted);">No milestones added.</div>'}
        </div>
      </div>
    `).join('');
  }

  // Habits and Gamification
  const habitsContainer = document.getElementById('habits-cards-list');
  if (habitsContainer) {
    const habits = state.goalsData.habits || [];
    habitsContainer.innerHTML = habits.map(h => `
      <div style="padding:16px; background:rgba(255,255,255,0.03); border:1px solid var(--border-color); border-radius:var(--radius-sm); margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div style="font-weight:700; color:#fff; font-size:14px;">${h.name}</div>
          <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
            🔥 Streak: <strong>${h.streak} days</strong> (Best: ${h.best_streak}) • 🛡️ Freezes: ${h.freezes_available}
          </div>
        </div>
        <button class="btn btn-primary btn-sm" onclick="checkInHabit('${h.name}')">
          Check In Today
        </button>
      </div>
    `).join('');
  }

  // Achievements Showcase
  const achContainer = document.getElementById('achievements-badges-list');
  if (achContainer) {
    const achs = state.goalsData.achievements || [];
    achContainer.innerHTML = achs.map(a => `
      <div style="padding:12px 16px; background:rgba(99,102,241,0.08); border:1px solid var(--border-color-strong); border-radius:var(--radius-sm); display:inline-flex; align-items:center; gap:10px; margin-right:10px; margin-bottom:10px;">
        <span style="font-size:22px;">🏆</span>
        <div>
          <div style="font-weight:700; color:#fff; font-size:13px;">${a.title}</div>
          <div style="font-size:11px; color:var(--text-muted);">${a.description}</div>
        </div>
      </div>
    `).join('');
  }
}

async function checkInHabit(habitName) {
  try {
    const res = await fetch(`${API_BASE}/goals/habits/checkin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ habit_name: habitName })
    }).then(r => r.json());

    if (res.status === 'success') {
      const details = res.details;
      showNotification(`Habit checked in! New streak: ${details.new_streak} days.`, 'success');
      await fetchAllData();
      renderGoals();
      renderDashboard();
    }
  } catch (err) {
    showNotification('Habit error: ' + err.message, 'danger');
  }
}

async function submitReflectionForm(e) {
  e.preventDefault();
  const energy = parseInt(document.getElementById('ref-energy-slider').value);
  const mood = document.getElementById('ref-mood-select').value;
  const text = document.getElementById('ref-text').value;
  const obstacles = document.getElementById('ref-obstacles').value;

  showNotification('Submitting reflection and scoring burnout risk...', 'info');
  try {
    const res = await fetch(`${API_BASE}/goals/reflections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ energy, mood, reflection: text, obstacles })
    }).then(r => r.json());

    if (res.status === 'success') {
      closeModal('reflection-modal');
      alert(`Daily Check-in Recorded!\n\n${res.result.coach_response.reflection.ai_feedback}\n\nBurnout Risk: ${res.result.coach_response.burnout_risk}`);
      await fetchAllData();
      renderGoals();
      renderDashboard();
    }
  } catch (err) {
    showNotification('Reflection error: ' + err.message, 'danger');
  }
}

// --------------------------------------------------------------------------
// 7. SIMULATION & SCENARIO LAB
// --------------------------------------------------------------------------
async function triggerScenario(scenarioKey) {
  showNotification(`Running Simulation Persona: ${scenarioKey}...`, 'info');
  const simLog = document.getElementById('simulation-output-log');
  simLog.innerHTML = `<div style="color:var(--text-muted);"><i class="lucide-loader"></i> Executing multi-agent closed loop...</div>`;

  try {
    const res = await fetch(`${API_BASE}/adaptive/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: scenarioKey })
    }).then(r => r.json());

    const out = res.adaptation_outcome;
    simLog.innerHTML = `
      <div style="background:rgba(0,0,0,0.3); border:1px solid var(--border-color); border-radius:var(--radius-sm); padding:16px;">
        <div style="font-weight:700; color:#fff; font-size:15px; margin-bottom:6px;">
          🎬 Scenario: ${res.scenario}
        </div>
        <div style="font-size:13px; color:var(--text-main); margin-bottom:10px;">
          <strong>Simulated Action:</strong> ${res.simulated_action}
        </div>
        <div style="font-size:13px; color:var(--accent-primary); margin-bottom:12px;">
          <strong>Core USP Mechanism:</strong> ${res.explanation}
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.08); padding-top:10px; font-size:12px; color:var(--text-muted);">
          <div>• System Mode: <strong>${out.adaptive_model?.current_system_mode || 'NORMAL'}</strong></div>
          <div>• Dispatches Created: <strong>${(out.dispatches || []).length}</strong></div>
          <div>• Preemptive Interventions:</div>
          ${(out.executed_interventions || []).map(i => `<div style="padding-left:12px; color:#a5b4fc;">→ ${i}</div>`).join('') || '<div style="padding-left:12px;">None required</div>'}
        </div>
      </div>
    `;

    await fetchAllData();
    renderAdaptiveModel();
    renderDashboard();
  } catch (err) {
    simLog.innerHTML = `<div style="color:var(--danger);">Simulation failed: ${err.message}</div>`;
  }
}

// --------------------------------------------------------------------------
// Event Listeners & Modals
// --------------------------------------------------------------------------
function setupEventListeners() {
  // Tutor chat input enter key
  document.getElementById('tutor-chat-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendTutorQuery();
  });

  // Energy slider display
  const slider = document.getElementById('ref-energy-slider');
  const sliderVal = document.getElementById('ref-energy-val');
  slider?.addEventListener('input', () => {
    if (sliderVal) sliderVal.textContent = slider.value + ' / 5';
  });
}

function openModal(id) {
  document.getElementById(id)?.classList.add('active');
}

function closeModal(id) {
  document.getElementById(id)?.classList.remove('active');
}

function showNotification(msg, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: ${type === 'danger' ? '#ef4444' : type === 'success' ? '#10b981' : '#6366f1'};
    color: #fff; padding: 12px 20px; border-radius: 8px; font-size: 13px; font-weight: 600;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4); transition: opacity 0.3s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
