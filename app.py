import streamlit as st
import pandas as pd, time, uuid, os, random
from datetime import datetime
import joblib

# ----------------- CONSTANTS -----------------

APP_TITLE = "Tone Judgment Study"
STUDY_CSV = "a2_study_with_outputs.csv"
MODEL_PKL = "a2_model.pkl"
LOG_DIR = "logs"
LOG_TRIALS = os.path.join(LOG_DIR, "a3_logs.csv")
LOG_SESS = os.path.join(LOG_DIR, "a3_sessions.csv")

os.makedirs(LOG_DIR, exist_ok=True)


# ----------------- DATA / MODEL LOADING -----------------

@st.cache_data
def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError("Put a2_study_with_outputs.csv next to app.py")
    df = pd.read_csv(path)

    need_cols = [
        "input_text",
        "ground_truth_label",
        "model_output_label",
        "model_score",
        "human_rating_placeholder",
        "ai_readable_label",
        "ai_rationale",
    ]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {STUDY_CSV}")
    return df


@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


df = load_data(STUDY_CSV)
pipe = load_model(MODEL_PKL)


# ----------------- HELPERS -----------------

def sample_trials(n=18):
    n = max(1, min(int(n), len(df)))
    idxs = list(range(len(df)))
    random.shuffle(idxs)
    return idxs[:n]


def survey_code():
    return "VS-A3-" + uuid.uuid4().hex[:6].upper()


def write_csv(path, row, header):
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(h, "")) for h in header) + "\n")


def init_state():
    """Initialize all session_state variables we need."""
    ss = st.session_state
    if "page" not in ss:
        ss.page = "home"
    if "pid" not in ss:
        ss.pid = ""
    if "cond" not in ss:
        ss.cond = "noai"   # "ai" or "noai"
    if "sync_ai" not in ss:
        ss.sync_ai = False
    if "idxs" not in ss:
        ss.idxs = []
    if "pos" not in ss:
        ss.pos = 0
    if "start_ts" not in ss:
        ss.start_ts = None
    if "trial_start" not in ss:
        ss.trial_start = None
    if "sid" not in ss:
        ss.sid = ""
    if "survey_done" not in ss:
        ss.survey_done = False
    if "code" not in ss:
        ss.code = ""


def layout_shell(body_fn):
    """Mimic base.html: title, topbar, footer, then inner page content."""
    st.markdown(
        f"### {APP_TITLE}\n"
        "<span style='color:#9ca3af;font-size:13px;'>Target time ≤ 10 minutes</span>",
        unsafe_allow_html=True,
    )
    body_fn()
    st.markdown(
        "<hr style='opacity:0.3;'>"
        "<span style='color:#9ca3af;font-size:12px;'>"
        "All texts are fictional and safe. You can select <b>Not sure</b> if a message feels unclear."
        "</span>",
        unsafe_allow_html=True,
    )


# ----------------- PAGES -----------------

def page_home():
    def body():
        st.markdown(
            """
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:18px;">
              <h4>Welcome</h4>
              <p style="color:#9ca3af;font-size:14px;">
                You will see short chat messages and judge them. In some cases,
                you may also see an AI suggestion before answering.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Query param support similar to Flask version (pid, cond, n, sync)
        try:
            q = st.query_params
        except AttributeError:
            q = st.experimental_get_query_params()

        preset_pid = q.get("pid", [""])[0] if q else ""
        preset_cond = q.get("cond", [""])[0] if q else ""
        preset_n = int(q.get("n", ["18"])[0]) if q else 18
        preset_sync = q.get("sync", ["0"])[0] == "1" if q else False

        with st.form("home_form"):
            pid = st.text_input(
                "Participant ID (optional)",
                value=st.session_state.pid or preset_pid,
            )

            cond_options = ["noai", "ai"]
            if preset_cond in cond_options:
                default_index = cond_options.index(preset_cond)
            else:
                default_index = 0 if st.session_state.cond == "noai" else 1

            cond = st.selectbox(
                "Condition",
                cond_options,
                index=default_index,
                help="noai = no AI suggestion, ai = with AI suggestion",
            )

            n_trials = st.number_input(
                "Number of trials",
                min_value=1,
                max_value=len(df),
                value=preset_n,
                step=1,
            )

            sync_ai = st.checkbox(
                "Use live AI model (if available)",
                value=st.session_state.sync_ai or preset_sync,
                help="If checked and model is available, use it instead of pre-computed labels.",
            )

            start_btn = st.form_submit_button("Start")

        if start_btn:
            st.session_state.pid = pid.strip() or "anon"
            st.session_state.cond = cond
            st.session_state.sync_ai = sync_ai
            st.session_state.idxs = sample_trials(n_trials)
            st.session_state.pos = 0
            st.session_state.start_ts = time.time()
            st.session_state.trial_start = time.time()
            st.session_state.sid = uuid.uuid4().hex[:12]
            st.session_state.survey_done = False
            st.session_state.code = ""
            st.session_state.page = "consent"
            st.rerun()

    layout_shell(body)


def page_consent():
    def body():
        st.markdown(
            """
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:18px;">
              <h3>Instructions</h3>
              <ul>
                <li>Read the message.</li>
                <li>If in <b>With AI</b>, you will also see the AI label, its confidence, and a one line rationale.</li>
                <li>Pick your answer. You can think of keys 1 / 2 / 3 as quick shortcuts for the options.</li>
                <li>There is a <b>Not sure</b> option for edge cases.</li>
              </ul>
              <p style="color:#9ca3af;font-size:14px;">
                This study shows fictional and non explicit chat lines. Time target is under 10 minutes.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.idxs:
            st.info("Session not initialized, returning to start.")
            if st.button("Back to start"):
                st.session_state.page = "home"
                st.rerun()
            return

        with st.form("consent_form"):
            agree = st.checkbox("I understand and want to continue", value=False)
            btn = st.form_submit_button("Start")

        if btn:
            if not agree:
                st.warning("Please check the box to continue.")
            else:
                st.session_state.page = "task"
                st.session_state.trial_start = time.time()
                st.rerun()

    layout_shell(body)


def page_task():
    def body():
        if not st.session_state.idxs:
            st.session_state.page = "home"
            st.rerun()

        pos = st.session_state.pos
        idxs = st.session_state.idxs
        total = len(idxs)

        if pos >= total:
            st.session_state.page = "survey"
            st.rerun()

        idx = idxs[pos]
        row = df.iloc[idx]

        if st.session_state.trial_start is None:
            st.session_state.trial_start = time.time()

        # Progress text + bar (like Trial X of Y + horizontal bar)
        current_trial = pos + 1
        st.markdown(
            f"<span style='color:#9ca3af;font-size:13px;'>Trial {current_trial} of {total}</span>",
            unsafe_allow_html=True,
        )
        st.progress(current_trial / total)

        # Message card
        st.markdown(
            f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:18px;margin-top:10px;">
              <div style="color:#9ca3af;font-size:13px;">Message</div>
              <div style="margin-top:6px;font-size:18px;
                          background:#0b1220;border-radius:10px;
                          border:1px dashed #1f2937;padding:14px;">
                {row["input_text"]}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI suggestion (if cond == "ai")
        ai_info = None
        if st.session_state.cond == "ai":
            if st.session_state.sync_ai and pipe is not None:
                try:
                    prob = float(pipe.predict_proba([row["input_text"]])[:, 1][0])
                    label = "CONSENSUAL_FLIRT" if prob >= 0.5 else "TOO_FORWARD"
                    ai_info = {
                        "label": label,
                        "score": prob,
                        "rationale": row.get("ai_rationale", "lexical pattern features"),
                    }
                except Exception:
                    ai_info = {
                        "label": row["ai_readable_label"],
                        "score": row["model_score"],
                        "rationale": row["ai_rationale"],
                    }
            else:
                ai_info = {
                    "label": row["ai_readable_label"],
                    "score": row["model_score"],
                    "rationale": row["ai_rationale"],
                }

        if ai_info is not None:
            st.markdown(
                f"""
                <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:18px;margin-top:10px;">
                  <div style="color:#9ca3af;font-size:13px;">AI suggestion</div>
                  <div>Label: <b style="color:#22d3ee">{ai_info['label']}</b></div>
                  <div>Confidence: <b>{ai_info['score']:.2f}</b></div>
                  <div>Rationale: <span style="color:#9ca3af;">{ai_info['rationale']}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Answer form (options differ by condition, same values as Flask)
        with st.form(f"trial_form_{pos}"):
            st.markdown(
                "<div style='color:#9ca3af;font-size:13px;margin-top:10px;'>"
                "Your answer  –  think of shortcuts 1, 2, 3"
                "</div>",
                unsafe_allow_html=True,
            )

            if st.session_state.cond == "ai":
                options = {
                    "I agree": "agree",
                    "I disagree": "disagree_too_forward",
                    "Not sure": "not_sure",
                }
            else:
                options = {
                    "Consensual Flirt": "consensual",
                    "Too Forward": "too_forward",
                    "Not sure": "not_sure",
                }

            label_list = list(options.keys())
            # Pick default selection index 0
            choice_label = st.radio(
                "Select one:",
                label_list,
                index=0,
                key=f"answer_choice_{pos}",
            )
            submit = st.form_submit_button("Next")

        if submit:
            answer_value = options[choice_label]
            rt = time.time() - float(st.session_state.trial_start or time.time())

            header = [
                "sid",
                "pid",
                "cond",
                "sync_ai",
                "trial_index",
                "message_index",
                "input_text",
                "ai_label",
                "ai_score",
                "ai_rationale",
                "answer",
                "response_time_ms",
                "ts",
                "user_agent",
            ]
            row_log = {
                "sid": st.session_state.sid,
                "pid": st.session_state.pid,
                "cond": st.session_state.cond,
                "sync_ai": int(bool(st.session_state.sync_ai)),
                "trial_index": pos,
                "message_index": idx,
                "input_text": str(row["input_text"]).replace(",", " "),
                "ai_label": ai_info["label"] if ai_info else "",
                "ai_score": ai_info["score"] if ai_info else "",
                "ai_rationale": ai_info["rationale"] if ai_info else "",
                "answer": answer_value,
                "response_time_ms": int(rt * 1000),
                "ts": datetime.utcnow().isoformat(),
                "user_agent": "streamlit",
            }
            write_csv(LOG_TRIALS, row_log, header)

            # Move to next trial
            st.session_state.pos += 1
            st.session_state.trial_start = time.time()
            st.rerun()

    layout_shell(body)


def page_survey():
    def body():
        if not st.session_state.idxs:
            st.session_state.page = "home"
            st.rerun()

        st.markdown(
            """
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:18px;">
              <h3>Quick survey</h3>
              <p>How confident did you feel overall?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("survey_form"):
            # mimic 1–5 radio with labels
            conf_label = st.radio(
                "Confidence (1–5):",
                [
                    "1 not confident",
                    "2",
                    "3",
                    "4",
                    "5 very confident",
                ],
            )
            submit = st.form_submit_button("Finish")

        if submit:
            # Convert label back to numeric "1"–"5"
            conf_value = conf_label.split()[0]  # first token is "1", "2", "3", ...
            code = survey_code()

            header = [
                "sid",
                "pid",
                "cond",
                "sync_ai",
                "n_trials",
                "study_minutes",
                "confidence_1to5",
                "code",
                "ts",
            ]
            study_minutes = round(
                (time.time() - st.session_state.start_ts) / 60.0, 2
            )
            row_log = {
                "sid": st.session_state.sid,
                "pid": st.session_state.pid,
                "cond": st.session_state.cond,
                "sync_ai": int(bool(st.session_state.sync_ai)),
                "n_trials": len(st.session_state.idxs),
                "study_minutes": study_minutes,
                "confidence_1to5": conf_value,
                "code": code,
                "ts": datetime.utcnow().isoformat(),
            }
            write_csv(LOG_SESS, row_log, header)

            st.session_state.survey_done = True
            st.session_state.code = code
            st.session_state.page = "finish"
            st.rerun()

    layout_shell(body)


def page_finish():
    def body():
        if not st.session_state.survey_done:
            st.warning("You have not completed the survey yet.")
            if st.button("Back to start"):
                st.session_state.page = "home"
                st.rerun()
            return

        st.markdown(
            f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:14px;padding:18px;">
              <h3>Thank you</h3>
              <p>Your survey code</p>
              <div style="margin-top:8px;
                          font-size:18px;
                          background:#0b1220;border-radius:10px;
                          border:1px dashed #1f2937;padding:14px;">
                <b>{st.session_state.code or "VS-A3-XXXXXX"}</b>
              </div>
              <p style="margin-top:10px;">
                Paste this code in MTurk Sandbox to submit the HIT.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_state()
            st.session_state.page = "home"
            st.rerun()

    layout_shell(body)


# ----------------- MAIN -----------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")
    init_state()

    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "consent":
        page_consent()
    elif page == "task":
        page_task()
    elif page == "survey":
        page_survey()
    elif page == "finish":
        page_finish()
    else:
        st.session_state.page = "home"
        page_home()


if __name__ == "__main__":
    main()
